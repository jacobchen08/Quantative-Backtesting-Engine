"""FastAPI application: lifespan (create tables), CORS, error mapping and routes (§11).

Route handlers are plain `def`: the work is CPU-bound and yfinance blocks, so FastAPI runs
them in its thread pool instead of stalling the event loop.
"""

import datetime as dt
import json
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app import __version__
from app.config import settings
from app.database import Base, engine, get_session
from app.diagnostics_cache import DiagnosticsCache
from app.models import BacktestRun  # importing the models also registers every table on Base
from app.schemas import (
    BacktestRequest,
    BacktestResponse,
    MonteCarloRequest,
    MonteCarloResponse,
    SensitivityRequest,
    SensitivityResponse,
    WalkForwardRequest,
    WalkForwardResponse,
)
from app.serialization import backtest_payload, monte_carlo_payload, sensitivity_payload, walk_forward_payload
from app.services import data_loader, monte_carlo, pipeline
from app.services.backtester import InsufficientDataError
from app.services.data_loader import DataSourceError
from app.services.monte_carlo import MonteCarloInputError

logging.basicConfig(level=settings.log_level)

SessionDep = Annotated[Session, Depends(get_session)]

# The two Engle–Granger tests dominate a warm pairs backtest, and they don't depend on z, costs or
# the hedge mode, so repeated requests on the same data reuse them (§14).
diagnostics_cache = DiagnosticsCache()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="Quant Engine", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- errors: always {"detail": "..."}


def _error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.exception_handler(DataSourceError)
async def _data_source_error(_request: Request, exc: DataSourceError) -> JSONResponse:
    return _error(exc.status_code, exc.detail)  # 404 no data, 502/503 upstream failure


@app.exception_handler(InsufficientDataError)
async def _insufficient_data(_request: Request, exc: InsufficientDataError) -> JSONResponse:
    return _error(422, str(exc))


@app.exception_handler(MonteCarloInputError)
async def _monte_carlo_input(_request: Request, exc: MonteCarloInputError) -> JSONResponse:
    return _error(422, str(exc))


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with one readable string, the same shape as every other error."""
    parts = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        message = str(error.get("msg", "invalid value")).removeprefix("Value error, ")
        parts.append(f"{where}: {message}" if where else message)
    return _error(422, "; ".join(parts))


# ---------------------------------------------------------------- routes


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine_version": __version__}


@app.post("/api/backtest", response_model=BacktestResponse)
def backtest(request: BacktestRequest, session: SessionDep) -> dict[str, Any]:
    """Run one backtest and store it in backtest_runs."""
    market = data_loader.load_market(session, request.tickers, request.start, request.end, request.risk_free)
    run = pipeline.run_strategy(market.prices, market.rf, request.to_params())
    report = pipeline.build_report(run, pairs_diagnostics=diagnostics_cache.get, end=request.end)
    payload = backtest_payload(
        report, {**request.resolved(), "rf_source": request.rf_source()}, market.warnings
    )
    record = BacktestRun(
        created_at=dt.datetime.now(dt.UTC),
        engine_version=__version__,
        request_json=json.dumps(payload["resolved"], allow_nan=False),
        metrics_json=json.dumps(
            {"metrics": payload["metrics"], "benchmark_metrics": payload["benchmark_metrics"]},
            allow_nan=False,
        ),
    )
    session.add(record)
    session.commit()
    return {"run_id": record.id, "engine_version": __version__, **payload}


@app.post("/api/sensitivity", response_model=SensitivityResponse)
def sensitivity(request: SensitivityRequest, session: SessionDep) -> dict[str, Any]:
    """In-sample Sharpe ratios over the parameter grid, on one common window (§9). Not stored."""
    market = data_loader.load_market(session, request.tickers, request.start, request.end, request.risk_free)
    result = pipeline.run_sensitivity(
        market.prices,
        market.rf,
        request.to_params(),
        L_values=None if request.L_values is None else tuple(request.L_values),
        z_entry_values=None if request.z_entry_values is None else tuple(request.z_entry_values),
    )
    return sensitivity_payload(result)


@app.post("/api/walk-forward", response_model=WalkForwardResponse)
def walk_forward(request: WalkForwardRequest, session: SessionDep) -> dict[str, Any]:
    """Walk-forward re-optimization over the parameter grid (§9.2). Not stored."""
    market = data_loader.load_market(session, request.tickers, request.start, request.end, request.risk_free)
    result = pipeline.run_walk_forward(
        market.prices,
        market.rf,
        request.to_params(),
        L_values=None if request.L_values is None else tuple(request.L_values),
        z_entry_values=None if request.z_entry_values is None else tuple(request.z_entry_values),
        train_days=request.train_days,
        test_days=request.test_days,
        anchored=request.anchored,
    )
    return walk_forward_payload(result, request.resolved(), market.warnings)


@app.post("/api/monte-carlo", response_model=MonteCarloResponse)
def run_monte_carlo(request: MonteCarloRequest, session: SessionDep) -> dict[str, Any]:
    """GBM, block-bootstrap and stationary-bootstrap projections of a strategy's or a ticker's daily
    returns (§10). Not stored."""
    if request.source == "strategy":
        bt = request.backtest
        market = data_loader.load_market(session, bt.tickers, bt.start, bt.end, bt.risk_free)
        run = pipeline.run_strategy(market.prices, market.rf, bt.to_params())
        returns, has_exposure = pipeline.strategy_mc_returns(run, request.window)
    else:
        close, _ = data_loader.load_close(session, request.ticker, request.start, request.end)
        returns, has_exposure = pipeline.ticker_returns(close), True
    seed = request.seed if request.seed is not None else secrets.randbelow(monte_carlo.MAX_SEED)
    result = monte_carlo.simulate(
        returns,
        horizon=request.horizon,
        n_paths=request.n_paths,
        block_len=request.block_len,
        seed=seed,
        zero_drift=request.zero_drift,
        has_exposure=has_exposure,
    )
    return monte_carlo_payload(result, request.resolved())
