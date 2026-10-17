"""API tests (§12, test_api.py): endpoints, validation, persistence and error mapping."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session
from yfinance.exceptions import YFRateLimitError

from app import __version__
from app.main import app, diagnostics_cache
from app.models import BacktestRun
from app.schemas import BacktestResponse, MonteCarloResponse, SensitivityResponse
from app.serialization import to_jsonable
from app.services import stats

START, END = "2018-01-01", "2021-12-31"
SMALL_MC = {"n_paths": 1000, "horizon": 21}


def parse(response: Any) -> dict:
    """The JSON body, failing on NaN or Infinity (Python's json accepts them by default)."""

    def reject(token: str) -> None:
        raise AssertionError(f"non-finite number {token} in the response")

    return json.loads(response.text, parse_constant=reject)


def post(client: TestClient, path: str, payload: dict, status: int = 200) -> dict:
    response = client.post(path, json=payload)
    assert response.status_code == status, response.text
    return parse(response)


def test_to_jsonable_makes_engine_output_strict_json() -> None:
    value = {
        "nan": float("nan"),
        "inf": np.float64(np.inf),
        "int": np.int64(3),
        "flag": np.bool_(True),
        "day": pd.Timestamp("2024-01-05"),
        "nat": pd.NaT,
        "array": np.array([1.0, np.nan, -np.inf]),
        "matrix": np.array([[1.0, np.nan], [2.0, 3.0]]),
        "dates": pd.DatetimeIndex(["2024-01-02", None]),
        "counts": pd.Series([1, 2]),
        "pair": (1.5, None),
    }
    out = to_jsonable(value)
    assert out == {
        "nan": None,
        "inf": None,
        "int": 3,
        "flag": True,
        "day": "2024-01-05",
        "nat": None,
        "array": [1.0, None, None],
        "matrix": [[1.0, None], [2.0, 3.0]],
        "dates": ["2024-01-02", None],
        "counts": [1, 2],
        "pair": [1.5, None],
    }
    json.dumps(out, allow_nan=False)  # strict JSON: would raise on NaN or inf
    assert type(out["int"]) is int and type(out["flag"]) is bool


def test_a_nan_in_a_non_nullable_field_fails_loudly(client: TestClient) -> None:
    out = post(client, "/api/backtest", {"strategy": "trend", "tickers": ["SPY"], "start": START, "end": END})
    out["metrics"]["full"]["total_return"] = float("nan")
    # Pydantic accepts NaN as a float and would serialize it as null, silently breaking the contract...
    BacktestResponse.model_validate(out)
    # ...so to_jsonable turns it into None first, which the non-nullable field rejects.
    with pytest.raises(ValidationError):
        BacktestResponse.model_validate(to_jsonable(out))


def test_health() -> None:
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "engine_version": __version__}


# ---------------------------------------------------------------- happy paths


@pytest.mark.parametrize(
    "request_fields",
    [
        {"strategy": "mean_reversion", "tickers": ["Y"]},
        {"strategy": "pairs", "tickers": ["Y", "X"]},  # never mentions SPY; risk_free defaults to "tbill"
        {
            "strategy": "pairs",
            "tickers": ["y", "x"],
            "hedge_mode": "rolling",
            "risk_free": 0.03,
            "borrow_bps": 150.0,
        },
        {"strategy": "trend", "tickers": ["SPY"], "vol_target": 0.10, "vol_refresh": "daily"},
        {"strategy": "mean_reversion", "tickers": ["Y"], "allow_short": False, "borrow_bps": 300.0},
    ],
    ids=["mean-reversion", "pairs-tbill", "pairs-constant-rf-borrow", "trend-overlay", "long-only-borrow"],
)
def test_backtest(client: TestClient, request_fields: dict) -> None:
    out = post(client, "/api/backtest", {**request_fields, "start": START, "end": END})
    BacktestResponse.model_validate(out)
    resolved, series = out["resolved"], out["series"]
    assert out["run_id"] >= 1 and out["engine_version"] == __version__
    assert resolved["tickers"] == [t.upper() for t in request_fields["tickers"]]
    assert resolved["first_evaluated_date"] == series["dates"][series["daily_return"].index(None) + 1]
    assert resolved["rf_source"] == (
        "constant 3.00% a year" if "risk_free" in request_fields else "13-week T-bill (^IRX)"
    )
    lengths = {name: len(values) for name, values in series.items() if values is not None}
    assert len(set(lengths.values())) == 1, lengths
    assert (out["diagnostics"] is not None) == (request_fields["strategy"] == "pairs")
    assert (series["beta"] is not None) == (request_fields["strategy"] == "pairs")
    assert (out["sizing"] is not None) == ("vol_target" in request_fields)
    assert set(out["metrics"]) == set(out["benchmark_metrics"]) == {"is", "oos", "full"}
    assert out["trades"] and all(t["segment"] in ("is", "oos") for t in out["trades"])
    # Phase 5: the borrow fee and long-only switch, the cost sweep, calendar and rolling statistics.
    borrow, allow_short = request_fields.get("borrow_bps", 0.0), request_fields.get("allow_short", True)
    assert (resolved["borrow_bps"], resolved["allow_short"]) == (borrow, allow_short)
    assert (out["metrics"]["full"]["ann_borrow_drag"] > 0) == (borrow > 0 and allow_short)
    assert out["benchmark_metrics"]["full"]["ann_borrow_drag"] is None
    if not allow_short:
        assert all(t["side"] == "long" for t in out["trades"]) and min(series["position"]) == 0
    sweep = out["cost_sensitivity"]
    assert sweep["cost_bps"] == sorted(sweep["cost_bps"]) and resolved["cost_bps"] in sweep["cost_bps"]
    assert all(len(sweep["sharpe"][seg]) == len(sweep["cost_bps"]) for seg in ("is", "oos", "full"))
    own = sweep["cost_bps"].index(resolved["cost_bps"])
    assert sweep["sharpe"]["full"][own] == out["metrics"]["full"]["sharpe"]
    months, years = out["calendar"]["monthly"], out["calendar"]["annual"]
    assert [m["year"] * 12 + m["month"] for m in months] == sorted(
        {m["year"] * 12 + m["month"] for m in months}
    )
    assert [y["year"] for y in years] == sorted({m["year"] for m in months})
    assert any(v is not None for v in series["rolling_sharpe"])  # lengths are checked above


def test_sensitivity(client: TestClient) -> None:
    out = post(
        client, "/api/sensitivity", {"strategy": "pairs", "tickers": ["Y", "X"], "start": START, "end": END}
    )
    SensitivityResponse.model_validate(out)
    assert out["L_values"] == [20, 40, 60, 90, 120] and out["z_entry_values"] == [1.5, 2.0, 2.5, 3.0]
    assert out["n_configs_tested"] == 20 and out["selected"] == {"L": 60, "z_entry": 2.0}
    assert out["window"]["start"] < out["window"]["end"]
    deflated = out["deflated"]
    assert 2 <= deflated["n_trials"] <= 20 and deflated["sr0_star"] > 0
    assert 0 <= deflated["selected_dsr"] <= 1 and 0 <= deflated["best"]["dsr"] <= 1
    best = deflated["best"]
    row, col = out["L_values"].index(best["L"]), out["z_entry_values"].index(best["z_entry"])
    assert (
        out["sharpe"][row][col] == best["sharpe"] == max(v for r in out["sharpe"] for v in r if v is not None)
    )


@pytest.mark.parametrize(
    "request_fields",
    [
        {"source": "ticker", "ticker": "spy", "start": START, "end": END},
        {
            "source": "strategy",
            "backtest": {"strategy": "pairs", "tickers": ["Y", "X"], "start": START, "end": END},
        },
        {
            "source": "strategy",
            "window": "is",
            "zero_drift": True,
            "seed": 7,
            "backtest": {"strategy": "trend", "tickers": ["SPY"], "start": START, "end": END},
        },
    ],
    ids=["ticker", "strategy-full", "strategy-is"],
)
def test_monte_carlo(client: TestClient, request_fields: dict) -> None:
    out = post(client, "/api/monte-carlo", {**request_fields, **SMALL_MC})
    MonteCarloResponse.model_validate(out)
    resolved = out["resolved"]
    assert 0 <= resolved["seed"] < 2**53 and resolved["n_hist"] >= 252
    for model in ("gbm", "bootstrap"):
        assert len(out[model]["sample_paths"]) == 20 and len(out[model]["bands"]["p50"]) == 22
        assert sum(out[model]["hist_counts"]) == 1000
    assert len(out["hist_edges"]) == 51
    if request_fields.get("seed") is not None:
        assert resolved["seed"] == 7 and resolved["mu_daily"] == 0.0  # zero_drift
        again = post(client, "/api/monte-carlo", {**request_fields, **SMALL_MC})
        assert again["gbm"]["bands"] == out["gbm"]["bands"]  # the same seed gives the same output


# ---------------------------------------------------------------- validation (422)


@pytest.mark.parametrize(
    "path, payload, phrase",
    [
        (
            "/api/backtest",
            {"strategy": "pairs", "tickers": ["Y", "X"], "z_entry": 1.0, "z_exit": 1.0},
            "z_exit",
        ),
        ("/api/backtest", {"strategy": "pairs", "tickers": ["Y", "X"], "end": "2018-12-31"}, "in-sample"),
        ("/api/backtest", {"strategy": "trend", "tickers": ["SPY"], "L": 600}, "between 21 and 504"),
        ("/api/backtest", {"strategy": "pairs", "tickers": ["Y", "Y"]}, "distinct"),
        ("/api/backtest", {"strategy": "trend", "tickers": ["SP Y"]}, "invalid ticker"),
        ("/api/backtest", {"strategy": "trend", "tickers": ["SPY"], "z_entri": 2}, "z_entri"),
        ("/api/backtest", {"strategy": "trend", "tickers": ["SPY"], "start": "1989-12-29"}, "1990-01-01"),
        (
            "/api/monte-carlo",
            {"source": "ticker", "ticker": "SPY", "n_paths": 50_000, "horizon": 504},
            "exceeds",
        ),
        ("/api/monte-carlo", {"source": "ticker", "ticker": "SPY", "seed": 2**53}, "seed"),
        ("/api/monte-carlo", {"source": "strategy"}, "backtest is required"),
        ("/api/monte-carlo", {"source": "ticker", "ticker": "SPY", "start": "2021-01-04"}, "at least 252"),
        (
            "/api/backtest",
            {"strategy": "pairs", "tickers": ["Y", "X"], "allow_short": False},
            "pairs needs a short leg, so allow_short=false is not available",
        ),
        ("/api/backtest", {"strategy": "mean_reversion", "tickers": ["Y"], "borrow_bps": 2001}, "borrow_bps"),
        ("/api/backtest", {"strategy": "mean_reversion", "tickers": ["Y"], "borrow_bps": -1}, "borrow_bps"),
    ],
    ids=[
        "z_exit>=z_entry",
        "too-few-days",
        "trend-L-600",
        "same-pair-legs",
        "bad-ticker",
        "unknown-field",
        "start-before-1990",
        "path-cap",
        "seed-2^53",
        "missing-backtest",
        "short-history",
        "pairs-long-only",
        "borrow-over-2000",
        "borrow-negative",
    ],
)
def test_invalid_requests_are_422_with_a_readable_detail(
    client: TestClient, path: str, payload: dict, phrase: str
) -> None:
    dates = (
        {"start": START, "end": END} if path == "/api/backtest" or payload.get("source") == "ticker" else {}
    )
    out = post(client, path, {**dates, **payload}, status=422)
    assert isinstance(out["detail"], str) and phrase in out["detail"], out["detail"]


def test_a_strategy_window_without_exposure_is_rejected(client: TestClient) -> None:
    never = {
        "strategy": "mean_reversion",
        "tickers": ["Y"],
        "L": 20,
        "z_entry": 5.0,
        "start": START,
        "end": END,
    }
    out = post(client, "/api/monte-carlo", {"source": "strategy", "backtest": never, **SMALL_MC}, status=422)
    assert "no position" in out["detail"]


def test_upstream_failures_map_to_404_and_503(client: TestClient, fake_yahoo) -> None:
    fake_yahoo.frames["NODATA"] = fake_yahoo.frames["SPY"].iloc[:0]
    fake_yahoo.errors["BUSY"] = YFRateLimitError()
    base = {"strategy": "trend", "start": START, "end": END}
    out = post(client, "/api/backtest", {**base, "tickers": ["NODATA"]}, status=404)
    assert out["detail"].startswith("No data for NODATA between 2018-01-01 and")
    assert (
        "rate-limited" in post(client, "/api/backtest", {**base, "tickers": ["BUSY"]}, status=503)["detail"]
    )


# ---------------------------------------------------------------- persistence


def count_runs(db_engine: Engine) -> int:
    with Session(db_engine) as s:
        return s.scalar(select(func.count()).select_from(BacktestRun))


def test_only_backtests_are_stored(client: TestClient, db_engine: Engine) -> None:
    request = {"strategy": "pairs", "tickers": ["Y", "X"], "start": START, "end": END}
    out = post(client, "/api/backtest", request)
    assert count_runs(db_engine) == 1
    with Session(db_engine) as s:
        row = s.get(BacktestRun, out["run_id"])
        assert row.engine_version == __version__
        assert json.loads(row.request_json) == out["resolved"]
        stored = json.loads(row.metrics_json)
    assert stored == {"metrics": out["metrics"], "benchmark_metrics": out["benchmark_metrics"]}
    post(client, "/api/sensitivity", request)
    post(client, "/api/monte-carlo", {"source": "strategy", "backtest": request, **SMALL_MC})
    assert count_runs(db_engine) == 1


# ---------------------------------------------------------------- diagnostics cache (§14)


def test_pairs_diagnostics_are_reused_while_their_inputs_are_unchanged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The §14 diagnostics cache: keyed on the legs' data and the split date, nothing else."""
    split_dates = []
    real = stats.pairs_diagnostics

    def counting(y_close: pd.Series, x_close: pd.Series, split_date: pd.Timestamp) -> object:
        split_dates.append(split_date)
        return real(y_close, x_close, split_date)

    monkeypatch.setattr(stats, "pairs_diagnostics", counting)
    base = {"strategy": "pairs", "tickers": ["Y", "X"], "start": START, "end": END}
    first = post(client, "/api/backtest", base)
    for change in ({"z_entry": 2.5}, {"z_exit": 0.0}, {"cost_bps": 50}, {"hedge_mode": "rolling"}):
        assert post(client, "/api/backtest", {**base, **change})["diagnostics"] == first["diagnostics"]
    assert len(split_dates) == 1

    # Under the default 70% split, a new L moves t0 and with it the split date: a new entry.
    moved = post(client, "/api/backtest", {**base, "L": 90})
    assert moved["resolved"]["split_date"] != first["resolved"]["split_date"]
    assert len(split_dates) == 2
    # With split_date fixed, L no longer changes the inputs.
    for L in (40, 120):
        post(client, "/api/backtest", {**base, "L": L, "split_date": first["resolved"]["split_date"]})
    assert len(split_dates) == 2

    # A cached response is identical to a freshly computed one, apart from the run id.
    cached = post(client, "/api/backtest", {**base, "z_entry": 3.0})
    diagnostics_cache.clear()
    fresh = post(client, "/api/backtest", {**base, "z_entry": 3.0})
    assert len(split_dates) == 3
    assert {**cached, "run_id": None} == {**fresh, "run_id": None}


# ---------------------------------------------------------------- ignored fields


@pytest.mark.parametrize(
    "path, payload, field",
    [
        (
            "/api/backtest",
            {"strategy": "mean_reversion", "tickers": ["Y"], "hedge_mode": "rolling"},
            "hedge_mode",
        ),
        ("/api/backtest", {"strategy": "trend", "tickers": ["SPY"], "z_entry": 2.5}, "z_entry"),
        ("/api/backtest", {"strategy": "trend", "tickers": ["SPY"], "vol_refresh": "daily"}, "vol_refresh"),
        ("/api/monte-carlo", {"source": "ticker", "ticker": "SPY", "window": "is", **SMALL_MC}, "window"),
        # Ignored fields aren't validated at all, so even nonsense is accepted:
        ("/api/backtest", {"strategy": "trend", "tickers": ["SPY"], "z_entry": 99, "z_exit": 100}, "z_exit"),
        ("/api/backtest", {"strategy": "pairs", "tickers": ["Y", "X"], "max_leverage": 1000}, "max_leverage"),
    ],
)
def test_fields_that_do_not_apply_are_ignored_and_echoed_as_null(
    client: TestClient, path: str, payload: dict, field: str
) -> None:
    out = post(client, path, {**payload, "start": START, "end": END})
    assert out["resolved"][field] is None


def test_end_is_clamped_to_yesterday_in_new_york(client: TestClient, frozen_today: dt.date) -> None:
    out = post(
        client,
        "/api/backtest",
        {"strategy": "trend", "tickers": ["SPY"], "start": START, "end": "2030-01-01"},
    )
    assert out["resolved"]["end"] == str(frozen_today - dt.timedelta(days=1))


# ---------------------------------------------------------------- trend


def test_trend_defaults(client: TestClient) -> None:
    request = {"strategy": "trend", "tickers": ["SPY"], "start": START, "end": END}
    out = post(client, "/api/backtest", request)
    assert out["resolved"]["L"] == 252 and out["sizing"] is None
    grid = post(client, "/api/sensitivity", {**request, "z_entry_values": [2.0]})  # ignored for trend
    assert grid["z_entry_values"] is None and grid["L_values"] == [21, 63, 126, 252, 378]
    assert len(grid["sharpe"]) == 5 and all(len(row) == 1 for row in grid["sharpe"])


def test_on_756_days_the_trend_backtest_works_but_the_grid_names_L_378(
    client: TestClient, prices: pd.DataFrame
) -> None:
    end = prices.index[755].date()  # the fixture's 756th trading day
    request = {"strategy": "trend", "tickers": ["SPY"], "start": START, "end": str(end)}
    out = post(client, "/api/backtest", request)
    assert out["series"]["dates"][-1] == str(end)
    detail = post(client, "/api/sensitivity", request, status=422)["detail"]
    assert "L = 378" in detail and "drop L = 378" in detail


# ---------------------------------------------------------------- ruin (§12, test_backtester item 9)


def test_a_one_day_oos_segment_after_ruin_still_validates(
    client: TestClient, fake_yahoo, prices: pd.DataFrame, yf_frame
) -> None:
    days = prices.index
    log_r = np.random.default_rng(11).normal(0.001, 0.005, len(days))
    log_r[700] = np.log(0.5)  # a -50% day, held 2x long
    fake_yahoo.frames["CRASH"] = yf_frame(pd.Series(100.0 * np.exp(np.cumsum(log_r)), index=days))
    split = str(days[700].date())
    out = post(
        client,
        "/api/backtest",
        {
            "strategy": "trend",
            "tickers": ["CRASH"],
            "start": START,
            "end": END,
            "signal_frequency": "daily",
            "vol_target": 0.5,
            "max_leverage": 2.0,
            "vol_refresh": "daily",
            "split_date": split,
            "risk_free": 0.02,
        },
    )
    BacktestResponse.model_validate(out)  # every null below is allowed by the response model
    oos = out["metrics"]["oos"]
    assert out["ruined"] and out["ruin_date"] == split == out["resolved"]["split_date"]
    assert oos["n_days"] == 1 and oos["total_return"] == -1.0
    for field in ("ann_vol", "sharpe", "sharpe_ci95", "sortino", "beta_spy", "corr_spy", "cagr"):
        assert oos[field] is None, field
    assert out["series"]["dates"][-1] == split and out["series"]["equity"][-1] == 0.0
    assert any(w.startswith("Ruin on") for w in out["warnings"])
    assert out["cost_sensitivity"] is None  # clipping makes excess non-linear in the cost (§8.1)
    assert out["calendar"]["monthly"][-1]["partial"] and out["calendar"]["annual"][-1]["partial"]


# ---------------------------------------------------------------- Phase 5: a zero borrow fee is the default


def test_a_zero_borrow_fee_is_the_default(client: TestClient) -> None:
    request = {"strategy": "mean_reversion", "tickers": ["Y"], "start": START, "end": END}
    implicit = post(client, "/api/backtest", request)
    explicit = post(client, "/api/backtest", {**request, "borrow_bps": 0, "allow_short": True})
    assert implicit["resolved"]["borrow_bps"] == 0.0 and implicit["resolved"]["allow_short"] is True
    assert {**implicit, "run_id": None} == {**explicit, "run_id": None}
