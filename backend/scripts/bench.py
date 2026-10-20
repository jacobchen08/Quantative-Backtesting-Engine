r"""Measured timings for the §14 performance targets (CLAUDE.md).

This script is the only source for runtime claims (§0, rule 4). It prints the hardware and the
Python and library versions, then:
- each §14 workload of the phases built so far, against its target;
- the 15-year pairs backtest broken down by stage;
- the default walk-forward on the same pair, as a request and as run_walk_forward alone;
- the same backtest repeated with only z_entry or only L changed, the case the diagnostics
  cache (§14) is for;
- the Kalman pairs model (§5.7) on the same pair, and a ten-ETF trend portfolio (§5.8);
- the Monte Carlo time per model, with a fixed and an automatic block length, and the peak
  memory of one run at the API caps.

Run it from backend\ in PowerShell:

    .venv\Scripts\python.exe scripts\bench.py
    .venv\Scripts\python.exe scripts\bench.py --repeats 30

It keeps its own SQLite price cache, backend\bench.db (git-ignored), so it never touches the
app's database or its run history. The first run downloads KO, PEP, SPY, ^IRX and the trend
portfolio's ETFs from Yahoo Finance; after that every request must be served from the cache,
and the bench stops if one reaches the network.

Requests go through FastAPI's TestClient, in process. The times include request validation,
the engine, storing the run, response validation and JSON encoding, but no network or browser.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import gc
import importlib.metadata
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))  # run as a script from backend\, so make the app package importable

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app import __version__
from app.database import Base, get_session, make_engine
from app.diagnostics_cache import DiagnosticsCache
from app.main import app, diagnostics_cache
from app.schemas import BacktestRequest, BacktestResponse, WalkForwardRequest
from app.serialization import backtest_payload
from app.services import data_loader, monte_carlo, pipeline, stats
from app.services.pipeline import BacktestReport

BENCH_DB = BACKEND / "bench.db"
PAIR = ["KO", "PEP"]
START, END = "2011-01-01", "2025-12-31"  # 15 calendar years
SEED = 20261016
BACKTEST = {"strategy": "pairs", "tickers": PAIR, "start": START, "end": END}  # every other field default
KALMAN = {**BACKTEST, "pairs_model": "kalman"}  # the default delta, 1e-6
DIVERSIFIED = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]  # the §13 preset
PORTFOLIO = {"strategy": "trend_portfolio", "tickers": DIVERSIFIED, "start": START, "end": END}
MONTE_CARLO = {"source": "strategy", "backtest": BACKTEST, "seed": SEED}  # 10,000 paths x 252 days
CAPS = {"n_paths": 50_000, "horizon": 253}  # n_paths x (horizon + 1) = 12,700,000, the API cap
# The L sweep: every 4th L from 20 to 252, skipping the default 60. Under the default split the split
# date moves about 0.3 trading days per unit of L, so a change of 4 or more always moves it, and every
# request in the sweep misses the diagnostics cache.
L_SWEEP = [L for L in range(20, 253, 4) if L != 60]
DEFAULT_REPEATS = 10
LIBRARIES = (
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "arch",
    "fastapi",
    "starlette",
    "pydantic",
    "SQLAlchemy",
)


# ---------------------------------------------------------------- machine description


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            command = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown CPU"


def ram_gib() -> float | None:
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus(dwLength=ctypes.sizeof(MemoryStatus))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / 2**30
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (AttributeError, ValueError, OSError):
        return None


def describe_machine() -> list[str]:
    ram = ram_gib()
    memory = f"{ram:.1f} GiB" if ram is not None else "unknown"
    versions = ", ".join(f"{name} {importlib.metadata.version(name)}" for name in LIBRARIES)
    return [
        f"- CPU: {cpu_name()}, {os.cpu_count()} logical cores; RAM: {memory}",
        f"- OS: {platform.platform()}; Python {platform.python_version()} "
        f"({platform.python_implementation()})",
        f"- Libraries: {versions}",
    ]


# ---------------------------------------------------------------- timing and formatting


def timed(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[float, Any]:
    """(wall time in ms, result) of one call, timed with time.perf_counter()."""
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return (time.perf_counter() - start) * 1000, value


def repeat(
    fn: Callable[[], object], repeats: int, *, before: Callable[[], object] | None = None
) -> list[float]:
    """One untimed warm-up call, then `repeats` timed calls; before() runs untimed before each."""
    gc.collect()
    times = []
    for i in range(repeats + 1):
        if before is not None:
            before()
        elapsed, _ = timed(fn)
        if i > 0:
            times.append(elapsed)
    return times


def ms(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f} ms"
    if value >= 10:
        return f"{value:.1f} ms"
    return f"{value:.2f} ms"


def row(label: str, times: list[float], target_ms: float | None = None, extra: str | None = None) -> str:
    """One Markdown table row: label, median, min, max, then the target and whether the median meets it."""
    median = statistics.median(times)
    cells = [label, ms(median), ms(min(times)), ms(max(times))]
    if target_ms is not None:
        target = f"< {target_ms / 1000:g} s" if target_ms >= 1000 else f"< {target_ms:g} ms"
        cells += [target, "yes" if median < target_ms else "**no**"]
    if extra is not None:
        cells.append(extra)
    return "| " + " | ".join(cells) + " |"


def table(*columns: str) -> list[str]:
    """A Markdown table header; every column after the first is right-aligned."""
    return ["| " + " | ".join(columns) + " |", "|---|" + "---:|" * (len(columns) - 1)]


# ---------------------------------------------------------------- the API, on the bench's own database


class NetworkGuard:
    """Replaces data_loader.fetch_from_yahoo: records downloads, and refuses them once armed."""

    def __init__(self) -> None:
        self.real = data_loader.fetch_from_yahoo
        self.downloaded: list[str] = []
        self.armed = False

    def __call__(self, ticker: str, start: dt.date, end_inclusive: dt.date) -> object:
        if self.armed:
            raise RuntimeError(f"the bench reached Yahoo for {ticker} after the warm-up: the cache was cold")
        self.downloaded.append(ticker)
        return self.real(ticker, start, end_inclusive)


def bench_client() -> tuple[TestClient, sessionmaker[Session]]:
    """A TestClient whose requests use bench.db, and a session factory for it.

    The app's lifespan, which would create the tables in the app's own database, doesn't run;
    the tables are created here instead.
    """
    engine = make_engine(f"sqlite:///{BENCH_DB.as_posix()}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override() -> Iterator[Session]:
        with factory() as session:
            yield session

    app.dependency_overrides[get_session] = override
    return TestClient(app), factory


def post(client: TestClient, path: str, body: dict) -> Any:
    """POST and check for HTTP 200. The body is left unparsed, so timings exclude client-side parsing."""
    response = client.post(path, json=body)
    if response.status_code != 200:
        raise SystemExit(f"{path} returned HTTP {response.status_code}: {response.text}")
    return response


def serialize(report: BacktestReport, resolved: dict, warnings: list[str]) -> str:
    """What the /api/backtest route does after build_report: payload, response validation, JSON."""
    body = {"run_id": 1, "engine_version": __version__, **backtest_payload(report, resolved, warnings)}
    return BacktestResponse.model_validate(body).model_dump_json()


def constant(value: object) -> Callable[..., object]:
    return lambda *_args: value


# ---------------------------------------------------------------- workloads


def stage_breakdown(factory: sessionmaker[Session], repeats: int) -> tuple[dict[str, list[float]], int]:
    """The /api/backtest steps called one at a time, in the route's order (§14).

    Returns the times per stage and the number of days on the run calendar.
    """
    request = BacktestRequest.model_validate(BACKTEST)
    params = request.to_params()
    resolved = {**request.resolved(), "rf_source": request.rf_source()}
    warm = DiagnosticsCache()  # will hold this pair and window, so every lookup in it is a hit
    names = ("load", "run", "diag_miss", "diag_hit", "report", "serialize")
    stages: dict[str, list[float]] = {name: [] for name in names}
    n_days = 0
    gc.collect()
    for i in range(repeats + 1):  # the first pass is the warm-up
        with factory() as session:
            t_load, market = timed(
                data_loader.load_market,
                session,
                request.tickers,
                request.start,
                request.end,
                request.risk_free,
            )
        t_run, run = timed(pipeline.run_strategy, market.prices, market.rf, params)
        y, x = run.prices[PAIR[0]], run.prices[PAIR[1]]
        t_miss, diagnostics = timed(stats.pairs_diagnostics, y, x, run.split_date)
        warm.get(y, x, run.split_date)
        t_hit, _ = timed(warm.get, y, x, run.split_date)
        t_report, report = timed(pipeline.build_report, run, pairs_diagnostics=constant(diagnostics))
        t_serialize, _ = timed(serialize, report, resolved, market.warnings)
        if i > 0:
            for name, value in zip(names, (t_load, t_run, t_miss, t_hit, t_report, t_serialize), strict=True):
                stages[name].append(value)
        n_days = len(market.prices)
    return stages, n_days


def parameter_iteration(
    client: TestClient, repeats: int, change: Callable[[int], dict]
) -> tuple[list[float], str]:
    """The backtest repeated, each request differing from the one before only by change(i).

    It starts from the run with every default, whose diagnostics are then cached, as when a
    user iterates on a result. Returns the times and the cache's hits and misses during them.
    """
    diagnostics_cache.clear()
    post(client, "/api/backtest", BACKTEST)
    post(client, "/api/backtest", {**BACKTEST, **change(0)})  # warm-up
    hits, misses = diagnostics_cache.hits, diagnostics_cache.misses
    gc.collect()
    times = [
        timed(post, client, "/api/backtest", {**BACKTEST, **change(i)})[0] for i in range(1, repeats + 1)
    ]
    return times, f"{diagnostics_cache.hits - hits} hits, {diagnostics_cache.misses - misses} misses"


def walk_forward_alone(factory: sessionmaker[Session], repeats: int) -> tuple[list[float], dict[str, int]]:
    """pipeline.run_walk_forward on the loaded 15-year data, without loading or serialization.

    Returns the times and the shape of one result: folds, configurations and distinct choices.
    """
    request = WalkForwardRequest.model_validate(BACKTEST)
    with factory() as session:
        market = data_loader.load_market(
            session, request.tickers, request.start, request.end, request.risk_free
        )
    params = request.to_params()
    times = repeat(lambda: pipeline.run_walk_forward(market.prices, market.rf, params), repeats)
    result = pipeline.run_walk_forward(market.prices, market.rf, params)
    shape = {
        "folds": len(result.folds),
        "configs": result.n_configs,
        "distinct": result.n_distinct_selected,
        "days": result.metrics.n_days,
    }
    return times, shape


def run_strategy_alone(factory: sessionmaker[Session], body: dict, repeats: int) -> list[float]:
    """pipeline.run_strategy on loaded data, without the report or serialization."""
    request = BacktestRequest.model_validate(body)
    with factory() as session:
        market = data_loader.load_market(
            session, request.tickers, request.start, request.end, request.risk_free
        )
    params = request.to_params()
    return repeat(lambda: pipeline.run_strategy(market.prices, market.rf, params), repeats)


def monte_carlo_peak_mb(factory: sessionmaker[Session]) -> float:
    """Peak traced memory (MB, 10^6 bytes) of one simulate() call at the API caps (§10.6, §14),
    with all three models.

    tracemalloc sees numpy's buffers, so this is the arrays' peak, not the process's RSS.
    """
    request = BacktestRequest.model_validate(BACKTEST)
    with factory() as session:
        market = data_loader.load_market(
            session, request.tickers, request.start, request.end, request.risk_free
        )
    run = pipeline.run_strategy(market.prices, market.rf, request.to_params())
    returns, exposed = pipeline.strategy_mc_returns(run, "full")
    gc.collect()
    tracemalloc.start()
    try:
        monte_carlo.simulate(returns, block_len=10, seed=SEED, has_exposure=exposed, **CAPS)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return peak / 1e6


def monte_carlo_runs(
    client: TestClient, body: dict, repeats: int
) -> tuple[list[float], dict[str, list[float]], dict]:
    """The Monte Carlo request timed end to end, simulate()'s own timers, and the last response."""
    request_times: list[float] = []
    inside: dict[str, list[float]] = {"total": [], "gbm": [], "bootstrap": [], "stationary": []}
    out: dict = {}
    gc.collect()
    for i in range(repeats + 1):  # the first run is the warm-up
        elapsed, response = timed(post, client, "/api/monte-carlo", body)
        if i > 0:
            out = response.json()
            request_times.append(elapsed)
            inside["total"].append(out["elapsed_ms_total"])
            for model in ("gbm", "bootstrap", "stationary"):
                inside[model].append(out[model]["elapsed_ms"])
    return request_times, inside, out


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the CLAUDE.md section 14 performance targets.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"timed runs per workload after one warm-up (1-50; default {DEFAULT_REPEATS})",
    )
    repeats = parser.parse_args().repeats
    if not 1 <= repeats <= 50:
        parser.error("--repeats must be between 1 and 50")

    for name in ("httpx", "httpx2"):  # the TestClient's HTTP library logs every request at INFO
        logging.getLogger(name).setLevel(logging.WARNING)
    guard = NetworkGuard()
    data_loader.fetch_from_yahoo = guard
    client, factory = bench_client()
    first = post(client, "/api/backtest", BACKTEST).json()  # downloads into bench.db on the first run only
    portfolio_first = post(client, "/api/backtest", PORTFOLIO).json()  # likewise, for the ten ETFs
    guard.armed = True
    split_date = first["resolved"]["split_date"]
    print(f"Measuring ({repeats} runs per workload)...", file=sys.stderr)

    cold = repeat(lambda: post(client, "/api/backtest", BACKTEST), repeats, before=diagnostics_cache.clear)
    stages, n_days = stage_breakdown(factory, repeats)
    l_label = f"L ({L_SWEEP[1]}, {L_SWEEP[2]}, ...)"  # the timed requests are i = 1, 2, ...
    iteration = [
        (
            "z_entry (1.55, 1.60, ...)",
            parameter_iteration(client, repeats, lambda i: {"z_entry": 1.5 + 0.05 * i}),
        ),
        (
            f"{l_label}, default 70% split",
            parameter_iteration(client, repeats, lambda i: {"L": L_SWEEP[i]}),
        ),
        (
            f"{l_label}, split_date fixed",
            parameter_iteration(client, repeats, lambda i: {"L": L_SWEEP[i], "split_date": split_date}),
        ),
    ]
    sensitivity = repeat(lambda: post(client, "/api/sensitivity", BACKTEST), repeats)
    walk_forward = repeat(lambda: post(client, "/api/walk-forward", BACKTEST), repeats)
    walk_forward_core, wf_shape = walk_forward_alone(factory, repeats)
    kalman = repeat(lambda: post(client, "/api/backtest", KALMAN), repeats, before=diagnostics_cache.clear)
    run_ols = run_strategy_alone(factory, BACKTEST, repeats)
    run_kalman = run_strategy_alone(factory, KALMAN, repeats)
    portfolio = repeat(lambda: post(client, "/api/backtest", PORTFOLIO), repeats)
    run_portfolio = run_strategy_alone(factory, PORTFOLIO, repeats)
    mc_request, inside, _ = monte_carlo_runs(client, MONTE_CARLO, repeats)
    two_models = [g + b for g, b in zip(inside["gbm"], inside["bootstrap"], strict=True)]
    mc_auto, inside_auto, auto_out = monte_carlo_runs(client, {**MONTE_CARLO, "block_len": "auto"}, repeats)
    auto_used = auto_out["resolved"]["block_len_used"]
    peak_mb = monte_carlo_peak_mb(factory)
    path_mb = CAPS["n_paths"] * (CAPS["horizon"] + 1) * 8 / 1e6
    p_res = portfolio_first["resolved"]
    p_days = len(portfolio_first["series"]["dates"])

    stage_rows = [
        ("Load prices and rf from the cache", stages["load"]),
        ("run_strategy (section 5.0, steps 1-6)", stages["run"]),
        ("Diagnostics: two Engle-Granger tests, static OLS, half-life", stages["diag_miss"]),
        ("Report: metrics, benchmark, trades, warnings, series", stages["report"]),
        ("Serialization: payload, response model, JSON", stages["serialize"]),
    ]
    stage_sum = [sum(run) for run in zip(*(times for _, times in stage_rows), strict=True)]
    downloads = (
        f"downloaded {', '.join(guard.downloaded)} on this run" if guard.downloaded else "no downloads"
    )
    lines = [
        f"# Quant Engine bench: engine {__version__}, {dt.datetime.now():%Y-%m-%d %H:%M}",
        "",
        *describe_machine(),
        f"- Data: {PAIR[0]}/{PAIR[1]} pairs, every other setting default, {START} to {END}: "
        f"{n_days:,} trading days, from bench.db ({downloads})",
        f"- Each row: {repeats} timed runs after one warm-up run, wall clock (time.perf_counter)",
        "",
        "## Section 14 targets",
        "",
        *table("Workload", "Median", "Min", "Max", "Target", "Met"),
        row("15-year pairs backtest, first request on the data (diagnostics computed)", cold, 300),
        row("15-year pairs backtest, z_entry changed (diagnostics cached)", iteration[0][1][0], 300),
        row("Default sensitivity grid (20 configurations)", sensitivity, 5000),
        row("Default walk-forward, 15-year pairs (20 configurations)", walk_forward, 8000),
        row(
            "Default Monte Carlo, 10,000 x 252: GBM and block bootstrap (their elapsed_ms)", two_models, 1000
        ),
        row("Default Monte Carlo, three models: the request", mc_request, 1500),
        row(
            "Default Monte Carlo, three models: the simulation alone (elapsed_ms_total)",
            inside["total"],
            1500,
        ),
        row("15-year Kalman pairs backtest, first request on the data (diagnostics computed)", kalman, 500),
        "",
        "## Pairs backtest by stage (first request on the data)",
        "",
        *table("Stage", "Median", "Min", "Max"),
        *(row(label, times) for label, times in stage_rows),
        row("Sum of the stages", stage_sum),
        row("Diagnostics from the cache instead (key and lookup)", stages["diag_hit"]),
        "",
        f"The whole request took {ms(statistics.median(cold))} (median, first table). The rest is",
        "parsing and validating the request, storing the run in backtest_runs, FastAPI's own response",
        "handling and the TestClient round trip.",
        "",
        "## Parameter iteration: each request changes one field of the one before",
        "",
        *table("Field changed", "Median", "Min", "Max", "Diagnostics cache"),
        *(row(label, times, extra=cache) for label, (times, cache) in iteration),
        "",
        "## Walk-forward (default: 756 training and 63 test days, rolling)",
        "",
        *table("Stage", "Median", "Min", "Max"),
        row("The request (as in the first table)", walk_forward),
        row("run_walk_forward alone (no loading or serialization)", walk_forward_core),
        "",
        f"It backtests {wf_shape['configs']} grid configurations plus the request's own, then makes "
        f"{wf_shape['folds']} folds: {wf_shape['days']:,} out-of-sample days, on which "
        f"{wf_shape['distinct']} distinct configurations were chosen.",
        "",
        "## Kalman pairs and the trend portfolio",
        "",
        *table("Workload", "Median", "Min", "Max"),
        row("run_strategy, rolling OLS (the default pairs model)", run_ols),
        row("run_strategy, Kalman filter (delta 1e-6)", run_kalman),
        row(f"Trend portfolio of {len(DIVERSIFIED)} ETFs: the request", portfolio),
        row(f"Trend portfolio of {len(DIVERSIFIED)} ETFs: run_strategy alone", run_portfolio),
        "",
        f"The trend portfolio ({', '.join(DIVERSIFIED)}; L = {p_res['L']}, a {p_res['vol_target']:.0%} "
        f"target per asset) runs on the intersection of their calendars: {p_days:,} series days from "
        f"{portfolio_first['series']['dates'][0]}, {len(portfolio_first['trades']):,} leg trades.",
        "",
        "## Monte Carlo",
        "",
        *table("Model (time inside the default request)", "Median", "Min", "Max"),
        row("GBM", inside["gbm"]),
        row("Circular block bootstrap (block length 10)", inside["bootstrap"]),
        row("Stationary bootstrap (mean block length 10)", inside["stationary"]),
        row('The request with block_len "auto"', mc_auto),
        row('The simulation alone with block_len "auto" (elapsed_ms_total)', inside_auto["total"]),
        "",
        f'With block_len "auto" the block lengths were {auto_used["circular"]} (circular) and '
        f"{auto_used['stationary']} (stationary).",
        "",
        f"Peak traced memory at the API caps ({CAPS['n_paths']:,} paths x {CAPS['horizon']} days, three "
        f"models): {peak_mb:.0f} MB, {peak_mb / path_mb:.2f}x the {path_mb:.0f} MB path array "
        "(tracemalloc, which sees numpy's buffers).",
        "",
        r"Measured with: .venv\Scripts\python.exe scripts\bench.py"
        + ("" if repeats == DEFAULT_REPEATS else f" --repeats {repeats}"),
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
