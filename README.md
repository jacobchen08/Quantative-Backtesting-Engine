# Quant Engine

A backtesting and Monte Carlo risk engine for daily US equity strategies, with a React dashboard. It runs single-asset mean reversion, pairs trading and time-series momentum, each with an optional volatility-targeting overlay. Results come split in-sample and out-of-sample, with conservative accounting: a one-day execution lag by default, per-leg trading costs, and cash at the T-bill rate. A parameter grid and Monte Carlo projections test how much a result can be trusted. The projections use GBM and a block bootstrap, checked against closed-form VaR and CVaR.

- **Specification:** [CLAUDE.md](CLAUDE.md), the source of truth for every formula and rule.
- **Methodology:** [docs/METHODOLOGY.md](docs/METHODOLOGY.md) justifies each formula, lists the limitations, and records every approved deviation from the specification.
- **Status:** Phases 0–4 are done: the engine, the API, the dashboard, the documentation and the benchmarks. Phases 5–9 are specified in CLAUDE.md §16 but not built yet.

![The dashboard on a KO/PEP pairs backtest](docs/screenshots/dashboard-pairs.png)

*KO/PEP pairs, 2015-01-02 to 2025-12-31, default settings. More views [below](#screenshots).*

## Setup (PowerShell)

Requires Python 3.12+ and Node 24+.

```powershell
Copy-Item .env.example .env

cd backend
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

cd ..\frontend
npm ci
```

## Run

```powershell
# Terminal 1: API on http://localhost:8000 (interactive docs at /docs)
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --env-file ..\.env

# Terminal 2: dashboard on http://localhost:5173
cd frontend
npm run dev
```

Prices are downloaded from Yahoo Finance on first use and cached in `backend\quant.db`.

## Tests and linters

```powershell
cd backend
.venv\Scripts\ruff.exe check
.venv\Scripts\ruff.exe format --check
.venv\Scripts\python.exe -m pytest

cd ..\frontend
npm run lint
npm test
npm run build
```

The backend tests use synthetic data and a fake network, so they never reach Yahoo.

## Benchmarks

Every runtime figure in this repository comes from `scripts/bench.py`:

```powershell
cd backend
.venv\Scripts\python.exe scripts\bench.py
```

The bench keeps its own price cache, `backend\bench.db`, so it never touches the app's database or run history. Its first run downloads the data, and every timed request after that is served from the cache. Requests go through FastAPI's TestClient in process, so the times include validation, the engine, storing the run and JSON encoding, but no network or browser.

The tables below are copied from its output on 2026-10-16:

- CPU: AMD Ryzen 7 PRO 8840HS w/ Radeon 780M Graphics, 16 logical cores; RAM: 59.7 GiB
- OS: Windows-11-10.0.26220-SP0; Python 3.14.6 (CPython)
- Libraries: numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, statsmodels 0.15.0, fastapi 0.143.2, starlette 1.8.0, pydantic 2.14.1, SQLAlchemy 2.1.4
- Data: KO/PEP pairs, every other setting default, 2011-01-01 to 2025-12-31: 3,772 trading days, from bench.db (no downloads)
- Each row: 10 timed runs after one warm-up run, wall clock (time.perf_counter)

**CLAUDE.md §14 targets**

| Workload | Median | Min | Max | Target | Met |
|---|---:|---:|---:|---:|---:|
| 15-year pairs backtest, first request on the data (diagnostics computed) | 361 ms | 339 ms | 410 ms | < 300 ms | **no** |
| 15-year pairs backtest, z_entry changed (diagnostics cached) | 150 ms | 136 ms | 170 ms | < 300 ms | yes |
| Default sensitivity grid (20 configurations) | 651 ms | 524 ms | 770 ms | < 5 s | yes |
| Default Monte Carlo, 10,000 x 252, both models: the request | 335 ms | 328 ms | 361 ms | < 1 s | yes |
| Default Monte Carlo: the simulation alone (elapsed_ms_total) | 243 ms | 237 ms | 257 ms | < 1 s | yes |

**Pairs backtest by stage (first request on the data)**

| Stage | Median | Min | Max |
|---|---:|---:|---:|
| Load prices and rf from the cache | 55.2 ms | 51.5 ms | 79.7 ms |
| run_strategy (section 5.0, steps 1-6) | 22.4 ms | 21.2 ms | 27.2 ms |
| Diagnostics: two Engle-Granger tests, static OLS, half-life | 184 ms | 176 ms | 203 ms |
| Report: metrics, benchmark, trades, warnings, series | 10.2 ms | 8.92 ms | 11.5 ms |
| Serialization: payload, response model, JSON | 37.8 ms | 37.1 ms | 54.2 ms |
| Sum of the stages | 312 ms | 295 ms | 373 ms |
| Diagnostics from the cache instead (key and lookup) | 0.15 ms | 0.15 ms | 0.18 ms |

The whole request took 361 ms (median). The rest is parsing and validating the request, storing the run, FastAPI's own response handling and the TestClient round trip.

**Parameter iteration: each request changes one field of the one before**

| Field changed | Median | Min | Max | Diagnostics cache |
|---|---:|---:|---:|---:|
| z_entry (1.55, 1.60, ...) | 150 ms | 136 ms | 170 ms | 10 hits, 0 misses |
| L (24, 28, ...), default 70% split | 361 ms | 343 ms | 392 ms | 0 hits, 10 misses |
| L (24, 28, ...), split_date fixed | 152 ms | 139 ms | 170 ms | 10 hits, 0 misses |

**Monte Carlo**

| Model (time inside the default request) | Median | Min | Max |
|---|---:|---:|---:|
| GBM | 127 ms | 125 ms | 133 ms |
| Circular block bootstrap (block length 10) | 113 ms | 109 ms | 122 ms |

Peak traced memory at the API caps (50,000 paths x 253 days): 216 MB, 2.13x the 102 MB path array (tracemalloc, which sees numpy's buffers).

**What the numbers mean**

- **The first request on new data misses the 300 ms target**, at 361 ms. Half of it is the two Engle–Granger cointegration tests.
- **The API caches those tests,** because they depend only on the two price series and the split date (METHODOLOGY §20). Changing z_entry, z_exit, costs or the hedge mode reuses them, and the backtest takes 150 ms.
- **Changing L recomputes them under the default split,** because a new lookback moves the default 70% split date. With `split_date` set, L changes hit the cache too (152 ms).
- **The sensitivity grid and Monte Carlo** are well inside their targets, at 651 ms against 5 s and 335 ms against 1 s.

## Screenshots

All taken from the dashboard running against the local API, on Yahoo data, on 2026-10-16.

| | |
|---|---|
| ![KPI grid](docs/screenshots/kpi-grid.png) KPI grid with the IS / OOS / Full toggle | ![Equity and drawdown](docs/screenshots/equity-drawdown.png) Equity and drawdown, with the OOS segment shaded |
| ![Signal panel](docs/screenshots/signal-panel.png) Signal panel: z-score with thresholds and trade markers, hedge ratio, gross exposure | ![Trades table](docs/screenshots/trades-table.png) Trades table |
| ![Sensitivity heatmap](docs/screenshots/sensitivity-heatmap.png) In-sample sensitivity grid | ![Monte Carlo](docs/screenshots/monte-carlo.png) Monte Carlo fan chart, histogram and risk table |
| ![Unknown ticker](docs/screenshots/error-unknown-ticker.png) An unknown ticker (404) | ![Grid needs more history](docs/screenshots/error-sensitivity-grid.png) A trend history too short for the grid (422) |

Full-page captures: [KO/PEP pairs](docs/screenshots/dashboard-pairs-full.png) and [SPY trend, 2006–2025, with a 10% volatility target](docs/screenshots/dashboard-trend-voltarget-full.png).
