# Quantitative Backtesting Engine

A backtesting and Monte Carlo risk engine for daily US equity strategies, with a React dashboard. It runs four strategies:
- single-asset mean reversion;
- pairs trading, with a rolling-OLS or Kalman-filter hedge ratio;
- time-series momentum (trend following) on one asset;
- a multi-asset trend portfolio of 2–12 ETFs, each leg sized to the same volatility.

Each can take a volatility-targeting overlay (the portfolio always has one), and all but pairs have a long-only switch. Results come split in-sample and out-of-sample, with conservative accounting: a one-day execution lag by default, per-leg trading costs, an optional borrow fee on shorts, and cash at the T-bill rate.

Several tools test how much a result can be trusted:
- a cost sweep with the break-even cost;
- rolling one-year Sharpe and beta, and calendar returns;
- a parameter grid with the deflated Sharpe ratio, which discounts the best cell for the number of cells tried;
- a walk-forward analysis, which re-chooses the grid's best configuration on each training window and trades it on the next test window, giving a track record that is out-of-sample throughout;
- Monte Carlo projections with GBM, a circular block bootstrap and a stationary bootstrap, with an optional data-driven block length, checked against closed-form VaR and CVaR.

The dashboard keeps a history of every backtest, with side-by-side comparison and "Load settings". Its settings live in the URL, so a link reproduces a run, and its trades, daily series and calendar returns export as CSV. Browser tests drive it in Chromium against an offline API.

- **Specification:** [CLAUDE.md](CLAUDE.md), the source of truth for every formula and rule.
- **Methodology:** [docs/METHODOLOGY.md](docs/METHODOLOGY.md) justifies each formula, lists the limitations, and records every approved deviation from the specification.
- **Status:** Phases 0–9 are done: the engine, the API, the dashboard, the documentation and the benchmarks, then costs and robustness metrics, walk-forward analysis, the Kalman pairs model, the trend portfolio and the stationary bootstrap, run history, URL state, CSV export, the log scale, offline mode and browser tests, and a [static demo](#static-demo). The demo is built and tested but not deployed yet.

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

Prices are downloaded from Yahoo Finance on first use and cached in `backend\quant.db`. With `QUANT_OFFLINE=1` set (in `.env` or the shell) the API never downloads: a ticker or date range that isn't cached returns 503.

The sidebar's settings are written to the URL when you press Run, so the link reproduces the run; opening a link fills the sidebar and sends nothing until you press Run. **Run history**, at the bottom of the page, lists every stored backtest. "Load settings" puts a run's settings back in the sidebar, and ticking up to four runs compares their metrics.

## Static demo

A read-only build of the dashboard for GitHub Pages, which replays recorded API responses and needs no backend. It has eight scenarios: pairs on KO/PEP, XOM/CVX, GOOGL/GOOG and NVDA/AMD, mean reversion on KO, SPY trend with the volatility target off and on, and the trend portfolio of SPY, QQQ, GLD and TLT. Each runs 2015-01-01 to 2025-12-31 with the default settings, and includes its sensitivity grid, walk-forward and Monte Carlo (seed 42). The sidebar is read-only apart from a scenario picker. How the replay works is in METHODOLOGY [§33](docs/METHODOLOGY.md).

**Not deployed yet.** The recorded series are derived from Yahoo Finance data, and publishing them waits on a decision about redistributing that data (CLAUDE.md §13). Until then `.github/workflows/pages.yml` builds the demo on every push to `main` but doesn't deploy it. Once deployed it will be at https://jacobchen08.github.io/Quantative-Backtesting-Engine/.

The current recording, in `frontend\public\demo`, was made on 2026-10-20 from data through 2025-12-31: 33 files, 8.65 MB, against a 10 MB limit.

```powershell
# Re-record. Downloads any prices missing from backend\quant.db first; adds nothing to the run history.
cd backend
.venv\Scripts\python.exe scripts\record_demo.py

# Build the demo and serve it at http://localhost:4173/Quantative-Backtesting-Engine/
cd ..\frontend
npm run build:demo
npm run preview:demo
```

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

The backend tests use synthetic data and a fake network, so they never reach Yahoo. They run in parallel, one worker per core (pytest-xdist); add `-n 0` to run them in one process.

### Browser tests

```powershell
cd frontend
npx playwright install chromium   # once
npx playwright test
```

Playwright starts its own three servers and stops them afterwards: the API on port 8001, offline, on a temporary database that `backend\scripts\seed_e2e.py` fills with the synthetic market (tickers SYNY, SYNX, SYNT and SPY, 2012–2019); a fresh production build on port 4173; and a fresh build of the [static demo](#static-demo) on port 4174. Nothing reaches the network, and dev servers on 8000 and 5173 are left alone. The 14 tests took 69.7 s here (`npx playwright test` in PowerShell on 2026-10-20, including both builds). CI runs them on Ubuntu.

## Benchmarks

Every runtime figure in this repository comes from `scripts/bench.py`:

```powershell
cd backend
.venv\Scripts\python.exe scripts\bench.py
```

The bench keeps its own price cache, `backend\bench.db`, so it never touches the app's database or run history. Its first run downloads the data, and every timed request after that is served from the cache. Requests go through FastAPI's TestClient in process, so the times include validation, the engine, storing the run and JSON encoding, but no network or browser.

The tables below are copied from its output on 2026-10-20, after the performance refactor, with no other workload running:

- CPU: AMD Ryzen 7 PRO 8840HS w/ Radeon 780M Graphics, 16 logical cores; RAM: 59.7 GiB
- OS: Windows-11-10.0.26220-SP0; Python 3.14.6 (CPython)
- Libraries: numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, statsmodels 0.15.0, arch 8.0.0, fastapi 0.143.2, starlette 1.8.0, pydantic 2.14.1, SQLAlchemy 2.1.4
- Data: KO/PEP pairs, every other setting default, 2011-01-01 to 2025-12-31: 3,772 trading days, from bench.db (no downloads)
- Each row: 10 timed runs after one warm-up run, wall clock (time.perf_counter)

**CLAUDE.md §14 targets**

| Workload | Median | Min | Max | Target | Met |
|---|---:|---:|---:|---:|---:|
| 15-year pairs backtest, first request on the data (diagnostics computed) | 299 ms | 270 ms | 327 ms | < 300 ms | yes |
| 15-year pairs backtest, z_entry changed (diagnostics cached) | 134 ms | 129 ms | 177 ms | < 300 ms | yes |
| Default sensitivity grid (20 configurations) | 423 ms | 340 ms | 456 ms | < 5 s | yes |
| Default walk-forward, 15-year pairs (20 configurations) | 385 ms | 373 ms | 423 ms | < 8 s | yes |
| Default Monte Carlo, 10,000 x 252: GBM and block bootstrap (their elapsed_ms) | 147 ms | 137 ms | 173 ms | < 1 s | yes |
| Default Monte Carlo, three models: the request | 384 ms | 362 ms | 420 ms | < 1.5 s | yes |
| Default Monte Carlo, three models: the simulation alone (elapsed_ms_total) | 294 ms | 268 ms | 320 ms | < 1.5 s | yes |
| 15-year Kalman pairs backtest, first request on the data (diagnostics computed) | 256 ms | 251 ms | 268 ms | < 500 ms | yes |

**Pairs backtest by stage (first request on the data)**

| Stage | Median | Min | Max |
|---|---:|---:|---:|
| Load prices and rf from the cache | 59.8 ms | 51.5 ms | 78.2 ms |
| run_strategy (section 5.0, steps 1-6) | 23.4 ms | 20.2 ms | 32.7 ms |
| Diagnostics: two Engle-Granger tests, static OLS, half-life | 137 ms | 114 ms | 149 ms |
| Report: metrics, benchmark, trades, warnings, series | 37.6 ms | 30.9 ms | 43.2 ms |
| Serialization: payload, response model, JSON | 17.1 ms | 15.8 ms | 27.1 ms |
| Sum of the stages | 281 ms | 234 ms | 310 ms |
| Diagnostics from the cache instead (key and lookup) | 0.17 ms | 0.14 ms | 0.23 ms |

The whole request took 299 ms (median). The rest is parsing and validating the request, storing the run, FastAPI's own response handling and the TestClient round trip.

**Parameter iteration: each request changes one field of the one before**

| Field changed | Median | Min | Max | Diagnostics cache |
|---|---:|---:|---:|---:|
| z_entry (1.55, 1.60, ...) | 134 ms | 129 ms | 177 ms | 10 hits, 0 misses |
| L (24, 28, ...), default 70% split | 315 ms | 296 ms | 368 ms | 0 hits, 10 misses |
| L (24, 28, ...), split_date fixed | 169 ms | 150 ms | 216 ms | 10 hits, 0 misses |

**Walk-forward (default: 756 training and 63 test days, rolling)**

| Stage | Median | Min | Max |
|---|---:|---:|---:|
| The request (as in the first table) | 385 ms | 373 ms | 423 ms |
| run_walk_forward alone (no loading or serialization) | 330 ms | 318 ms | 369 ms |

It backtests 20 grid configurations plus the request's own, then makes 46 folds: 2,897 out-of-sample days, on which 15 distinct configurations were chosen.

**Kalman pairs and the trend portfolio**

| Workload | Median | Min | Max |
|---|---:|---:|---:|
| run_strategy, rolling OLS (the default pairs model) | 20.0 ms | 19.3 ms | 25.4 ms |
| run_strategy, Kalman filter (delta 1e-6) | 31.6 ms | 23.2 ms | 34.5 ms |
| Trend portfolio of 10 ETFs: the request | 366 ms | 341 ms | 516 ms |
| Trend portfolio of 10 ETFs: run_strategy alone | 83.4 ms | 68.8 ms | 90.1 ms |

The trend portfolio (SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, VNQ; L = 252, a 40% target per asset) runs on the intersection of their calendars: 3,520 series days from 2012-01-03, 219 leg trades.

**Monte Carlo**

| Model (time inside the default request) | Median | Min | Max |
|---|---:|---:|---:|
| GBM | 80.9 ms | 73.1 ms | 93.3 ms |
| Circular block bootstrap (block length 10) | 64.9 ms | 57.6 ms | 79.8 ms |
| Stationary bootstrap (mean block length 10) | 141 ms | 130 ms | 166 ms |
| The request with block_len "auto" | 378 ms | 350 ms | 405 ms |
| The simulation alone with block_len "auto" (elapsed_ms_total) | 295 ms | 268 ms | 304 ms |

With block_len "auto" the block lengths were 2 (circular) and 2 (stationary).

Peak traced memory at the API caps (50,000 paths x 253 days, three models): 216 MB, 2.13x the 102 MB path array (tracemalloc, which sees numpy's buffers).

**Before and after the performance refactor** (medians from two bench runs on 2026-10-20, same machine; every API response is byte-identical between the two versions apart from the Monte Carlo's timings)

| Workload | Before | After |
|---|---:|---:|
| 15-year pairs backtest, first request on the data | 414 ms | 299 ms |
| Two Engle–Granger tests (the diagnostics stage) | 201 ms | 137 ms |
| Serialization stage | 51.9 ms | 17.1 ms |
| Default sensitivity grid | 689 ms | 423 ms |
| Default walk-forward | 728 ms | 385 ms |
| Default Monte Carlo, three models: the request | 682 ms | 384 ms |
| Trend portfolio of 10 ETFs: run_strategy alone | 139 ms | 83.4 ms |

The changes, each checked to leave every output bit-identical (METHODOLOGY §20):
- OpenBLAS runs on one thread. The two Engle–Granger tests fit 53 least-squares regressions of about 2,600 rows and up to 28 columns, and at that size OpenBLAS's threads cost more than they save: one thread nearly halves the tests' time.
- The sensitivity grid and the walk-forward skip the trade bookkeeping of their 20 backtests; they only read the daily returns.
- The entry and exit rule, the trend portfolio's sizing and the JSON conversion of the chart series work on whole arrays instead of element by element or leg by leg. The stationary bootstrap finds where each block began with a loop of whole-row maxima, about 20 times faster than NumPy's accumulate down the columns.
- The Monte Carlo bands (five quantiles of 10,000 paths on each of 253 days) are computed in four blocks of days at once; NumPy releases the interpreter lock while it partitions.

**What the numbers mean**

- **The first request on new data now meets the 300 ms target, narrowly**: 299 ms median, with runs from 270 to 327 ms. The two Engle–Granger cointegration tests are still almost half of it (137 ms).
- **The API caches those tests,** because they depend only on the two price series and the split date (METHODOLOGY §20). Changing z_entry, z_exit, costs or the hedge mode reuses them, and the backtest takes 134 ms.
- **Changing L recomputes them under the default split,** because a new lookback moves the default 70% split date. With `split_date` set, L changes hit the cache too (169 ms).
- **A first Kalman request is no slower than a first rolling-OLS request** (256 ms and 299 ms): both run the same Engle–Granger tests, and the gap between the medians is run-to-run variation. Inside `run_strategy` the filter takes 31.6 ms against 20.0 ms for rolling OLS.
- **Runs vary.** Before the refactor, the Phase 7 run on 2026-10-19 measured the first pairs request at 351 ms, and the run on 2026-10-20 at 414 ms, on the same machine. The before and after figures above come from the same day.
- **The stationary bootstrap is still the slowest model,** at 141 ms against 65 ms for the block bootstrap: it draws a restart flag and a start for every day of every path (METHODOLOGY §29).
- **Memory is unchanged** at 216 MB: the peak comes from the block bootstrap, and the stationary bootstrap builds its paths in chunks of 2,048.
- **A 10-ETF trend portfolio takes 366 ms,** with 83 ms in `run_strategy`: ten legs' forecasts, signals and sizing, and per-leg trades.
- **The sensitivity grid and the walk-forward** are well inside their targets, at 423 ms against 5 s and 385 ms against 8 s.

## Screenshots

All taken on 2026-10-19 from the dashboard running against the local API, on Yahoo data, by `npm run screenshots` (`frontend\scripts\screenshots.ts`). With the API on 8000 and `npm run dev` on 5173 running, that script regenerates every file in `docs\screenshots`; it downloads any data the cache lacks. The KO/PEP section views show the in-sample segment, where the cost sweep has a break-even line. The walk-forward view has the buy & hold line hidden with its toggle, so the axis fits the two pairs strategies.

| | |
|---|---|
| ![KPI grid](docs/screenshots/kpi-grid.png) KPI grid with the IS / OOS / Full toggle, costs and the break-even cost | ![Equity and drawdown](docs/screenshots/equity-drawdown.png) Equity and drawdown, with the OOS segment shaded |
| ![Rolling statistics](docs/screenshots/rolling-statistics.png) Rolling one-year Sharpe and beta to SPY | ![Calendar returns](docs/screenshots/calendar-returns.png) Monthly and annual returns, with partial periods hatched |
| ![Signal panel](docs/screenshots/signal-panel.png) Signal panel: z-score with thresholds and trade markers, hedge ratio, gross exposure | ![Trades table](docs/screenshots/trades-table.png) Trades table |
| ![Cost sweep](docs/screenshots/cost-sweep.png) Sharpe against one-way cost, with the break-even cost | ![Sensitivity heatmap](docs/screenshots/sensitivity-heatmap.png) In-sample sensitivity grid and its deflated Sharpe ratios |
| ![Walk-forward](docs/screenshots/walk-forward.png) Walk-forward: the test windows stitched together against L 60 / z 2.0 held fixed, the metrics and the fold table | ![Monte Carlo](docs/screenshots/monte-carlo.png) Monte Carlo fan chart, histogram and risk table |
| ![Kalman signal panel](docs/screenshots/signal-panel-kalman.png) KO/PEP with the Kalman model: the innovation z-score and the filtered hedge ratio | ![Trend-portfolio signal panel](docs/screenshots/signal-panel-trend-portfolio.png) Trend portfolio of 10 ETFs: each leg's score and trades, the executed weights, gross exposure and each leg's vol forecast |
| ![Trend-portfolio KPI grid](docs/screenshots/kpi-grid-trend-portfolio.png) The trend portfolio's KPI grid against its equal-weight basket, with the per-asset target and the legs' days at their cap | ![Stationary bootstrap](docs/screenshots/monte-carlo-stationary.png) Monte Carlo of the trend portfolio with automatic block lengths, showing the stationary bootstrap |
| ![Log-scale equity](docs/screenshots/equity-drawdown-log.png) SPY trend with a 10% volatility target, 2006–2025, on the log scale | ![Run history](docs/screenshots/run-history.png) Run history, comparing four stored runs on the full window |
| ![Unknown ticker](docs/screenshots/error-unknown-ticker.png) An unknown ticker (404) | ![Grid needs more history](docs/screenshots/error-sensitivity-grid.png) A trend history too short for the grid (422) |

Full-page captures: [KO/PEP pairs](docs/screenshots/dashboard-pairs-full.png), [SPY trend, 2006–2025, with a 10% volatility target](docs/screenshots/dashboard-trend-voltarget-full.png), and [the diversified trend portfolio of 10 ETFs, 2015–2025](docs/screenshots/dashboard-trend-portfolio-full.png).
