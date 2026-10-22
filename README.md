# Quantitative Backtesting Engine

I built this to test daily trading strategies on US stocks and ETFs the way I'd want someone else's results tested: with realistic costs, a clean split between the data a strategy was tuned on and the data that judges it, and a few tools that try to talk you out of a good-looking result. A FastAPI backend does the math and a React dashboard shows it.

There are four strategies: mean reversion on a single stock, pairs trading (with either a rolling regression or a Kalman filter for the hedge ratio), trend following on one asset, and a trend portfolio of 2 to 12 ETFs, each sized to the same volatility. Any of them can be sized to a volatility target, and all but pairs can be run long-only. The accounting is deliberately conservative: trades fill a day after the signal by default, every leg pays a trading cost, shorts can pay a borrow fee, and idle cash earns the T-bill rate. Results are reported in-sample and out-of-sample.

Most of the project is about whether a result deserves to be trusted:
- a **cost sweep** that shows the Sharpe ratio at every trading cost and the cost at which the edge disappears;
- rolling one-year Sharpe and beta, and monthly and yearly returns;
- a **parameter grid** with the deflated Sharpe ratio, which marks down the best cell for the number of cells tried;
- a **walk-forward analysis**, which re-picks the best settings on each training window and trades them on the next, so the whole track record is out-of-sample;
- **Monte Carlo** projections from three models (GBM and two bootstraps of the real returns), checked against closed-form VaR and CVaR.

The dashboard also keeps a history of past runs you can compare side by side, puts its settings in the URL so a link reproduces a run, and exports trades and series as CSV.

**Built with** Python (FastAPI, NumPy, pandas, SciPy, statsmodels, SQLAlchemy, SQLite), React and TypeScript (Vite, Tailwind CSS, Recharts), and pytest, Vitest, Playwright and GitHub Actions for testing.

**Where to read more.** If you're new to the code, start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): it follows a request from the Run button through the engine and back, and explains the vocabulary. [CLAUDE.md](CLAUDE.md) is the specification the code is built against; every formula and rule is defined there, and the § numbers in code comments point into it. [docs/METHODOLOGY.md](docs/METHODOLOGY.md) explains why each formula is the way it is, lists the model's limitations, and records every place the code departs from the spec.

All nine planned phases are finished, including the [static demo](#static-demo), which is built and tested but not deployed yet.

![The dashboard on a KO/PEP pairs backtest](docs/screenshots/dashboard-pairs.png)

*KO/PEP pairs, 2015-01-02 to 2025-12-31, default settings. More views [below](#screenshots).*

## Setup (PowerShell)

You need Python 3.12 or later and Node 24 or later.

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

Prices come from Yahoo Finance the first time a ticker is used, and are cached in `backend\quant.db` after that. Set `QUANT_OFFLINE=1` (in `.env` or the shell) to stop the API downloading anything; a ticker or date range that isn't cached then returns a 503.

Pressing Run writes the sidebar's settings into the URL, so you can share or bookmark a run. Opening such a link fills in the sidebar but doesn't send anything until you press Run. The **Run history** panel at the bottom of the page lists every stored backtest: "Load settings" puts a run's settings back in the sidebar, and ticking up to four runs compares their metrics.

## Static demo

The demo is a read-only build of the dashboard for GitHub Pages. It replays recorded API responses, so it needs no backend. There are eight scenarios: pairs on KO/PEP, XOM/CVX, GOOGL/GOOG and NVDA/AMD, mean reversion on KO, SPY trend with and without the volatility target, and a trend portfolio of SPY, QQQ, GLD and TLT. Each covers 2015-01-01 to 2025-12-31 with the default settings, and includes its sensitivity grid, walk-forward and Monte Carlo (seed 42). The only control you can change is the scenario picker. METHODOLOGY [§33](docs/METHODOLOGY.md) explains how the replay works.

It isn't deployed yet. The recorded series are derived from Yahoo Finance data, and I haven't decided whether redistributing them is acceptable (CLAUDE.md §13). Until then `.github/workflows/pages.yml` builds the demo on every push to `main` but skips the deploy. Once it's live it will be at https://jacobchen08.github.io/Quantitative-Backtesting-Engine/.

The current recording in `frontend\public\demo` was made on 2026-10-20 from data through 2025-12-31. It is 33 files and 8.65 MB, under the 10 MB limit.

```powershell
# Re-record. Downloads any prices missing from backend\quant.db first; adds nothing to the run history.
cd backend
.venv\Scripts\python.exe scripts\record_demo.py

# Build the demo and serve it at http://localhost:4173/Quantitative-Backtesting-Engine/
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

The backend tests run on synthetic data with a fake network, so they never touch Yahoo. pytest-xdist runs them in parallel, one worker per core; add `-n 0` to run them in a single process.

### Browser tests

```powershell
cd frontend
npx playwright install chromium   # once
npx playwright test
```

Playwright starts three servers of its own and shuts them down afterwards. The first is the API on port 8001, offline, on a temporary database that `backend\scripts\seed_e2e.py` fills with the synthetic market (tickers SYNY, SYNX, SYNT and SPY, 2012–2019). The second is a fresh production build on port 4173, and the third a fresh build of the [static demo](#static-demo) on port 4174. Nothing goes to the network, and any dev servers you have on 8000 and 5173 are left alone. The 14 tests took 69.7 s on my machine (`npx playwright test` in PowerShell on 2026-10-20, including both builds). CI runs them on Ubuntu.

## Benchmarks

Every timing here comes from `scripts/bench.py`, apart from the paired comparison in the speed-ups table below:

```powershell
cd backend
.venv\Scripts\python.exe scripts\bench.py
```

The bench has its own price cache, `backend\bench.db`, so it never touches the app's database or run history. The first run downloads the data; after that every timed request is served from the cache. Requests go through FastAPI's TestClient in the same process, so the times include validation, the engine, storing the run and JSON encoding, but no network or browser.

These tables are copied from its output on 2026-10-21, after the second refactor, with nothing else running:

- CPU: AMD Ryzen 7 PRO 8840HS w/ Radeon 780M Graphics, 16 logical cores; RAM: 59.7 GiB
- OS: Windows-11-10.0.26220-SP0; Python 3.14.6 (CPython)
- Libraries: numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, statsmodels 0.15.0, arch 8.0.0, fastapi 0.143.2, starlette 1.8.0, pydantic 2.14.1, SQLAlchemy 2.1.4
- Data: KO/PEP pairs, every other setting default, 2011-01-01 to 2025-12-31: 3,772 trading days, from bench.db (no downloads)
- Each row: 10 timed runs after one warm-up run, wall clock (time.perf_counter)

**CLAUDE.md §14 targets**

| Workload | Median | Min | Max | Target | Met |
|---|---:|---:|---:|---:|---:|
| 15-year pairs backtest, first request on the data (diagnostics computed) | 290 ms | 249 ms | 323 ms | < 300 ms | yes |
| 15-year pairs backtest, z_entry changed (diagnostics cached) | 121 ms | 109 ms | 167 ms | < 300 ms | yes |
| Default sensitivity grid (20 configurations) | 221 ms | 181 ms | 264 ms | < 5 s | yes |
| Default walk-forward, 15-year pairs (20 configurations) | 232 ms | 205 ms | 286 ms | < 8 s | yes |
| Default Monte Carlo, 10,000 x 252: GBM and block bootstrap (their elapsed_ms) | 139 ms | 128 ms | 147 ms | < 1 s | yes |
| Default Monte Carlo, three models: the request | 347 ms | 335 ms | 388 ms | < 1.5 s | yes |
| Default Monte Carlo, three models: the simulation alone (elapsed_ms_total) | 279 ms | 264 ms | 308 ms | < 1.5 s | yes |
| 15-year Kalman pairs backtest, first request on the data (diagnostics computed) | 243 ms | 237 ms | 267 ms | < 500 ms | yes |

**Pairs backtest by stage (first request on the data)**

| Stage | Median | Min | Max |
|---|---:|---:|---:|
| Load prices and rf from the cache | 53.3 ms | 35.5 ms | 71.6 ms |
| run_strategy (section 5.0, steps 1-6) | 21.3 ms | 16.3 ms | 26.4 ms |
| Diagnostics: two Engle-Granger tests, static OLS, half-life | 154 ms | 123 ms | 164 ms |
| Report: metrics, benchmark, trades, warnings, series | 43.2 ms | 36.7 ms | 57.9 ms |
| Serialization: payload, response model, JSON | 21.3 ms | 16.8 ms | 36.6 ms |
| Sum of the stages | 291 ms | 261 ms | 308 ms |
| Diagnostics from the cache instead (key and lookup) | 0.19 ms | 0.14 ms | 0.22 ms |

The whole request took 290 ms (median). The difference from the sum of the stages is request parsing and validation, storing the run, FastAPI's own response handling and the TestClient round trip.

**Parameter iteration: each request changes one field of the one before**

| Field changed | Median | Min | Max | Diagnostics cache |
|---|---:|---:|---:|---:|
| z_entry (1.55, 1.60, ...) | 121 ms | 109 ms | 167 ms | 10 hits, 0 misses |
| L (24, 28, ...), default 70% split | 277 ms | 253 ms | 347 ms | 0 hits, 10 misses |
| L (24, 28, ...), split_date fixed | 140 ms | 133 ms | 168 ms | 10 hits, 0 misses |

**Walk-forward (default: 756 training and 63 test days, rolling)**

| Stage | Median | Min | Max |
|---|---:|---:|---:|
| The request (as in the first table) | 232 ms | 205 ms | 286 ms |
| run_walk_forward alone (no loading or serialization) | 157 ms | 152 ms | 189 ms |

That's 20 grid configurations plus the request's own, cut into 46 folds: 2,897 out-of-sample days, with 15 different configurations chosen along the way.

**Kalman pairs and the trend portfolio**

| Workload | Median | Min | Max |
|---|---:|---:|---:|
| run_strategy, rolling OLS (the default pairs model) | 14.7 ms | 14.3 ms | 18.5 ms |
| run_strategy, Kalman filter (delta 1e-6) | 26.9 ms | 17.3 ms | 30.7 ms |
| Trend portfolio of 10 ETFs: the request | 306 ms | 269 ms | 422 ms |
| Trend portfolio of 10 ETFs: run_strategy alone | 55.6 ms | 48.8 ms | 69.5 ms |

The trend portfolio holds SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC and VNQ, with L = 252 and a 40% volatility target per asset. It runs on the days all ten traded: 3,520 days from 2012-01-03, with 219 trades across the legs.

**Monte Carlo**

| Model (time inside the default request) | Median | Min | Max |
|---|---:|---:|---:|
| GBM | 78.9 ms | 71.6 ms | 81.2 ms |
| Circular block bootstrap (block length 10) | 60.1 ms | 56.1 ms | 70.5 ms |
| Stationary bootstrap (mean block length 10) | 138 ms | 127 ms | 167 ms |
| The request with block_len "auto" | 348 ms | 329 ms | 374 ms |
| The simulation alone with block_len "auto" (elapsed_ms_total) | 283 ms | 267 ms | 306 ms |

With block_len "auto" the block lengths were 2 (circular) and 2 (stationary).

At the API's limits (50,000 paths x 253 days, three models), peak memory was 216 MB, 2.13 times the 102 MB path array (measured with tracemalloc, which sees NumPy's buffers).

### Two rounds of speed-ups

I went through the code twice looking for speed, on 2026-10-20 and 2026-10-21. Both times every API response stayed byte-identical on 27 requests covering every strategy and option, apart from the Monte Carlo's wall-clock timings (METHODOLOGY §20).

| Workload | Before the first | After the first | After the second |
|---|---:|---:|---:|
| 15-year pairs backtest, first request on the data | 414 ms | 299 ms | 0.93x of that |
| 15-year pairs backtest, z_entry changed | n/a | 134 ms | 0.85x |
| Default sensitivity grid | 689 ms | 423 ms | 0.49x |
| Default walk-forward | 728 ms | 385 ms | 0.51x |
| Trend portfolio of 10 ETFs: the request | n/a | 366 ms | 0.77x |
| Default Monte Carlo, three models: the request | 682 ms | 384 ms | 0.94x |

The first two columns are bench medians from a single day. The last column had to be measured differently: on 2026-10-21 this laptop's speed drifted by as much as 2x between bench runs, which swamped the differences I was trying to measure. So I alternated the old and new code in eight back-to-back rounds of the same workloads, through the same TestClient on a copy of bench.db, and took the median of the eight per-round ratios. The grid's ratio ranged from 0.47 to 0.53.

The second round changed five things:
- The daily return arithmetic of §6.1 (held weights, P&L, borrow, turnover, cost) now runs on plain NumPy arrays. It gives exactly the same floats as the pandas version, about 15 times faster, and the grids run it 21 times per request.
- The grids compute each lookback's score once. A configuration's score and volatility forecast depend on L but not on z_entry, so the 20-cell pairs grid now computes 5 scores instead of 20. A test checks that no cell changes.
- Trades are numbered only when they are going to be recorded. The grids never record them, and the days a position is held now come straight from the positions.
- The trend portfolio's trades go into one table instead of one per leg.
- Price reads parse the stored dates in one call instead of building a Python date per row.

### Reading the numbers

The first request on new data meets the 300 ms target, but only just: 290 ms median in the run above. About half of that is the two Engle–Granger cointegration tests. I kept statsmodels' `coint` for them, as §5.4 requires, rather than write a faster version that might not match it to the last digit.

Those tests depend only on the two price series and the split date, so the API caches them (METHODOLOGY §20). Changing z_entry, z_exit, costs or the hedge mode reuses the cached result, and the backtest drops to 121 ms. Changing L doesn't, under the default split, because a new lookback moves the default 70% split date. With `split_date` set explicitly, L changes hit the cache too (140 ms).

Timings on this machine vary by up to 2x over an evening, so only compare figures from the same run. Each row shows its own min-to-max spread.

The stationary bootstrap is the slowest Monte Carlo model, at 138 ms against 60 ms for the block bootstrap, because it draws a restart flag and a starting point for every day of every path (METHODOLOGY §29). Its random draws determine the seeded results, so I left it as it is. Memory hasn't changed at 216 MB: the peak comes from the block bootstrap, and the stationary bootstrap builds its paths in chunks of 2,048.

The sensitivity grid and the walk-forward are well inside their targets, at 221 ms against 5 s and 232 ms against 8 s.

## Screenshots

All of these were taken on 2026-10-19 from the dashboard running against the local API on Yahoo data, by `npm run screenshots` (`frontend\scripts\screenshots.ts`). With the API on 8000 and `npm run dev` on 5173, that script regenerates every file in `docs\screenshots`, downloading any data the cache lacks. The KO/PEP views show the in-sample segment, because that's where the cost sweep has a break-even line. In the walk-forward view I hid the buy & hold line with its toggle, so the axis fits the two pairs strategies.

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
