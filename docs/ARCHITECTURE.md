# How the code fits together

A guide for reading this repository for the first time. It explains what happens when you press **Run**, where each piece lives, the vocabulary the code uses, and how the results are checked. The formulas themselves are in [CLAUDE.md](../CLAUDE.md), and the reasons behind them are in [METHODOLOGY.md](METHODOLOGY.md).

**About the § numbers.** Comments and docstrings cite sections of CLAUDE.md, the project's specification: "§6.1" means section 6.1 there. They let a reader jump from any line of code to the rule it implements.

## The big picture

```
 Browser (React)                       API (FastAPI)                      Engine (pure Python)
 ───────────────                       ─────────────                      ────────────────────
 Sidebar ─► form.ts builds requests
            api.ts sends them ───────► main.py route
                                         data_loader: prices + T-bill ──► pipeline.run_strategy
                                         (SQLite cache, Yahoo Finance)       strategies  score → signal → weights
                                                                             sizing      volatility targeting
                                                                             backtester  daily P&L, costs, trades
                                                                          report.build_report
                                                                             metrics     Sharpe, drawdown, ...
                                                                             stats       cointegration tests
                                         serialization: to JSON ◄──────── result dataclasses
 transform.ts shapes it ◄──────────────  schemas: validated response
 components/ draw the charts
```

Three layers, each with one job:

1. **The engine** (`backend/app/services/`) does all the math. Its modules are *pure*: prices go in, results come out, with no database, network or global state. That makes every formula testable on small hand-made inputs.
2. **The API** (`backend/app/main.py` and its neighbours) validates requests, loads prices, calls the engine, and turns the results into JSON.
3. **The dashboard** (`frontend/src/`) collects settings, sends requests, and draws what comes back.

## What happens when you press Run

1. **The sidebar's settings become requests.** `frontend/src/form.ts` turns the form into a backtest request, a sensitivity-grid request and a Monte Carlo request, sending only the fields the chosen strategy uses. `api.ts` posts the backtest first.
2. **The API validates the request.** `schemas.py` (Pydantic) checks every field's range, fills in defaults, and rejects bad input with a 422 that names the rule.
3. **Prices are loaded.** `data_loader.py` serves each ticker from the SQLite cache, downloading from Yahoo Finance only what is missing, and builds the daily risk-free rate from the 13-week T-bill yield. Every ticker is aligned on the dates they all traded.
4. **The strategy runs.** `pipeline.run_strategy` performs the steps of §5.0 in order:
   - `strategies.py` computes the **score** (how stretched a price or spread is, or how strong a trend is), turns it into a **signal** of −1, 0 or +1, and then into **weights** per ticker;
   - `sizing.py` optionally scales those weights to a target volatility;
   - `backtester.py` applies the execution lag, computes each day's return after costs, finds the trades, and handles the rare case of losing 100%.
5. **The report is built.** `report.build_report` computes the metrics for the in-sample, out-of-sample and full periods (`metrics.py`), the benchmark, the pairs diagnostics (`stats.py`), the cost sweep, calendar returns, warnings, and the series the charts draw.
6. **The response goes back.** `serialization.py` converts the results to JSON (NaN becomes `null`), the response model checks the shape, and the run is stored for the run history.
7. **The dashboard draws it.** `transform.ts` reshapes the arrays into chart rows, and the components in `components/` render them. Then the sensitivity grid and Monte Carlo requests go out in parallel and fill their panels when they return.

The sensitivity grid (`sensitivity.py`) and the walk-forward (`walk_forward.py`) reuse step 4 for every parameter combination they try. The Monte Carlo (`monte_carlo.py`) takes the strategy's daily returns from step 4 and simulates thousands of possible futures.

## Where things live

### Backend (`backend/app/`)

| File | What it does |
|---|---|
| `main.py` | The API's routes, and the mapping from errors to HTTP status codes. Start here on the backend. |
| `schemas.py` | Request and response models: every field, its default and its valid range. |
| `serialization.py` | Engine results → JSON-safe dictionaries. |
| `models.py`, `database.py`, `config.py` | Database tables, the database connection, settings from environment variables. |
| `diagnostics_cache.py` | Remembers the slow cointegration tests between requests on the same data. |
| `services/data_loader.py` | The price cache and the single place that talks to Yahoo Finance. |
| `services/calendar.py` | Today's date in New York, aligning tickers on common dates, the daily T-bill rate. |
| `services/pipeline.py` | `StrategyParams` (one configuration) and `run_strategy` (the §5.0 steps). |
| `services/strategies.py` | The four strategies' scores, entry and exit rules, and weights. |
| `services/sizing.py` | Volatility targeting. |
| `services/backtester.py` | Execution lag, daily returns, costs, borrow fees, ruin, and trades. |
| `services/report.py` | Everything `/api/backtest` returns about one run. |
| `services/metrics.py` | Sharpe, Sortino, drawdown, beta and the rest; the cost sweep; rolling and calendar statistics; warnings. |
| `services/stats.py` | Engle–Granger cointegration test, half-life, Kalman filter, deflated Sharpe ratio. |
| `services/sensitivity.py` | The parameter grid. |
| `services/walk_forward.py` | Walk-forward re-optimization over that grid. |
| `services/monte_carlo.py` | GBM and bootstrap simulations, VaR and CVaR. |

`backend/scripts/` holds `bench.py` (the timings in the README), `seed_e2e.py` (the browser tests' database) and `record_demo.py` (the static demo's data). `backend/tests/` has one test file per engine module, plus `test_api.py` for the endpoints; `synthetic.py` generates the fake market they all use.

### Frontend (`frontend/src/`)

| File | What it does |
|---|---|
| `App.tsx` | The page: the sidebar, the result panels, and when each request is sent. Start here on the frontend. |
| `form.ts` | The sidebar's state and the requests built from it. |
| `api.ts` | Typed `fetch` wrappers for each endpoint. |
| `types.ts` | TypeScript types for every response, matching `schemas.py`. |
| `transform.ts` | Pure functions from responses to chart rows, table orderings and labels. |
| `format.ts` | Number and date formatting (the API sends decimals; percentages are made here). |
| `urlState.ts` | Settings in the page URL, so a link reproduces a run. |
| `history.ts`, `csv.ts` | Run history helpers; CSV export. |
| `demo.ts` | The static demo, which replays recorded responses instead of calling the API. |
| `components/` | One file per panel (`KpiGrid`, `BacktestCharts`, `MonteCarloPanel`, …), the sidebar (`Sidebar`, `SidebarSections`, `fields`), and shared pieces (`ui`, `chart`). |

The helpers' unit tests sit beside them as `.test.ts` files (Vitest). `frontend/e2e/` holds the browser tests (Playwright).

## Vocabulary

### Names in the code

The engine uses short names that match the formulas in CLAUDE.md:

| Name | Meaning |
|---|---|
| `L` | Lookback: how many trading days a rolling window covers. |
| `z`, `z_entry`, `z_exit` | The score of the mean-reversion and pairs strategies (how many standard deviations a price or spread is from normal), and the thresholds for opening and closing a position. |
| `s`, `signal` | The position the strategy wants: +1 long, −1 short, 0 flat. |
| `W` | Target weights: the fraction of equity the strategy wants in each ticker at each close. |
| `k` | Execution lag in trading days. With the default k = 1, a signal at today's close is traded at tomorrow's close. |
| `w`, `pos` | The weights and position actually held: `W` and `s` delayed by `k` days. |
| `R`, `rf` | Each ticker's daily return, and the daily risk-free rate. |
| `t0` | The first day the strategy's score exists (earlier days are a warm-up for the rolling windows). |
| `e` | The first day that counts in the results: `t0 + k`, the first day a position can be held. |
| `stop` | One past the last day that counts: the end of the data, or the day after a 100% loss ("ruin"). |
| `E0` | Starting capital. |
| IS, OOS | In-sample and out-of-sample: the first 70% of the evaluated days and the rest, by default. |

Positions such as `e` and `stop` are integer positions on the run calendar (row numbers), not dates.

### Finance terms

- **Backtest:** replaying a trading rule on past prices to see how it would have done.
- **Sharpe ratio:** average return above the risk-free rate, divided by its volatility, per year. Higher is better.
- **Drawdown:** how far the account is below its previous peak.
- **Mean reversion:** betting that a price that has moved far from its recent average will come back.
- **Pairs trading:** the same bet on the gap between two related stocks (KO and PEP, say), long one and short the other.
- **Cointegration:** a statistical sign that two prices move together over the long run, which is what pairs trading relies on. The Engle–Granger test measures it.
- **Trend following (time-series momentum):** buying what has gone up over the past year and shorting what has gone down.
- **Volatility targeting:** shrinking positions when markets are turbulent and growing them when calm, to keep risk roughly constant.
- **In-sample / out-of-sample:** the period a strategy's settings could have been tuned on, and the later period that tests them honestly.
- **Walk-forward:** repeatedly tuning on a past window and testing on the next one, so every result is out-of-sample.
- **Deflated Sharpe ratio:** a correction for having tried many settings and kept the best.
- **Monte Carlo:** simulating many possible futures. GBM draws returns from a normal distribution; the bootstraps resample blocks of real past returns.
- **VaR / CVaR:** Value at Risk, the loss exceeded only 5% (or 1%) of the time, and the average loss in those worst cases.

## How the results are checked

- **No look-ahead.** The most important backend test changes every price after some date and checks that nothing on or before that date changes, for every strategy and option. A strategy that peeked at the future would fail it.
- **Hand-computed cases.** Small examples worked out by hand (in CLAUDE.md §12) are checked exactly.
- **Reconciliation.** The profit of all trades must add up to the total return.
- **Independent references.** Fast vectorized code is compared with slow, obvious loops written in the tests.
- **Browser tests** drive the real dashboard against an offline API.
- **Planted bugs.** Before each change was accepted, deliberate bugs were inserted into the new code to confirm the tests catch them.

The commands are in the [README](../README.md#tests-and-linters).
