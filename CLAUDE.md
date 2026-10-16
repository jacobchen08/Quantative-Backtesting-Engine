# Quant Engine: Engineering Specification

A backtesting and Monte Carlo risk engine. This file is the source of truth for the AI coding agent and for human reviewers. **Formulas, code snippets, thresholds and tolerances are normative.**

Spec reviewed on 2026-10-15 against numpy 2.5, pandas 3.0, scipy 1.18, statsmodels 0.15, yfinance 1.7, FastAPI 0.143 / Starlette 1.8 / Pydantic 2.14, SQLAlchemy 2.1, SQLite 3.50, React 19, Vite 8, Tailwind CSS 4.3 and Recharts 3.10. Two labels mark where facts came from:
- **(verified)** = library or data behaviour observed directly.
- **(measured)** = a number from a reference prototype run during that review. These numbers come from the prototype, not the final code, so never quote them as results.

## 0. Rules for the Agent (read first)
1. **This file is the source of truth.** Implement formulas, snippets, thresholds and tolerances exactly as written. Do not substitute a "standard" version from memory or silently "improve" anything. If you believe something here is wrong, explain why and wait for approval.
2. **Ask instead of guessing.** If anything is ambiguous, contradictory, or doesn't work with the installed library versions, stop and ask.
3. **Never weaken a test to make it pass.** No `skip`, `xfail`, loosened tolerances or deleted assertions. Find the root cause. If you believe a test contradicts this spec, say so and ask.
4. **Never invent numbers.** Runtimes, Sharpe ratios and returns in code, comments, docs or the README must come from a run you actually performed, quoted with the command that produced them.
5. **Work phase by phase (§16).** At the end of each phase, run the full test suite and linters, show the output, list any deviations from this spec, then stop and wait for review. Commit only after the user approves.
6. **Windows first.** The developer uses Windows 11 and PowerShell, and every documented command must work there. No Makefiles or bash-only scripts; automate with Python scripts.
7. **Record approved deviations** in `docs/METHODOLOGY.md` under "Deviations", with the reason.
8. **Keep the core pure.** `pipeline.py`, `strategies.py`, `sizing.py`, `backtester.py`, `metrics.py`, `stats.py` and `monte_carlo.py` take arrays or Series and return arrays, Series or dataclasses. They do no I/O, touch no database or network, and use no globals. All I/O lives in `data_loader.py` and the API layer.
9. **Document semantics in docstrings.** Every public function states its units, what "the value at date t" means, and its timing.

## 1. Scope
A local, single-user research tool that:
- backtests three daily strategies on US-listed stocks and ETFs: **single-asset mean reversion**, **pairs trading** (statistical arbitrage) and **time-series momentum** (trend following);
- can size any of them with an optional **volatility-targeting overlay**;
- reports **in-sample (IS) and out-of-sample (OOS)** performance with explicit, conservative accounting: an execution lag, per-leg costs, and financing at the T-bill rate;
- projects forward risk with **Monte Carlo** (GBM and block bootstrap), checked against closed-form results;
- shows everything in a React dashboard.

**Stack:** Python (FastAPI, NumPy, pandas, SciPy, statsmodels, SQLAlchemy, yfinance); SQLite by default with optional PostgreSQL; React + TypeScript (Vite, Tailwind CSS v4, Recharts).

**Non-goals (do not build):**
- authentication or multiple users;
- intraday data;
- non-US exchanges or mixed trading calendars;
- live or paper trading;
- schema migrations (tables are created at startup);
- stop-losses, position sizing or leverage beyond what is specified;
- deployment beyond local `uvicorn` + `vite`.

## 2. Global Conventions

### 2.1 Units and indexing
- Returns, rates, weights, drawdowns, VaR and CVaR are **decimals** (0.01 = 1%) throughout the backend. Basis points appear only in the `cost_bps` request field. Percent formatting happens only in the frontend.
- Every time series is a pandas object indexed by **trading date**: tz-naive, sorted ascending, unique. Assert this where data enters the core.
- There are 252 trading days per year.
- Rolling windows are **trailing**, include day t, and use `min_periods=L`. Variances and standard deviations use `ddof=1` (the pandas rolling default) unless stated otherwise.
- The simple return of an asset on day t is `C / C.shift(1) - 1`, where C is the **adjusted** close. Don't use `pct_change`: its fill behaviour changed across pandas versions. Log return: ℓ_t = ln(1 + r_t).

### 2.2 Timing contract (no look-ahead)
- A signal s_t may use data up to and including the close of day t.
- The executed position is p_t = s_{t−k}, where k = `execution_lag` ∈ {0,…,5}, default 1.
  - k = 1 fills the order at the **next** day's close. This is the conservative default.
  - k = 0 fills at the same close that produced the signal. It is optimistic, so offer it only as an option.
- Day-t P&L comes from the position held **into** day t: p_{t−1}·r_t. With k = 1 that is s_{t−2}·r_t.
- The lag is applied in **exactly one place**, the backtester. Strategies return raw signals and never shift them.
- Signals generated in the last k days of the data are never executed.

### 2.3 Risk-free rate and cash
- `risk_free="tbill"` (default) uses the 13-week T-bill yield, Yahoo ticker `^IRX`, which is quoted in **percent**: 5.21 means 5.21% (verified). The rate accruing over day t is the one observed at the previous close: rf_t = IRX_{t−1} / 100 / 252.
- `risk_free=<number>` uses a constant annual rate a, so rf_t = a / 252. Unit tests use constants.
- **Either way, `rf` is a `pd.Series` indexed by the run calendar,** never a NumPy array or a bare float. For a constant rate that is `pd.Series(a / 252, index=calendar)`. Strategy C calls `rf.rolling(...)` (§5.5) and the backtester checks `rf.index` (§6.1), and a NumPy array has neither: `.rolling` raises `AttributeError` (verified). Every function that takes `rf` checks `isinstance(rf, pd.Series)` first and raises `TypeError` otherwise, so a wrong type fails with a clear message instead of an incidental `AttributeError`.
- Capital not invested in long legs, plus short-sale proceeds, earns rf. There are no borrow fees and no financing spread. **A flat strategy therefore earns exactly rf and has exactly zero excess return** (§6.1).

### 2.4 Numerical and serialization rules
- **Same-row tolerances only.** Guards for degenerate statistics use values from the same date. A threshold built from a full-series statistic (such as a median of the whole series) is look-ahead and will fail the perturbation test.
- **Undefined means null.** A statistic with a near-zero denominator, no trades or no exposure is `None`/`null`. It is never `NaN`, `inf` or a huge finite number. The standard deviation of a constant series is often not exactly 0: rounding leaves a tiny value that depends on the constant, and numpy's `np.std` and pandas' `Series.std` behave alike. For 252 copies of 0.1 both return 1.4e-17, which would give a naive Sharpe ratio of about 7e15; for 252 copies of 0.3 both return exactly 0 (verified with numpy 2.5 and pandas 3.0). The threshold rules must produce `null` either way. Each statistic's threshold is given where it is defined.
- **JSON.** Starlette answers with HTTP 500 if a response contains `NaN` or `inf` (verified). Convert at the API boundary with one helper, `to_jsonable()`. It maps non-finite floats to `None`, numpy arrays and scalars to Python types, and dates to `"YYYY-MM-DD"`.
- **pandas 3 Copy-on-Write.** Never use chained assignment (`df["a"][mask] = v`); use `.loc` or direct Series assignment.
- **RNG.** Always use `np.random.default_rng(seed)`. Never use `np.random.seed` or the legacy global functions.

## 3. Repository Layout and Tooling
```
quant-engine/
  CLAUDE.md  README.md  .gitignore  .env.example  docker-compose.yml
  .github/workflows/ci.yml
  backend/
    pyproject.toml           # ruff + pytest configuration
    requirements.txt         # exact pins (pip freeze of runtime deps)
    requirements-dev.txt     # pytest, ruff
    app/
      __init__.py            # __version__ = engine version
      main.py                # FastAPI app, lifespan (create tables), CORS, routes
      config.py              # env: DATABASE_URL, CORS_ORIGINS, LOG_LEVEL
      database.py            # engine + session dependency
      models.py              # ORM tables (§4.1)
      schemas.py             # Pydantic v2 request/response models (§11)
      serialization.py       # to_jsonable()
      services/
        calendar.py          # today_ny() (injectable clock), alignment helpers
        data_loader.py       # cache + the single network seam fetch_from_yahoo()
        strategies.py        # scores, entry/exit rules, unit weights  (pure)
        sizing.py            # volatility-targeting overlay             (pure)
        pipeline.py          # run_strategy() (§5.0) and run_sensitivity() (§9)  (pure)
        backtester.py        # lag, returns, costs, equity, trades      (pure)
        metrics.py           # performance metrics and warnings        (pure)
        stats.py             # Engle–Granger, half-life                 (pure)
        monte_carlo.py       # GBM, bootstrap, closed form              (pure)
    scripts/bench.py         # measured timings: the only source for runtime claims
    tests/                   # §12
  frontend/                  # Vite React-TS app (§13)
  docs/METHODOLOGY.md        # §15
```
- **Python:** 3.12 or later (developed on 3.14). List `tzdata` explicitly in the requirements: Windows has no system time-zone database, and `zoneinfo` needs one.
- **PostgreSQL:** driver `psycopg[binary]`, URL form `postgresql+psycopg://…`. `docker-compose.yml` provides an optional Postgres 16.
- **Lint:** `ruff check` and `ruff format --check` for the backend; ESLint from the Vite template for the frontend.
- **CI (GitHub Actions, every push):**
  - backend lint and tests on Python 3.12 and 3.14;
  - frontend `npm ci`, lint, `vitest run`, and `npm run build`.
- **Config files:**
  - `.env.example` lists every environment variable with its default.
  - `.gitignore` covers `.venv/`, `node_modules/`, `*.db`, `__pycache__/`, `.env` and `dist/`.

## 4. Data Layer

### 4.1 Database
- `DATABASE_URL` defaults to `sqlite:///./quant.db`. For SQLite, pass `connect_args={"check_same_thread": False}`. Create tables with `metadata.create_all()` in the FastAPI lifespan.
- `historical_prices`:
  - `ticker` String(16); `date` Date.
  - `open`, `high`, `low`, `close`, `adj_close`: Float.
  - `volume`: BigInteger, nullable. Split-adjusted volumes can exceed 2³¹.
  - Primary key (`ticker`, `date`).
- `ticker_coverage`:
  - `ticker` (primary key), `cov_start` Date, `cov_end` Date, `fetched_at` DateTime (UTC).
  - One contiguous **requested** range per ticker.
- `backtest_runs`:
  - `id` (primary key), `created_at`, `engine_version`, `request_json`, `metrics_json`.
  - Every `/api/backtest` call is stored here. The internal runs inside `/api/sensitivity` and `/api/monte-carlo` are not.

### 4.2 Cache rules
- `today_ny()` returns the current date in America/New_York; tests monkeypatch it. A bar is final only if its date is earlier than `today_ny()`. **Never store a bar dated `today_ny()` or later**, because during market hours Yahoo returns a partial bar.
- A request for [start, end] needs data through `end* = min(end, today_ny() − 1 day)`.
- **Cache hit:** the ticker's coverage row has `cov_start ≤ start` and `cov_end ≥ end*`. Serve from the database with no network call.
- **Cache miss:** refetch the whole union `[min(start, cov_start), max(end*, cov_end)]` in one download. If the ticker has no coverage row yet, fetch just `[start, end*]`.
  - Upsert every returned row with `ON CONFLICT (ticker, date) DO UPDATE` (overwrite), then set coverage to the union.
  - Write the rows and the coverage in **one transaction**.
  - Two concurrent requests for the same ticker are safe: both upserts overwrite with the same data.
- **Why the union refetch:** Yahoo back-adjusts history. Every dividend rewrites all earlier adjusted closes, and every split rewrites all earlier closes (verified: KO's 2024-01-02 adjusted close reads 55.31 against a raw close of 59.82 when fetched in 2026). Rows fetched at different times are therefore on different adjustment bases, and stitching them together creates fake returns at the seam. Refetching the whole range keeps each ticker on one basis.
- Coverage records the **requested** range, not the dates actually returned. Otherwise a ticker that listed after `start` would be refetched on every request.
- Upserts use the dialect-specific insert (`sqlalchemy.dialects.sqlite.insert` / `sqlalchemy.dialects.postgresql.insert`) with `.on_conflict_do_update(...)`.
  - **Chunk at 1,000 rows per statement.** SQLite accepts at most 32,766 bound parameters per statement (verified), and 20 years of one ticker × 8 columns exceeds that.

### 4.3 Yahoo access (single network seam)
- All network access goes through one function, `fetch_from_yahoo(ticker, start, end_inclusive) -> DataFrame`. Nothing else imports yfinance, and tests replace this function.
- Call it once per ticker, and log each call (ticker, range, row count, ms) at INFO level:
  ```python
  yf.download(ticker, start=start, end=end_inclusive + timedelta(days=1),
              auto_adjust=False, actions=False, multi_level_index=False,
              progress=False, threads=False)
  ```
  Every argument here is needed. In yfinance 1.7 (verified):
  - `end` is **exclusive**;
  - `auto_adjust` defaults to True, which drops `Adj Close`;
  - `multi_level_index` defaults to True.
- Flatten columns defensively anyway: if a MultiIndex appears, keep the price level. Rename columns to snake_case and normalize the index to tz-naive dates.
- **Errors.** Never retry in a loop.

  | Condition | Response |
  |---|---|
  | Empty frame. yfinance returns an empty frame for unknown tickers instead of raising (verified). | 404 `"No data for {ticker} between {start} and {end}"` |
  | `yfinance.exceptions.YFRateLimitError` | 503 |
  | Any other download exception, or a timeout | 502 |

### 4.4 Cleaning (tradable tickers)
- Drop rows with a missing `close` or `adj_close`, de-duplicate dates (keep the last), and sort.
- Drop rows with `close ≤ 0` or `adj_close ≤ 0`, and add a warning listing their dates.
- Warn, but don't drop, when |daily return| > 50%. It may be a bad print or an unadjusted split.
- Warn, but don't drop, when 5 or more consecutive adjusted closes are identical. That may be stale data or a trading halt. It matters for vol targeting, whose volatility estimate shrinks during a stale run and pushes the position size up toward the leverage cap.
- `^IRX` is a rate, not a price. Skip the positivity check, and **drop** any row whose close is missing or outside [−1, 25] (percent). Don't clip at 0: yields went negative (−0.105%) during 2015–2025 (verified).
  - Drop the row; never keep it as NaN. `pd.merge_asof` (§4.5) does not skip NaN rows. It returns the NaN (verified), and the "any NaN → 422" rule would then fail every backtest that spans that date.
  - With the row dropped, the lookup falls back to the previous valid rate (verified).

### 4.5 Alignment
- A run's calendar is the **intersection** (inner join on date) of the adjusted closes of every tradable ticker in the run. That always **includes SPY**, which is loaded for the benchmark and for beta. If SPY is also one of the request's legs, load it once.
- Compute returns **after** joining, so a return correctly spans any dropped date. **Never forward-fill prices.** Stale prices create fake zero-return days.
- **Risk-free series.** rf on day t is the last `^IRX` close dated **strictly before** t, i.e. the rate observed at the previous close. That observation can be more than one stock-trading day old, for two reasons:
  - The ^IRX series occasionally has no row on a day the stock market trades. Verified on 2015–2025 Yahoo data: this happened once, on 2016-11-11. ^IRX *does* have rows on Columbus Day 2023–2025 and Veterans Day 2024.
  - Bad prints are dropped (§4.4).

  Implement it as an as-of lookup over the cleaned series (no NaN rows), not as a left join plus a shift:
  ```python
  left = pd.DataFrame({"date": calendar})
  right = irx.rename("irx").rename_axis("date").reset_index()
  asof = pd.merge_asof(left, right, on="date", allow_exact_matches=False)["irx"]
  rf = pd.Series(asof.to_numpy() / 100 / 252, index=calendar, name="rf")   # a Series on the calendar (§2.3)
  ```
  - Load `^IRX` from **start − 14 calendar days**, so the first calendar day always has an earlier observation.
  - This as-of lookup is a forward fill, which is fine for rates. Prices must never be forward-filled.
  - Never back-fill.
  - Any NaN that remains → 422.
- **One date resolution.** Normalize every date index to `datetime64[ns]` (`.as_unit("ns")`) wherever data enters the system: loader output, database reads and test fixtures. yfinance 1.7 returns `datetime64[s]` (verified), and pandas 3 refuses some operations across resolutions: `merge_asof` raises `MergeError` on `[s]` vs `[ns]` keys (verified).

## 5. Strategies (`services/strategies.py`, `services/sizing.py`)
Each strategy returns:
- a **score**: z for Strategies A and B, the trailing excess return m for Strategy C;
- a raw signal s_t ∈ {−1, 0, +1};
- **unit weights**: gross exposure 1, before any sizing;
- diagnostic series.

Strategies never shift. The optional vol-targeting overlay (§5.6) rescales the unit weights.

### 5.0 Call order (`services/pipeline.py`)
The overlay has to mask the score **before** the entry rule produces a signal. For pairs, though, the vol forecast needs the rolling β that the strategy computes. So the work can't be done as "the strategy returns a signal, then sizing rescales it". It has to be interleaved.

`run_strategy(prices, rf, params)` is the **single entry point**, used by the API, the sensitivity grid and the tests. It runs these steps in this exact order:
1. **Raw score** (`strategies.py`): z for A; z and the rolling β for B; the raw daily m for C.
2. **Vol forecast** (`sizing.py`, only if `vol_target` is set): build u (§5.6; for B it uses step 1's β), then `sigma`. Then mask the score: `score = score.where(sigma.notna())`.
3. **Signal** (`strategies.py`): the entry/exit rule for A and B. For C, the monthly sampling (if monthly) followed by the sign. The score as it stands after this step (masked, and sampled for C) is the score reported by the API and used to find t₀.
4. **Unit weights** (`strategies.py`), including `beta_used` for B. They come out as a DataFrame on the run calendar whose column labels are the **tickers**, matching the returns frame `R` (§6.1).
5. **Sizing** (`sizing.py`, only if `vol_target` is set): scale, refresh, W (§5.6). Without the overlay, W = the unit weights.
6. **Evaluation**: t₀ = the position of the first non-NaN reported score; then `resolve_split` (§6.3), then the backtester (§6.1).

### 5.1 Entry/exit rule (shared by both strategies)
- **Short** when z > z_entry; **long** when z < −z_entry.
- **Exit a short** once z < z_exit; **exit a long** once z > −z_exit. This is the crossing rule.
- A NaN z (warm-up or a degenerate window) means **flat**.
- Comparisons are strict, so z exactly at a threshold triggers nothing. A direct flip between short and long in one bar is allowed.
- Defaults: L = 60, z_entry = 2.0, z_exit = 0.5. Valid ranges: 0 < z_entry ≤ 5 and 0 ≤ z_exit < z_entry.

**Do not use the symmetric band rule** ("flat when |z| < z_exit"). With daily bars, z can jump straight across the band, for example from +0.7 to −0.7. The band rule then keeps a short open while the spread sits *below* its mean, a wrong-way position, until z drifts back into the band. In the §12 example that is four extra days (measured).

**Required implementation.** It uses two independent hysteresis legs, which can't both be on when 0 ≤ z_exit < z_entry. It matched a reference state machine bit for bit on 3,000 random sequences (verified):
```python
short_m = pd.Series(np.nan, index=z.index)
short_m[z > z_entry] = 1.0
short_m[z < z_exit] = 0.0
long_m = pd.Series(np.nan, index=z.index)
long_m[z < -z_entry] = 1.0
long_m[z > -z_exit] = 0.0
short_m[z.isna()] = 0.0
long_m[z.isna()] = 0.0
signal = long_m.ffill().fillna(0.0) - short_m.ffill().fillna(0.0)
```

### 5.2 Strategy A: single-asset mean reversion (Bollinger z-score on the adjusted close)
```python
lc = np.log(close / close.iloc[0])            # rebased log price: causal and translation-invariant
r = lc.rolling(L, min_periods=L)
m, v = r.mean(), r.var()
z = (lc - m) / np.sqrt(v.where(v > 1e-12))    # a constant window has var exactly 0.0 (verified) -> NaN -> flat
```
The unit weight equals the signal: one leg, gross exposure 1, as `signal.to_frame(ticker)`.

### 5.3 Strategy B: pairs trading (Y = `tickers[0]`, X = `tickers[1]`, adjusted closes)
```python
y, x = np.log(Y / Y.iloc[0]), np.log(X / X.iloc[0])     # rebased: same z, less rounding error
ry, rx = y.rolling(L, min_periods=L), x.rolling(L, min_periods=L)
my, mx, vy, vx = ry.mean(), rx.mean(), ry.var(), rx.var()
beta = ry.cov(x) / vx.where(vx > 1e-12)                  # rolling OLS slope of y on x
resid_var = vy - beta**2 * vx                            # = SSR / (L - 1)
spread = (y - my) - beta * (x - mx)                      # OLS residual of today's point
z = spread / np.sqrt(resid_var.where(resid_var > 1e-8 * vy))
```
**Why this z-score.** It measures today's point against the OLS fit on the trailing window.
- Do **not** compute y − β_t·x and then take a rolling mean and std of that. It mixes spreads built with different β values.
- The identity uses `ddof=1` variances, so it matches the residual std with `ddof=1` (divide by L−1). It does not match the textbook OLS standard error, which divides by L−2.

**Why the 1e-8 threshold.**
- For exactly collinear log prices, rounding leaves `resid_var / vy` at about 1e-11 to 1e-10 (measured). A 1e-12 guard lets that through and produces noise z-scores.
- Real pairs sit far above 1e-8: a simulated share-class pair measured about 1e-4.
- Rebasing cuts the rounding noise from up to 2.5e-10 to about 4e-12 (measured).
- Don't clip negative values to 0. A zero denominator gives ±inf z-scores, and those trigger trades.

**Hedge ratio:**
- `hedge_mode="fixed_at_entry"` (default): `beta_used = beta.where(signal.ne(signal.shift()) & signal.ne(0)).ffill()`. β is captured on the day a position opens (an entry from flat, or a direct flip) and held for the whole trade, so daily drift in β creates no turnover.
- `hedge_mode="rolling"`: `beta_used = beta`. The pair is re-hedged daily, and the drift creates turnover and cost.
- Exits are always decided on the rolling z, even though the frozen-β spread is what's held. This is a known simplification: an exit z built on the frozen β would depend on the trade's entry date and could not be vectorized. Document it.

**Unit leg weights** (gross exposure 1):
```python
open_ = signal != 0
w_y = np.where(open_, signal / (1 + beta_used.abs()), 0.0)
w_x = np.where(open_, -signal * beta_used / (1 + beta_used.abs()), 0.0)
```
- `np.where` returns unlabeled arrays. Wrap them as `pd.DataFrame({tickers[0]: w_y, tickers[1]: w_x}, index=calendar)`. The column labels must be the tickers so they match `R` (§6.1); mismatched labels silently produce zero P&L.
- Weights must be exactly 0.0 when flat. In pandas, a NaN β (warm-up) times a zero signal is still NaN.
- If `beta_used` is NaN while `signal != 0`, raise an error. With the guards above in place this can't happen.
- A long spread position (s = +1) holds, per unit, long $1 of Y and short $β of X, because Δspread ≈ r_Y − β·r_X.

### 5.4 Diagnostics (`services/stats.py`, pairs only)
The IS diagnostic window is [first loaded date, split_date). Warm-up days are legitimate in-sample data here.
- **Engle–Granger p-value:** `statsmodels.tsa.stattools.coint(y, x, trend="c", autolag="aic")[1]`, with Y regressed on X.
  - Never use plain `adfuller` on OLS residuals. Its critical values ignore that β was estimated. On independent random walks it flagged false cointegration 15% of the time at the 5% level, against 6% for `coint` (measured).
  - Capture warnings with `warnings.catch_warnings(record=True)`.
  - If `coint` emits a `CollinearityWarning`, report `null` and add a warning. On exactly collinear inputs it returns p = 0, i.e. "perfectly cointegrated" (verified).
- **Static OLS** α̂ and β̂, on the same window.
- **Half-life.** Fit Δs_t = a + b·s_{t−1} by least squares on the static residual s = y − α̂ − β̂x, then compute HL = −ln 2 / ln(1 + b). Return `null` unless −1 < b < 0.
- **OOS Engle–Granger p-value** on [split_date, end], as a stability check. It is only displayed and never used for any decision.

### 5.5 Strategy C: time-series momentum (trend following on one adjusted close)
This follows Moskowitz, Ooi & Pedersen (2012): go long when the asset's trailing excess return is positive, short when it is negative.
```python
m = np.log(close / close.shift(L)) - rf.rolling(L, min_periods=L).sum()   # trailing L-day excess log return
# overlay on: step 2 of §5.0 masks m HERE (m = m.where(sigma.notna())), before the sampling below
if signal_frequency == "monthly":                                          # default
    m = m.where(first_of_period(m.index, "M")).ffill()                     # decide on the first trading day of each month
signal = np.sign(m).fillna(0.0)                                            # exactly 0 or NaN -> flat
```
- **Lookback.** L defaults to 252 trading days (12 months); valid range 21–504. `rf` is the §4.5 Series. With a constant `risk_free` it is a constant-valued Series (§2.3), never a bare float.
- **Score.** The score is the sampled m. It is NaN until the first month start after a full lookback, so t₀ (§6.2) is a month start.
- **Why monthly by default.** Checked daily, the sign whipsaws whenever the 12-month return hovers near zero. Measured on SPY, 2006–2025, at 10 bps:

  | Signal checked | Position changes/yr | Turnover | Cost drag | Sharpe |
  |---|---|---|---|---|
  | daily | 4.9 | 9.8×/yr | 0.98%/yr | 0.21 |
  | monthly | 1.0 | 2.0×/yr | 0.20%/yr | 0.31 |

  `signal_frequency="daily"` stays available so the dashboard can show this.
- **Weights.** The unit weight equals the signal: one leg, gross exposure 1, as `signal.to_frame(ticker)`. There is no entry/exit hysteresis, because the sign *is* the position.
- **Period helper** (also used by §5.6). It must use a positional `Series.shift`:
  ```python
  def first_of_period(index, freq):            # True on the first trading day of each week ("W") or month ("M")
      p = pd.Series(index.to_period(freq), index=index)
      return p.ne(p.shift(1))
  ```
  Do **not** compare `index.to_period(freq)` with `index.to_period(freq).shift(1)`. `PeriodIndex.shift` moves each period *value* forward by one period instead of lagging by one row, so every day compares unequal and becomes a "period start". In the review prototype this silently turned monthly rebalancing into daily rebalancing (verified).

### 5.6 Volatility-targeting overlay (optional, any strategy; `services/sizing.py`)
The overlay scales a strategy's unit weights so its expected volatility stays roughly constant: smaller positions when the market is turbulent, larger when it is calm, up to a leverage cap. It is off by default (`vol_target = null`).

**Unit-position return u** (the thing being sized):
- Strategies A and C: the asset's daily return r.
- Strategy B: u_t = (r_Y,t − β_{t−1}·r_X,t) / (1 + |β_{t−1}|). This is the day-t return of a unit long-spread position formed with yesterday's rolling β (§5.3). Use the rolling β, not `beta_used`, so that u exists even while the strategy is flat.

The overlay runs in two separate places in the §5.0 call order. Keep them as two functions; don't merge them into one call.

**Step 2 of §5.0: forecast and mask.** This runs *before* any signal exists.
```python
sigma = np.sqrt(252 * u.pow(2).ewm(halflife=vol_halflife, adjust=True, min_periods=vol_halflife).mean())
score = score.where(sigma.notna())                              # mask the warm-up before the entry rule
```

**Step 5 of §5.0: scale, refresh and size.** This runs *after* steps 3–4 have produced `signal` and `unit_weights`.
```python
scale = (vol_target / sigma).clip(upper=max_leverage)          # sigma == 0 -> inf -> clipped to the cap
if vol_refresh == "daily":
    refresh = pd.Series(True, index=sigma.index)
else:
    refresh = first_of_period(sigma.index, "W" if vol_refresh == "weekly" else "M")
scale_used = scale.where(refresh | signal.ne(signal.shift())).ffill()
W = unit_weights.mul(scale_used, axis=0)
W.loc[signal == 0] = 0.0                                        # exactly 0 when flat (NaN x 0 is NaN)
```
If `W` contains NaN on any day, raise.

- **The forecast.** `sigma` is a zero-mean, exponentially weighted volatility forecast, as in RiskMetrics.
  - `ewm(adjust=True)` equals the explicit normalized weighted average Σ(1−α)^i·u²_{t−i} / Σ(1−α)^i, with α = 1 − 2^(−1/H) (verified to 4e-16).
  - It uses data through the close of day t. The backtester's lag k applies to the sized weights like everything else.
  - The default half-life is 42 trading days, close to the 60-day centre of mass used by Moskowitz, Ooi & Pedersen.
- **Warm-up.** With the overlay on, set the strategy's score to NaN wherever `sigma` is NaN. Do this **before** the entry rule, and before Strategy C's monthly sampling (step 2 of §5.0). Then t₀ (§6.2) is automatically the first day on which both the score and the forecast exist, and no position can carry over from the warm-up.
- **Refresh.** A signal change always resizes immediately, so every entry is sized with the current forecast. Between signal changes, the size updates only on `vol_refresh` days. Weekly is the default. Measured on SPY trend, 2006–2025, with a 10% target:

  | Refresh | Turnover | Yearly realized-vol range |
  |---|---|---|
  | daily | 2.6×/yr | 6.9–13.7% |
  | weekly | 2.1×/yr | 6.8–14.0% |
  | monthly | 1.7×/yr | 6.6–17.0% |
- **Leverage.** Leverage above 1 is financed at rf with no extra formula: §6.1's total = rf + Σw(r − rf) lets the cash weight go negative (verified exact for w = 2). Leverage also raises ruin risk: at 2×, a −50% day ruins a long position (§6.2).
- **What it achieves** (measured on SPY trend with the monthly signal and a 10% target):
  - full-period vol fell from 19.9% to 10.3%;
  - the yearly vol range narrowed from 6.7–41.3% to 6.9–13.7%;
  - Sharpe rose from 0.31 to 0.45.

  The forecast lags sudden volatility spikes, so realized vol overshoots the target during crashes. Document this.
- **Diagnostic: `pct_days_at_cap`.** The share of **invested days in the evaluation window** (days where `held_id` is not null, §7) on which the gross exposure held into the day, Σ_i |w_{i,t−1}|, is ≥ `max_leverage` × (1 − 1e-9).
  - It is measured on executed, held positions, after the lag, the same way as `pct_days_in_market` and `avg_gross_exposure`.
  - It is equivalent to checking `scale_used.shift(k + 1) ≥ max_leverage` (verified identical to 2e-16), but needs no shift arithmetic.
  - It is `null` when the overlay is off, and `null` when the strategy never held a position.

## 6. Backtest Accounting (`services/backtester.py`)

### 6.1 Daily returns and costs
**Inputs.** All are keyword arguments, and none has a default.
- `R`: simple returns, one column per leg.
- `W`: raw (unlagged) target weights, with the **same index and column labels** as `R`.
- `s`: the raw signal, with the same index as `R`.
- `rf`: the daily risk-free rate (§4.5), with the same index as `R`; no NaN.
- `t0`: the integer position of the first non-NaN score (§6.2). It must satisfy t0 + k ≥ 1, so that a baseline day exists. Unit tests pass it explicitly. It must have **no default**: a forgotten t0 would quietly put the warm-up days back into the metrics.
- `k`, `cost_bps`, `E0`.
- `split_date`: a date that has already been resolved. The pipeline passes the output of `resolve_split` (§6.3); unit tests pass any date that leaves at least one IS and one OOS day.

**Validate on entry** and raise `ValueError` on any failure:
- `W.index`, `s.index` and `rf.index` all equal `R.index`, and `W.columns` equals `R.columns`.
  - Why: pandas aligns on labels. Weights labelled `Y`/`X` multiplied by returns labelled `KO`/`PEP` give **all-zero P&L with no error** (verified).
- `(s.iloc[:t0] == 0).all()`: there is no signal before t0.
- At least **1** IS day and **1** OOS evaluated day. The 252/126 minimums are **not** checked here (§6.3), so 6-day toy series work in unit tests.

Compute everything on the full run calendar:
```python
c = cost_bps / 10_000                      # one-way, per unit of traded notional, per leg
w = W.shift(k).fillna(0.0)                 # executed end-of-day weights
pos = s.shift(k).fillna(0.0)               # executed position (used to identify trades)
w_prev = w.shift(1).fillna(0.0)            # weights held into day t
R = R.copy(); R.iloc[0] = 0.0              # only the first row may be NaN
assert not R.isna().any().any() and not rf.isna().any()   # pandas .sum() would hide NaN as 0
gross_excess = (w_prev * R.sub(rf, axis=0)).sum(axis=1)
turnover = (w - w_prev).abs().sum(axis=1)  # every leg counts
cost = c * turnover                        # charged on the trade day
excess = gross_excess - cost               # daily return in excess of rf
total = rf + excess                        # daily total return (cash earns rf)
```
- In formula form: total_t = rf_t + Σ_i w_{i,t−1}(r_{i,t} − rf_t) − cost_t. Cash and short proceeds earn rf, and a flat day's excess return is exactly 0.
- Round-trip costs are 2c × gross exposure. Gross exposure is 1 without the overlay, for one asset or a pair. With vol targeting, size changes inside a trade also cost money; that turnover is charged and attributed to the trade (§7.1).
- **Approximations** (document both):
  1. Cost is charged as a fraction of the day's starting equity. This ignores the second-order term cost × same-day return.
  2. Target weights are fractions of current equity and are reset to target daily. Turnover from price drift is ignored, so only changes in target weights cost money.

### 6.2 Evaluation window and equity
- t₀ is the integer position (in the run calendar) of the first non-NaN **score**. That is z for Strategies A and B and the sampled m for Strategy C, after the vol-forecast mask when the overlay is on (§5.6). If the score is NaN on every day (too little data, or every window degenerate), return 422. The **first evaluated day**, e, is k trading days after t₀: the first day a position can be executed and a cost can occur.
- Metrics, the IS/OOS split and Monte Carlo inputs use only days on or after e. Including flat warm-up days would dilute volatility and the Sharpe ratio.
- **Equity:**
  - A **baseline point**, E₀, is dated the trading day before e.
  - After that, E_t = E₀ ∏_{e≤u≤t}(1 + total_u).
  - Drawdown is DD_t = E_t / cummax(E)_t − 1. The running max includes the baseline, so a loss on day e shows up as a drawdown.
- **Ruin.** A single-asset short loses more than 100% on any day the stock more than doubles (GME rose 134.8% on 2021-01-27; verified). Let r be the first evaluated day with total_r ≤ −1. Then:
  - **The evaluation window ends at r.** Series, metrics and trades all stop there.
  - total_r is clipped to −1, so E_r = 0 and the drawdown is −100%. The model assumes limited liability and no margin calls.
  - Clipping raises that day's excess return. Credit the difference (the clipped excess minus the unclipped excess, which is ≥ 0) to the trade held into r (`held_id` at r), so the §7.2 reconciliation still holds. That trade always exists: a flat day earns rf minus at most 2c, so it can never reach −100%.
  - Positions still open at r are reported as `open` trades.
  - Set `ruined = true` and add a warning.
  - A segment that would start after r gets `null` metrics.
  - Monte Carlo is rejected for any window that contains r, because ln(1 + R) is undefined there.
  - `split_date` and the IS/OOS length checks (§6.3) use the untruncated calendar, so they never depend on whether or when ruin happens.

### 6.3 In-sample / out-of-sample
- **`resolve_split(eval_dates, requested)`** is a pure helper in `backtester.py`. The pipeline calls it, and the sensitivity grid calls it too, so both resolve the split identically.
  - With no requested date, it returns the date at index ⌊0.7·n_eval⌋ of the evaluated days (untruncated by ruin).
  - A requested date that isn't a trading day becomes the first trading day on or after it.
  - It **enforces the minimums of 252 IS days and 126 OOS days**. If either is short, it raises `InsufficientDataError` with a message giving both counts, which the API maps to 422.
  - This check lives here, at the service boundary, and not in the backtester (§6.1). That is how unit tests can call the backtester on short toy series.
- IS = evaluated days with date < `split_date`. OOS = evaluated days with date ≥ `split_date`.
- Compute signals on the full series, then slice. Trailing windows are causal, so OOS signals correctly use pre-split history.
- Each segment's metrics use only that segment's daily returns:
  - its equity is rebased to E₀, with the baseline at the last day before the segment;
  - its max drawdown is measured within the segment.

### 6.4 Benchmarks
- Strategies A and C are benchmarked against buy-and-hold of the traded ticker. Strategy B is benchmarked against buy-and-hold of SPY. Benchmarks are never vol-targeted.
- Benchmarks use the same evaluated window, rebased to E₀, with no costs and no cash (w = 1).
- `benchmark_metrics` has the same fields as the strategy metrics. The trade and cost fields are `null`, and `pct_days_in_market` is 1.

## 7. Trades (`services/backtester.py`)
A trade is a maximal run of non-zero, same-sign **executed** position, `pos`. Never identify trades from the unlagged signal.
```python
trade_id = (pos.ne(pos.shift()) & pos.ne(0)).cumsum().where(pos.ne(0))   # NaN when flat
held_id = trade_id.shift(1)                                              # the trade that earns day t's P&L
```

### 7.1 Attribution
Attribution follows P&L timing, not the position mask. Masking by `pos != 0` drops each trade's last-day return and its exit cost.
```python
a = w_prev.abs().sum(axis=1); b = w.abs().sum(axis=1)
exit_frac = (a / (a + b).where(a + b > 0)).fillna(0.0)
to_held = gross_excess - cost * exit_frac       # attributed to held_id
to_trade = -(cost - cost * exit_frac)           # attributed to trade_id
pnl = (pd.concat([pd.DataFrame({"id": held_id, "v": to_held}),
                  pd.DataFrame({"id": trade_id, "v": to_trade})])
       .dropna(subset=["id"]).groupby("id")["v"].sum())
```
- This is exact for entries, exits and direct flips. In `"rolling"` mode, the cost of in-trade rebalancing lands entirely on the same trade.
- A trade's P&L is its excess return as a fraction of equity, summed arithmetically.

### 7.2 Reconciliation (asserted in tests)
- Over the evaluation window, the P&L of all trades, closed and open, sums to the total excess return within 1e-9 (measured worst case: 4e-16). Flat days contribute exactly 0.
- The identity does **not** hold within the IS or OOS slice separately. A trade belongs to the segment in which it was **entered**, even if it closes after the split.
- This can't leak OOS information into parameter choice, because selection uses only daily-return Sharpe ratios, never trade statistics.

### 7.3 Trade bookkeeping
Use these exact definitions. The "obvious" versions are wrong (verified). If ruin occurred, run them on the series truncated at r (§6.2), so "the last day" is r.
```python
ids = pnl.index                                   # every trade, including one entered on the final day
entry_date = trade_id.index.to_series().groupby(trade_id).first()   # first date where trade_id == id
last_held = held_id.index.to_series().groupby(held_id).last()       # last date where held_id == id
holding_days = held_id.value_counts().reindex(ids, fill_value=0)
is_open = pd.Series(ids == trade_id.iloc[-1], index=ids)            # still held at the end of the last day
exit_date = last_held.reindex(ids).where(~is_open)                  # NaT -> null for open trades
```
- **`exit_date` is the last date where `held_id == id`.** That is the day the exit order fills. Do **not** use the last date where `trade_id == id`: that is the day *before* the exit. In the §12 toy it lands one bar early, and on a flip on the final day, three bars early.
- **A trade entered on the final day** appears in `trade_id` but never in `held_id`. It is `open`, has `holding_days = 0` (that's why the `reindex` is there), and `pnl = −entry cost`. Without the `reindex`, `held_id.value_counts()` has no row for it (verified).
- `signal_date` = `entry_date` minus k trading days. `exit_signal_date` = `exit_date` minus k trading days (null for open trades).
- Open trades are counted separately and excluded from the win rate.
- Trade ids are 1-based integers in the API. pandas holds them as floats because of the NaNs, so cast them.
- **Trade record fields:**
  - `id`, `side` ("long"/"short"; for pairs, "long" means long the spread: long Y, short X), `segment` ("is"/"oos"), `open`;
  - `signal_date` (k trading days before `entry_date`) and `entry_date`;
  - `exit_signal_date` and `exit_date` (`null` while open);
  - `holding_days`, `pnl`;
  - `beta_at_entry`: `beta_used` on `signal_date` for pairs, `null` otherwise.

## 8. Metrics and Warnings (`services/metrics.py`)
- Every metric is computed per segment (IS, OOS and full) over that segment's n evaluated days. Standard deviations use `ddof=1`.
- "No exposure" means `pct_days_in_market == 0`.
- Excess returns drive the Sharpe ratio, Sortino ratio and beta. Total returns drive total return, CAGR, volatility and drawdown.

| Field | Definition | `null` when |
|---|---|---|
| `n_days` | number of evaluated days | — |
| `total_return` | E_n/E₀ − 1 | — |
| `cagr` | (E_n/E₀)^(252/n) − 1 | E_n ≤ 0 |
| `ann_vol` | std(total)·√252 | n < 2 (a one-day segment, possible only when ruin truncates it, has an undefined std) |
| `sharpe` | mean(excess)/std(excess)·√252 | n < 2, std(excess) < 1e-12, or no exposure |
| `sharpe_ci95` | [SR_d ± 1.96·√((1 + SR_d²/2)/n)]·√252, where SR_d is the daily Sharpe (Lo 2002; assumes iid returns) | `sharpe` is null |
| `sortino` | mean(excess)/DD·√252, with DD = √(mean(min(excess, 0)²)) over **all** n days | n < 2, DD < 1e-12, or no exposure |
| `max_drawdown` | min_t DD_t (≤ 0) | — |
| `beta_spy`, `corr_spy` | OLS slope and correlation of strategy excess return on SPY excess return (r_SPY − rf) | n < 2, no exposure, or either std < 1e-12 |
| `pct_days_in_market` | share of days where `held_id` is not null | — |
| `n_closed_trades`, `n_open_trades` | trades entered in the segment | benchmark |
| `win_rate` | share of closed trades with `pnl > 0` | no closed trades |
| `avg_holding_days` | mean holding days of closed trades | no closed trades |
| `ann_turnover` | mean(turnover)·252 | benchmark |
| `ann_cost_drag` | mean(cost)·252 | benchmark |
| `avg_gross_exposure` | mean over all n days of Σ_i \|w_{i,t−1}\|, the exposure held into each day (flat days count as 0) | — (1 for benchmarks) |

**Warnings** are a list of human-readable strings. Generate exactly these:
- IS Engle–Granger p ≥ 0.05: "no evidence the pair is cointegrated in-sample".
- The Engle–Granger test raised a collinearity warning, so its p-value is `null`.
- Fewer than 10 closed trades in a segment: "trade statistics are unreliable".
- `beta_used ≤ 0` at any entry: "the hedge adds exposure instead of offsetting it".
- Rows dropped for non-positive prices, daily returns above 50%, and runs of 5 or more identical closes (§4.4).
- Ruin (§6.2).
- Vol targeting at the leverage cap on more than 25% of invested days: "the volatility target is not reachable within max_leverage".

## 9. Parameter Sensitivity (`POST /api/sensitivity`)
- **Default grids.** All other parameters, including the vol-targeting settings, come from the request.
  - Strategies A and B: L ∈ {20, 40, 60, 90, 120} × z_entry ∈ {1.5, 2.0, 2.5, 3.0}, i.e. 20 configurations. `z_exit` comes from the request.
  - Strategy C: L ∈ {21, 63, 126, 252, 378} (1, 3, 6, 12 and 18 months), i.e. 5 configurations. There is no z_entry axis, so `z_entry_values` is `null` and the Sharpe matrix has a single column.
- A request may override the grid, with at most 8 values per axis. Every z_entry must be greater than z_exit. For Strategy C, overriding `z_entry_values` is ignored.
- **Implementation.** `pipeline.run_sensitivity(prices, rf, params, grid)` is a pure function that implements the grid; the endpoint only calls it. It loops over the configurations (this loop is allowed), in three steps:
  1. **Resolve the split once**, for the request's own configuration: `split_date = resolve_split(...)`, exactly as `/api/backtest` would.
  2. **Run every configuration** through `run_strategy` (§5.0) with **its own** t₀, passing that same `split_date` as the requested split. Without it, each configuration would compute its own 70% split and the window would no longer be common.
     - Do **not** force the grid's latest t₀ onto shorter-lookback configurations. They start signaling earlier, so §6.1's "no signal before t₀" check would raise `ValueError`.
  3. **Score on the common window**, [the latest first-evaluated day across the grid, `split_date`). Take each configuration's own daily excess returns on that window and compute the Sharpe there with the §8 rules. The result is `null` where it's undefined, e.g. for a configuration ruined before the window.

  Without a common window, larger-L configurations would be scored over shorter periods.
- **Not enough history.** The common window must be at least 252 days.
  - The longest lookback needs the most history, so the grid can fail where a single backtest succeeds. For example, a trend backtest on about 3 years of data works at L = 252, while its 18-month (L = 378) configuration leaves too short a common window.
  - If a configuration raises `InsufficientDataError` or has no valid score, catch it and return 422 naming the culprit. For example: "Sensitivity grid: L = 378 leaves only {n} in-sample days in the common window (252 needed). Use a longer date range or drop L = 378 from the grid." A bare `InsufficientDataError` from one configuration would not tell the user which L caused it.
- **Response:** the IS Sharpe matrix (rows = L, columns = z_entry, `null` where undefined), `n_configs_tested`, the window, and the request's own cell.
  - **OOS results are never used or returned.** Each configuration is still backtested over the whole calendar, so OOS returns exist internally, but only the common IS window is scored.
- The UI notes that this window can differ slightly from the KPI grid's IS window.

## 10. Monte Carlo (`services/monte_carlo.py`)

### 10.1 Input and estimation
**Sources:**
- `source="strategy"`: the strategy's daily total returns over the evaluation window (`window="full"`, default) or over IS only (`window="is"`).
- `source="ticker"`: the ticker's daily returns over [start, end].

**Validation.** Return 422 with a clear message unless all of these hold:
- n_hist ≥ 252;
- every R > −1;
- for `source="strategy"`: the window has exposure (`pct_days_in_market > 0`). A σ̂ floor can't catch a strategy that never trades, because cash earning the T-bill rate still varies: a cash-only return series measured σ̂ = 7.5e-5 on 2015–2025 ^IRX data;
- σ̂ ≥ 1e-8, a floor on degenerate input that also guarantees the histogram edges below are strictly increasing;
- `block_len` ≤ n_hist / 2.

**Estimation.** ℓ = ln(1 + R). μ̂ = mean(ℓ) and σ̂ = std(ℓ, ddof=1), both per day (Δt = 1).

**`zero_drift=true`** sets μ̂ := 0 for GBM and ℓ := ℓ − mean(ℓ) for the bootstrap. This is a common risk convention: it avoids projecting a noisy estimated drift.

### 10.2 Model 1: GBM
The log increments are μ̂ + σ̂·Z, with **no −½σ² term**.
- μ̂ is already the drift of the log price.
- The Itô correction applies only when you start from the arithmetic mean m = mean(R); then μ_log ≈ m − ½σ².

Build the paths in place, so that only one (T+1, N) array ever exists. Measured peak: 1.0× that array.
```python
P = np.empty((T + 1, N)); P[0] = 0.0
rng.standard_normal(out=P[1:])
P[1:] *= sigma; P[1:] += mu
np.cumsum(P, axis=0, out=P)          # cumulative log return; equity multiple = exp(P)
```

### 10.3 Model 2: circular moving-block bootstrap
Resampling blocks of the historical ℓ keeps the fat tails, volatility clustering and short-range autocorrelation that GBM ignores.
```python
nb = -(-T // block_len)
idx = rng.integers(0, n_hist, size=(nb, 1, N)) + np.arange(block_len)[None, :, None]  # int64
idx = idx.reshape(nb * block_len, N)[:T]       # time-major (T, N), C-contiguous
P = np.empty((T + 1, N)); P[0] = 0.0
np.take(ell, idx, out=P[1:], mode="wrap")      # "wrap" = circular, and no output buffering
del idx
np.cumsum(P, axis=0, out=P)
```
- Measured peak memory is about 2× the path array.
- Two details keep it there:
  - `np.take` silently copies int32 indices to intp, so build the indices as int64 to begin with.
  - With the default `mode="raise"`, `np.take` buffers the whole output.
  Without both fixes the measured peak was 3.6×.
- Expose the increment generators separately from path aggregation, so tests can inspect the raw increments.

### 10.4 Outputs (for each model)
Draw from one `rng` in a fixed order: GBM increments, GBM sample-path choice, bootstrap increments, bootstrap sample-path choice. For each model, in this order:
1. `terminal = np.expm1(P[-1])`, as a copy.
2. **20 sample paths:** `cols = rng.choice(N, 20, replace=False)`, then `np.exp(P[:, cols])`.
3. **Percentile bands** p5/p25/p50/p75/p95 of the equity multiple for t = 0..T:
   ```python
   np.exp(np.quantile(P, [.05, .25, .5, .75, .95], axis=1, overwrite_input=True))
   ```
   - Quantiles commute with exp. The measured difference from computing them on simple returns is about 1e-9.
   - `overwrite_input=True` avoids a full copy (measured: 1 MB instead of 200 MB), but it leaves P unusable afterwards. Delete P before generating the next model.
4. **Empirical risk**, as positive loss fractions, for α ∈ {0.95, 0.99}. Use the default (linear) `np.quantile` method.
   - VaR_α = −Q_{1−α}(terminal)
   - CVaR_α = −mean(terminal | terminal ≤ Q_{1−α})

**After both models have run:** build the **histogram**. Its 50 bins share edges across both models, spanning the 0.1% to 99.9% quantiles of the pooled terminal returns. Values outside that range go into the first or last bin.
```python
lo, hi = np.quantile(np.concatenate([terminal_gbm, terminal_boot]), [0.001, 0.999])
edges = np.linspace(lo, hi, 51)
assert np.all(np.diff(edges) > 0)
counts = np.histogram(np.clip(terminal, edges[0], edges[-1]), bins=edges)[0]   # per model
```
The `np.clip` is required. `np.histogram` silently drops values outside the edges: here it lost 0.2% of paths, which are exactly the tail the chart exists to show (verified). The counts for each model must sum to N.

### 10.5 Closed-form GBM benchmark
The terminal log return is X ~ N(m, s²), with m = μ̂T and s = σ̂√T. For tail probability p = 1 − α, let q = m + s·Φ⁻¹(p). Then:
- VaR_α = 1 − e^q
- CVaR_α = 1 − e^{m + s²/2}·Φ((q − m − s²)/s) / p

Return both α levels next to the simulated values. During review, simulation agreed with these formulas: over 150 seeds × 2 levels, the deviations behaved like standard-normal noise when scaled by the §12 standard errors (measured).

### 10.6 Seed, timing and memory
- `seed` is optional. If it is absent, use `secrets.randbelow(2**53)` and **return the seed that was used**.
  - Seeds must stay below 2^53, because JavaScript loses integer precision above that.
  - `np.random.SeedSequence().entropy` is 128 bits (verified), so it can't be used directly.
- Return `elapsed_ms` for each model and for the total, measured with `time.perf_counter()`.
- **API caps** (n_paths ≤ 50,000, and n_paths × (horizon + 1) ≤ 12,700,000) keep peak memory around 200 MB. For comparison, the bootstrap at 100k × 252 measured 410 MB.

## 11. API Contract (`main.py`, `schemas.py`)

### 11.1 Endpoints
`GET /api/health`, `POST /api/backtest`, `POST /api/sensitivity`, `POST /api/monte-carlo`.
- **Handlers are plain `def`, not `async def`.** The work is CPU-bound and yfinance blocks. FastAPI runs `def` handlers in a thread pool, whereas blocking inside `async def` stalls the event loop.
- CORS origins come from `CORS_ORIGINS` (default `http://localhost:5173`). The Vite dev server also proxies `/api`.
- **Errors** return `{"detail": "..."}` with these status codes:
  - 422: validation failure or insufficient data;
  - 404: no data for a ticker;
  - 502 or 503: upstream data failure (§4.3).

### 11.2 Requests (Pydantic v2; validate everything listed)
**`BacktestRequest`**

| Field | Type | Default | Rule |
|---|---|---|---|
| `strategy` | `"mean_reversion"` \| `"pairs"` \| `"trend"` | required | |
| `tickers` | list[str] | required | 1 for mean_reversion and trend, 2 distinct for pairs; uppercased before validation; `^[A-Z0-9.^=-]{1,15}$` |
| `start`, `end` | date | required | start ≥ 1990-01-01; end clamped to today_ny() − 1 day (echoed back in `resolved`); start < end must still hold after clamping |
| `L` | int \| null | null → 60 (mean_reversion, pairs) or 252 (trend) | 10–252 for mean_reversion and pairs; 21–504 for trend. The default is resolved after the strategy is known (a Pydantic model validator) and echoed in `resolved` |
| `z_entry` | float | 2.0 | 0 < z_entry ≤ 5; mean_reversion and pairs only |
| `z_exit` | float | 0.5 | 0 ≤ z_exit < z_entry; mean_reversion and pairs only |
| `signal_frequency` | `"monthly"` \| `"daily"` | `"monthly"` | trend only |
| `cost_bps` | float | 10 | 0–100; one-way, per leg |
| `execution_lag` | int | 1 | 0–5 |
| `hedge_mode` | `"fixed_at_entry"` \| `"rolling"` | `"fixed_at_entry"` | pairs only |
| `vol_target` | float \| null | null (overlay off) | 0.01–0.50 (annualized) |
| `vol_halflife` | int | 42 | 5–126; only when `vol_target` is set |
| `vol_refresh` | `"daily"` \| `"weekly"` \| `"monthly"` | `"weekly"` | only when `vol_target` is set |
| `max_leverage` | float | 2.0 | 1–4; only when `vol_target` is set |
| `split_date` | date \| null | null | must leave ≥ 252 IS and ≥ 126 OOS evaluated days |
| `risk_free` | `"tbill"` \| float | `"tbill"` | float in [−0.01, 0.25] |
| `initial_capital` | float | 100000 | > 0 |

**Fields that don't apply** to the chosen mode are **ignored, never rejected**, and are echoed as `null` in `resolved`. That covers:
- `z_entry` and `z_exit` for trend;
- `signal_frequency` for mean_reversion and pairs;
- `hedge_mode` for mean_reversion and trend;
- `vol_halflife`, `vol_refresh` and `max_leverage` when `vol_target` is null;
- in Monte Carlo requests, `window` and `backtest` when source = ticker, and `ticker`/`start`/`end` when source = strategy.

Fields the table marks "required iff" still return 422 when missing.

**`SensitivityRequest`** is `BacktestRequest` plus two optional grid overrides:
- `L_values`: at most 8 values, each within the strategy's L range.
- `z_entry_values`: at most 8 values, each in (z_exit, 5]. Ignored for trend.

**`MonteCarloRequest`**

| Field | Type | Default | Rule |
|---|---|---|---|
| `source` | `"strategy"` \| `"ticker"` | required | |
| `backtest` | BacktestRequest | — | required iff source = strategy |
| `window` | `"full"` \| `"is"` | `"full"` | strategy only |
| `ticker`, `start`, `end` | str, date, date | — | required iff source = ticker; validated like `BacktestRequest` |
| `n_paths` | int | 10000 | 1,000–50,000 |
| `horizon` | int | 252 | 5–504; and n_paths × (horizon + 1) ≤ 12,700,000 |
| `block_len` | int | 10 | 1–63; ≤ n_hist / 2 |
| `seed` | int \| null | null | 0 ≤ seed < 2^53 |
| `zero_drift` | bool | false | |

### 11.3 Responses
These use TypeScript notation, and `src/types.ts` mirrors them exactly. "Resolved" means the request with every default filled in and `end` clamped.

**Why the series starts at s0 = min(t0, e − 1).** The baseline day is e − 1 = t0 + k − 1. When k ≥ 2 that falls after t0, so a series starting at the baseline would leave out the first k − 1 valid scores. A trade signaled on those days would then have a `signal_date` before `dates[0]`, and the signal panel couldn't plot its marker. Starting at s0 keeps every signal date inside the series.
```ts
type Metrics = {
  n_days: number; total_return: number; cagr: number | null; ann_vol: number | null;   // ann_vol null when n_days < 2
  sharpe: number | null; sharpe_ci95: [number, number] | null; sortino: number | null;
  max_drawdown: number; beta_spy: number | null; corr_spy: number | null;
  pct_days_in_market: number; n_closed_trades: number | null; n_open_trades: number | null;
  win_rate: number | null; avg_holding_days: number | null;
  ann_turnover: number | null; ann_cost_drag: number | null;   // trade/cost fields are null for benchmarks
  avg_gross_exposure: number;                                   // 1 for benchmarks
};
type Trade = {
  id: number; side: "long" | "short"; segment: "is" | "oos"; open: boolean;
  signal_date: string; entry_date: string; exit_signal_date: string | null; exit_date: string | null;
  holding_days: number; pnl: number; beta_at_entry: number | null;
};
type BacktestResponse = {
  run_id: number; engine_version: string;
  resolved: ResolvedBacktestRequest & { split_date: string; first_evaluated_date: string; rf_source: string };
  series: {                        // aligned arrays from s0 = min(t0, e - 1) to the end (or the ruin day)
    dates: string[];
    equity: (number | null)[]; benchmark_equity: (number | null)[]; drawdown: (number | null)[];
                                              // null before the baseline day e - 1; E0 / E0 / 0 at the baseline
    daily_return: (number | null)[];          // daily TOTAL return (matches equity); null up to and including the baseline
    position: number[];                       // executed position pos
    gross_exposure: number[];                 // executed sum of |w| at the end of each day (shows leverage over time)
    score: (number | null)[];                 // z (mean_reversion, pairs) or sampled trailing excess return m (trend)
    beta: (number | null)[] | null; beta_used: (number | null)[] | null;   // pairs only
    vol_forecast: (number | null)[] | null;   // annualized sigma (§5.6); null when the overlay is off
    is_oos: boolean[];
  };
  sizing: { vol_target: number; max_leverage: number;
            pct_days_at_cap: number | null } | null;   // whole object null when the overlay is off; pct null if never invested
  trades: Trade[];
  metrics: { is: Metrics; oos: Metrics | null; full: Metrics };            // oos is null only if ruin happened in IS
  benchmark_metrics: { is: Metrics; oos: Metrics | null; full: Metrics };  // same window as the strategy
  diagnostics: { coint_p_is: number | null; coint_p_oos: number | null; half_life_is: number | null;
                 static_beta_is: number | null; static_alpha_is: number | null } | null;    // pairs only (null for mean_reversion and trend)
  ruined: boolean; ruin_date: string | null; warnings: string[];
};
type SensitivityResponse = {
  L_values: number[]; z_entry_values: number[] | null;          // null for trend
  sharpe: (number | null)[][];                                  // rows = L; one column for trend
  n_configs_tested: number; window: { start: string; end: string };
  selected: { L: number; z_entry: number | null } | null;       // the request's own cell, if it is in the grid
};
type ModelResult = {
  bands: { p5: number[]; p25: number[]; p50: number[]; p75: number[]; p95: number[] };   // length T+1
  sample_paths: number[][];        // 20 x (T+1) equity multiples
  hist_counts: number[];           // 50
  var95: number; var99: number; cvar95: number; cvar99: number; elapsed_ms: number;
};
type MonteCarloResponse = {
  resolved: ResolvedMonteCarloRequest & { seed: number; n_hist: number; mu_daily: number; sigma_daily: number };
  gbm: ModelResult; bootstrap: ModelResult;
  closed_form_gbm: { var95: number; var99: number; cvar95: number; cvar99: number };
  hist_edges: number[];            // 51, shared by both models
  elapsed_ms_total: number;
};
```

## 12. Tests (pytest; no network; full suite under 60 s)

### `conftest.py`
- **Fake network (autouse).** A fixture replaces `fetch_from_yahoo` with a fake. The fake serves synthetic frames, counts its calls, and raises for any ticker it doesn't know. No test can reach the network.
- **Frozen clock (autouse).** A fixture freezes `today_ny()` at a date after the fixture data ends.
- **Synthetic data.** At least 1,000 business days (`pd.bdate_range`), generated from a fixed seed, in the exact yfinance column layout. It includes:
  - a cointegrated pair (Y, X);
  - SPY;
  - `^IRX`, held constant at 2.0 (i.e. 2%).
- **Why 1,000 days.** The API rejects shorter series.
  - With L = 120 and k = 1, a 1,000-day series leaves 880 evaluated days: 616 IS and 264 OOS.
  - Trend at its maximum L = 504, with monthly sampling, still leaves about 474 evaluated days: about 331 IS and 143 OOS.
  - Trend's default sensitivity grid (largest L = 378) keeps a common IS window longer than 252 days.
- Short hand-built series are fine for unit tests that call pure functions directly.
- **Database.** Each test gets a temp-file SQLite database (`tmp_path`), wired in with a dependency override.

### `test_strategies.py`
1. **Entry/exit rule against a reference loop.** The vectorized rule must match a reference state machine (written as a loop in the test) bit for bit. Test at least 1,000 random z sequences, including NaNs and values exactly at the thresholds.
2. **Jump through the band.** With z_entry = 2 and z_exit = 0.5, z = [0, 2.5, 1.5, 0.7, −0.7, −1.2, −1.6, −1.0, −0.3] gives positions [0, −1, −1, −1, 0, 0, 0, 0, 0].
3. **Pairs z-score against polyfit.** The pairs z and β match an explicit `np.polyfit` on at least 20 sampled windows, within 1e-8. Normalize with `np.std(residuals, ddof=1)`; measured error is about 1e-11.
4. **Degenerate inputs:**
   - exactly collinear log prices (Y = X²): every z is NaN and the signal is flat;
   - a flat-X window: z is NaN there, and no inf appears anywhere;
   - Strategy A on a flat price: z is NaN and the signal is flat.
5. **Hedge ratio and flat weights.** With `fixed_at_entry`, `beta_used` changes only on days that open a position. Weights are exactly 0.0 whenever flat, including during warm-up.
6. **Trend score:**
   - m matches a hand computation of ln(C_t/C_{t−L}) − Σ rf;
   - m is NaN for the first L rows, with the signal flat there;
   - a trailing excess return of exactly 0 gives a flat signal. Build that with rf = 0 and C_t = C_{t−L};
   - it runs with both a constant-rate `rf` Series and a varying, T-bill-style one. Passing `rf` as a NumPy array raises `TypeError`.
7. **Monthly sampling** (this test catches the `PeriodIndex.shift` trap):
   - on daily data spanning several month boundaries, the monthly trend signal changes only on first trading days of months;
   - the first position appears on the first month start on or after a full lookback;
   - `first_of_period` marks exactly one day per calendar month.
8. **Vol targeting:**
   - `sigma` equals an explicit loop over Σ(1−α)^i·u²_{t−i} / Σ(1−α)^i within 1e-12 relative, and is NaN before `vol_halflife` observations (measured error 4e-16).
   - **Vol tracking.** Use synthetic returns whose true volatility alternates between 10% and 40% every 500 days, with a constant long signal, a 15% target, H = 42 and cap 4. In every regime, measured after skipping 3 half-lives, realized vol / target lies in [0.8, 1.2]. Measured over 200 seeds × 6 regimes: 0.86–1.08, never outside. Unscaled realized vol in the same regimes differs by about 4×.
   - All-zero returns give `scale` = `max_leverage`, never inf or NaN.
   - With `vol_refresh="weekly"`, `scale_used` changes only on first trading days of weeks or on signal changes (measured: zero violations).
   - A `vol_halflife` longer than the strategy's lookback delays t₀ to the forecast's first valid day. Check this for all three strategies through `run_strategy` (§5.0), including pairs, whose forecast depends on the strategy's own rolling β.
   - Weights are exactly 0.0 whenever flat.

### `test_backtester.py`
1. **Look-ahead perturbation (the most important test).** Cover every strategy, both hedge modes, both trend signal frequencies, and vol targeting off and on with each refresh schedule. In each case:
   - multiply all prices strictly after date k by random lognormal noise;
   - score, signal, weights, `vol_forecast`, `pos`, excess and total for every date ≤ k must be bit-identical (`np.array_equal(a, b, equal_nan=True)`). The review prototype measured zero differences across all trend and overlay combinations;
   - use at least 4 values of k;
   - assert that something after k did change, so the test can't pass vacuously.
2. **Hand-computed toy cases** (verified during review). Setup: c = 10 bps, R = [0, .01, .02, −.03, .04, .05] on days 0–5, and `rf = pd.Series(0.0, index=dates)` (a Series, §2.3). Pass `t0 = 1` and `split_date` = day 4 explicitly. The backtester must accept this 6-day series, because the 252/126 minimums live in `resolve_split` (§6.3).
   - k = 0, pos = [0,1,1,0,0,0]: one trade, P&L −0.012, 2 holding days, entry on day 1 and **exit on day 3**. Day 2, the naive answer, is wrong.
   - Direct flip, pos = [0,1,1,−1,−1,0]: two trades, P&L −0.012 and −0.092. The flip day's 2c cost is split c/c.
   - The first case's signal with k = 1: excess = [0, 0, −0.001, −0.03, 0.039, 0] and trade P&L 0.008.
   - **Entry on the final day,** pos = [0,0,0,0,0,1]: one open trade with holding_days 0, P&L −0.001, and no exit date.
   - **Flip on the final day,** pos = [0,1,1,1,1,−1]:
     - trade 1 is closed with exit on day 5, 4 holding days and P&L 0.078;
     - trade 2 is open with 0 holding days and P&L −0.001.
3. **Costs:**
   - net ≤ gross element-wise, with equality on days with zero turnover;
   - total cost = c × total turnover;
   - a pairs entry is charged on both legs.
4. **Cash.** A strategy that never trades has excess exactly 0 every day, total = rf, and `null` Sharpe and Sortino.
5. **Reconciliation.** Over at least 100 random runs, covering both hedge modes, vol targeting on and off (with in-trade resizing), and k ∈ {0, 1, 2}, |Σ pnl − Σ excess| < 1e-9 (measured 2e-16 with the overlay). An open trade is excluded from the win rate.
6. **Leverage financing.** With a constant weight of 2 and cost 0, total = rf + 2(r − rf) exactly on every held day (verified exact).
7. **Evaluation window and series:**
   - the first evaluated day is t₀ + k;
   - the baseline point is present;
   - a loss on the first evaluated day produces a negative drawdown;
   - a trade that straddles the split belongs to IS;
   - with k = 2, a signal on t₀ has its `signal_date` inside `series.dates`, and the equity-type arrays are null before the baseline;
   - compounding `daily_return` from the baseline reproduces `equity`.
8. **Input validation.** Each of these raises `ValueError`:
   - weights and returns with mismatched column labels;
   - an `rf` index that differs from the returns index;
   - a non-zero signal before t0;
   - a `split_date` that leaves zero IS or zero OOS days.

   Also check **`resolve_split`**:
   - with no requested date it returns the date at index ⌊0.7·n_eval⌋;
   - a requested weekend date rolls forward to the next trading day;
   - 251 IS days or 125 OOS days raise `InsufficientDataError`.
9. **Ruin.** A day with total ≤ −1 means:
   - the window ends at r, with that day's total clipped to −1 and E_r = 0;
   - a position open at r is an `open` trade;
   - `ruined` is set;
   - if r falls in IS, OOS metrics are `null`.
   - A 2×-levered long hit by a −50% day must trigger it.
   - **One-day segment.** Ruin on the split date (the first OOS day) leaves an OOS segment with `n_days = 1`. Its `ann_vol`, `sharpe`, `sortino`, `beta_spy` and `corr_spy` are `null`, and the full response still validates against the Pydantic response model. A non-nullable float rejects `None` with a ValidationError (verified). Ruin can't happen on the first evaluated day itself: no position is held into that day (the first position is entered at its close), so its total return is rf minus at most the entry cost.
10. **`pct_days_at_cap`:**
    - it equals the share of invested days whose held gross exposure is ≥ `max_leverage` × (1 − 1e-9);
    - it agrees with the `scale_used.shift(k + 1) ≥ max_leverage` formulation for k ∈ {0, 1, 2};
    - it is `null` for a run with the overlay on that never holds a position, and the whole `sizing` object is `null` with the overlay off.

### `test_sensitivity.py`
These test `pipeline.run_sensitivity` directly.
- **Own t₀ per configuration.** The default A/B grid and the default trend grid run on the 1,000-day fixture with no `ValueError`, even though shorter lookbacks signal before the grid's latest t₀.
- **One split for all.** Every configuration is evaluated with the request's resolved `split_date`, and the reported window equals [latest first-evaluated day, `split_date`).
- **Slice scoring.** Each cell equals the §8 Sharpe of that configuration's daily excess returns on the common window, recomputed independently in the test.
- **Undefined cells.** A configuration with no exposure in the window (e.g. `z_entry_values=[5.0]` on calm data) gives a `null` cell, not an error.
- **Not enough history.** On 756 trading days (about 3 years), the trend backtest at L = 252 succeeds, but the default trend grid raises an error that names L = 378. The review prototype checked 58 start-date alignments: in every one the backtest's split leaves 337–352 IS days and 145–151 OOS days, while the L = 378 common window is only 209–231 days.

### `test_metrics.py`
- **Hand-computed values:**
  - the Sharpe ratio uses the daily rf;
  - the Sortino ratio uses the all-days downside formula;
  - the max drawdown of [100, 120, 90, 130, 65] is −0.5;
  - CAGR on a known path;
  - `sharpe_ci95` on a known input.
- **Null rules:**
  - a constant series gives a `null` Sharpe, not a huge finite number. Test it with a numpy array whose std is tiny but non-zero (252 copies of 1/3: about 6e-17) and with a pandas Series whose std is exactly 0 (252 copies of 0.3);
  - no downside days gives a `null` Sortino;
  - no trades gives a `null` win rate.

### `test_stats.py`
- **Engle–Granger error rates.** Generate AR(1) noise with `scipy.signal.lfilter([1], [1, -phi], eps)`, not a Python loop.
  - Across 100 independent random-walk pairs (n = 750), the share with `coint` p < 0.05 is ≤ 0.12 (measured 0.05–0.06 on two seed sets).
  - Across 100 cointegrated pairs (AR(1) residual, φ = 0.95), the share is ≥ 0.95 (measured 0.99–1.00).
- **Collinear inputs.** `coint` returns p = 0 with a `CollinearityWarning`, and the API reports `null` plus a warning.
- **Half-life:**
  - recovers the true value within 15% on an AR(1) with **n = 20,000** and φ = 0.95. At n = 5,000 the estimate misses 15% for 9% of seeds (measured), which makes the test seed-dependent; at n = 20,000, 0 of 200 seeds missed;
  - is `null` when b ≥ 0.

### `test_monte_carlo.py`
- **Shapes and seeds.** Output shapes are correct. The same seed gives identical output, and different seeds differ.
- **Bands and risk ordering:**
  - bands are monotone (p5 ≤ … ≤ p95) on every day, and all equal 1 at t = 0;
  - VaR99 ≥ VaR95, and CVaR_α ≥ VaR_α.
- **GBM increments.** Run with σ̂ = 0.02, N = 10,000 and T = 252.
  - The mean is within 5 standard errors of μ̂ (SE = σ̂/√(N·T)), and the std is within 5 standard errors of σ̂ (SE ≈ σ̂/√(2·N·T)).
  - With these values, a wrongly added −½σ² term moves the mean by ½σ̂·√(N·T) ≈ 16 SE, so the test reliably catches it. With small σ̂ or N·T it would not.
- **Bootstrap increments:**
  - every increment is a historical value (`np.isin`);
  - blocks are contiguous modulo n. Test this with `ell = np.arange(n, dtype=float)`, so each value reveals its index.
- **Closed form against simulation.** Call the service directly, not the API, with N = 200,000, T = 21 (about 35 MB) and a fixed seed. Let p = 1 − α, Q = the empirical p-quantile of `terminal`, `tail = terminal[terminal <= Q]`, and ES = mean(tail).
  - **VaR** is within 4 × SE_VaR, where SE_VaR = e^q · √(p(1−p)/N) · s / φ(Φ⁻¹(p)) (delta method).
  - **CVaR** is within 4 × SE_CVaR, where SE_CVaR = √((var(tail, ddof=1) + (1−p)·(Q − ES)²) / (N·p)), the full asymptotic standard error.
  - Do **not** use the naive std(tail)/√len(tail) for CVaR. It ignores the error in estimating the quantile and understates the standard error by about 1.4×. With the naive formula, 0.3% of seed/level combinations exceeded 4 SE; with the full formula, none of 300 did (measured over 150 seeds × 2 levels).
- **Zero drift.** With `zero_drift`, the mean terminal log return is ≈ 0.
- **Histogram.** On heavy-tailed bootstrap input (e.g. Student-t with 3 degrees of freedom), each model's `hist_counts` sum to exactly N.
- **Rejections:**
  - σ̂ < 1e-8;
  - any R ≤ −1;
  - `block_len` > n/2;
  - a strategy window with no exposure, including a cash-only series whose σ̂ is well above the floor.

### `test_data_loader.py`
- **Dates.** The requested end date is included (exclusive-`end` handling).
- **Cache hits and refetches:**
  - a second identical request makes zero fetch calls;
  - a ticker whose data starts after `start` is not refetched on repeat requests.
- **Adjustment-basis change.** The fake's second response rescales all history by 0.98. After an extension request, every stored row is on the new basis, with no jump at the old/new seam.
- **Storage:**
  - a bar dated `today_ny()` is never stored;
  - overlapping upserts create no duplicate rows;
  - a 5,000-row upsert works on SQLite.
- **Errors.** An empty frame maps to 404, and a rate-limit error maps to 503.
- **Alignment:**
  - mismatched dates are inner-joined, never forward-filled;
  - every returned date index is `datetime64[ns]`, even though the fake serves `datetime64[s]` like yfinance.
- **Risk-free rate:**
  - rf on day t is the last ^IRX value strictly before t, including across a day the stock calendar has and ^IRX doesn't (like 2016-11-11);
  - a request starting on such a day still gets a rate on its first day, thanks to the 14-day buffer;
  - an out-of-range ^IRX print (e.g. 99.0) is dropped, and the following days use the previous valid rate: no NaN, no 422.

### `test_api.py`
- **Happy paths.** Every endpoint works on fixture data, including a pairs request that never mentions SPY and a request with `risk_free="tbill"`.
- **No NaN in responses.** Parse every response with `json.loads(r.text, parse_constant=...)`, where the hook fails the test. Python's `json` accepts `NaN` by default (verified), so a plain `json.loads` would let NaN through unnoticed.
- **Validation (422 for each):**
  - z_exit ≥ z_entry;
  - too few days for IS/OOS;
  - n_paths × horizon over the cap;
  - seed ≥ 2^53.
- **Persistence.** A `backtest_runs` row is written for `/api/backtest` only.
- **Ignored fields.** Each of these is accepted (no 422) and echoed as `null` in `resolved`:
  - `hedge_mode` sent with a mean_reversion request;
  - `z_entry` sent with a trend request;
  - `vol_refresh` sent while `vol_target` is null;
  - `window` sent with a ticker-source Monte Carlo request.
- **Trend:**
  - a trend request with no `L` resolves it to 252, and `L = 600` returns 422 (the range is 21–504);
  - `sizing` is `null` when the overlay is off;
  - trend sensitivity returns `z_entry_values: null` and a one-column Sharpe matrix;
  - on a 756-trading-day range, `/api/backtest` for trend succeeds, while `/api/sensitivity` returns 422 whose `detail` names L = 378.

## 13. Frontend (`/frontend`)

### Setup
- Scaffold with `npm create vite@latest frontend -- --template react-ts`.
- Add Tailwind CSS **v4** through the `@tailwindcss/vite` plugin, with `@import "tailwindcss";` in the main CSS. v4 has **no** `tailwind.config.js` and no `npx tailwindcss init`, so don't follow v3 instructions.
- Use Recharts 3.
- `src/types.ts` mirrors §11.3 exactly; `src/api.ts` contains typed fetch wrappers.
- The base URL comes from `VITE_API_URL`, which defaults to empty (same origin). In development, requests therefore go through a Vite proxy that forwards `/api` to `http://localhost:8000`.
- The layout is a desktop tool (≥ 1280 px wide) with a dark terminal aesthetic: a sidebar plus a main panel.

### Sidebar
- Strategy (mean reversion, pairs, trend) and tickers.
  - Pair presets:
    - KO/PEP;
    - XOM/CVX;
    - GOOGL/GOOG: near-collinear share classes that exercise the guards;
    - NVDA/AMD: likely not cointegrated, which demonstrates the warnings.
  - Trend presets: SPY, QQQ, GLD and TLT. These are diversified ETFs, which is where trend following is usually applied.
- Dates and an optional split date.
- L, which switches to its per-strategy default when the strategy changes. For trend, offer month presets: 1, 3, 6, 12 and 18 months.
- z_entry and z_exit (mean reversion and pairs), signal frequency (trend), and hedge mode (pairs).
- Cost bps, execution lag, and risk-free rate (T-bill or a constant).
- **Vol targeting:** an on/off toggle, the target (annual %), half-life, refresh schedule, and max leverage.
- Show only the controls that apply to the chosen strategy. Hidden controls are not sent.
- Monte Carlo settings: paths, horizon, block length, seed and zero drift.
- Requests are sent only from a **Run** button.

### Views
1. **KPI grid**, with an IS | OOS | Full toggle:
   - Sharpe with its 95% CI, Sortino, max drawdown, CAGR, win rate, closed/open trades, % days in market, turnover, average gross exposure and beta to SPY;
   - with vol targeting on, the target, realized vol (`ann_vol`) and % of days at the leverage cap;
   - benchmark figures alongside;
   - pairs diagnostics: IS and OOS coint p-values and half-life;
   - VaR95 and CVaR95 from the latest Monte Carlo run;
   - `null` shows as "—".
2. **Equity and drawdown.** Strategy vs benchmark on top and underwater drawdown below, synchronized with `syncId`. The OOS region is shaded and the split date is marked.
   - Every synchronized chart, including the signal panel, uses the same `series` arrays, so their indexes and dates line up.
   - Equity values are null before the baseline and render as a gap. Don't drop those rows: that would shift the indexes.
3. **Signal panel.** The score, with entry/exit markers at **signal dates**.
   - Mean reversion and pairs: z, with ±z_entry and ±z_exit lines.
   - Trend: the trailing excess return as a step line, with a zero line.
   - Pairs: also the rolling β and `beta_used`.
   - Below the score, a small `gross_exposure` chart shows leverage over time, plus the annualized vol forecast when the overlay is on.
4. **Trades table**, sortable, built from `trades`.
5. **Sensitivity heatmap** (a CSS grid):
   - a diverging colour scale centred on a Sharpe of 0;
   - the request's own cell outlined;
   - "N configs tested" and the window shown;
   - for trend, a single row over L.
6. **Monte Carlo:**
   - **Fan chart:** range Areas for p5–p95 and p25–p75 (a `dataKey` that returns `[low, high]`), a p50 line, and 20 thin sample paths, with a GBM / bootstrap toggle.
   - **Terminal-return histogram** on a numeric x-axis, with a red `ReferenceLine` at −VaR95.
   - A table comparing GBM simulated, GBM closed form and bootstrap.
   - The seed and elapsed time.
7. **Warnings banner** listing the API's `warnings`.

### Behaviour and tests
- Charts use `isAnimationActive={false}` and `dot={false}`; there are about 4,000 points per series.
- Show loading, empty and error states. Error states render the API's `detail`.
- An assumptions footer states the cost (one-way bps), the execution lag, the rf source, the daily-rebalancing assumption and, when on, the vol-targeting settings.
- Use Vitest for pure helpers such as formatters and null handling.

## 14. Non-Functional Requirements
- **Determinism.** The same request on the same cached data gives an identical backtest. Monte Carlo output is identical for a given seed.
- **Performance targets.** These are targets, not claims. Measure them with `scripts/bench.py`, which also prints the hardware, Python and library versions.

  | Workload | Target |
  |---|---|
  | 15-year pairs backtest, warm cache | < 300 ms |
  | Default sensitivity grid | < 5 s |
  | Default Monte Carlo (10k × 252, both models) | < 1 s |

  For reference, the review prototype measured about 128 ms for GBM and 106 ms for the bootstrap at this size, including bands.
- **Memory.** Monte Carlo peaks at about 200 MB at the API caps.
- **Tests.** No network, and the full suite finishes in under 60 s.

## 15. Documentation
**`docs/METHODOLOGY.md`** contains:
- **Justifications.** A derivation or justification for every formula and rule in this file:
  - timing; cash and rf; costs; rebalancing;
  - the residual z-score (including the ddof choice); the crossing exit rule; the frozen hedge ratio; the degenerate-window guards;
  - trade attribution; Engle–Granger vs ADF; the Sharpe CI;
  - time-series momentum (Moskowitz, Ooi & Pedersen 2012), the excess-return sign, and monthly vs daily signal evaluation (whipsaw and cost);
  - vol targeting: the EWMA forecast, the half-life choice, the leverage cap, the refresh trade-off, and financing at rf. Cite Moreira & Muir (2017) and Harvey et al. (2018) on why constant-vol sizing can raise risk-adjusted returns;
  - GBM drift; the block bootstrap; closed-form VaR/CVaR;
  - the IS/OOS protocol.
- **Limitations:**
  - survivorship and selection bias (pairs chosen with hindsight);
  - close-price fills, a fixed bps cost, no market impact, and no borrow fees or short-sale constraints;
  - exits use the rolling z while the frozen-β spread is held;
  - multiple testing across the grid (configurations are counted, but there is no deflated Sharpe);
  - iterating parameters while watching OOS results turns OOS into IS;
  - the Sharpe CI assumes iid returns;
  - GBM assumes iid normal log returns;
  - the bootstrap assumes stationarity and can't exceed historical extremes;
  - Monte Carlo ignores parameter uncertainty and inherits any overfitting in the backtest;
  - trend following here trades one asset at a time. The published results come from diversified portfolios of dozens of futures, so single-asset Sharpe ratios are much lower;
  - the vol forecast lags sudden spikes, so realized vol overshoots the target in crashes;
  - vol targeting sizes up when measured vol falls, including during stale-price runs (capped at `max_leverage`);
  - leverage is financed at the T-bill rate with no borrowing spread;
  - Yahoo data quality.
- **Deviations:** any approved departures from this spec, with reasons.

**`README.md`** contains:
- setup and run commands that work in PowerShell;
- how to run the tests;
- screenshots;
- benchmark numbers copied from `scripts/bench.py` output, with the hardware stated.

## 16. Phases and Definition of Done
Stop at the end of every phase (§0, rule 5).

| Phase | Scope | Done when |
|---|---|---|
| 0. Scaffold | `git init`, §3 layout, tooling, CI, `.env.example`, empty app | `ruff check`, `pytest` (collects) and `npm run build` succeed in PowerShell |
| 1. Core math | `pipeline` (`run_strategy`, `run_sensitivity`), `strategies` (A, B, C), `sizing`, `backtester` (with `resolve_split`), `metrics`, `stats`, `monte_carlo` and their test files; synthetic data only | those tests pass, including the perturbation (all strategies, overlay on and off), reconciliation, vol-tracking and closed-form tests |
| 2. Data + API | models, cache, loader, endpoints, serialization, `test_data_loader`, `test_api` | the full suite passes, and one real KO/PEP backtest via `/docs` succeeds (network allowed in this check only) |
| 3. Frontend | §13 | lint, Vitest and build pass; a manual run against the local API; screenshots |
| 4. Docs + bench | §15, `scripts/bench.py` | every formula in this file is justified in METHODOLOGY.md; README numbers come from bench output |

**Stretch goals** (only if asked):
- cross-sectional momentum across a stock universe (needs a survivorship-bias-aware universe);
- a multi-asset time-series momentum portfolio;
- Kalman-filter hedge ratio;
- walk-forward re-optimization;
- deflated Sharpe ratio;
- stationary bootstrap;
- stop-loss and time-stop rules;
- GARCH Monte Carlo.
