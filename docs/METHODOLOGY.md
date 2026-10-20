# Methodology

Why the engine computes what it computes, what it can't tell you, and every approved departure from the specification, [CLAUDE.md](../CLAUDE.md) (§15). Section signs (§) refer to CLAUDE.md. This covers everything built through Phase 8; Phase 9 adds its own sections when it lands.

**Measured figures.** A figure marked *(engine)* was measured on 2026-10-16 (Phases 0–4), 2026-10-17 (Phases 5 and 6) or 2026-10-19 (Phase 7) with this code, through the same loader and pipeline as `/api/backtest`, on Yahoo Finance data cached that day. Each comes with the request that reproduces it; fields not shown are the defaults (10 bps one-way, k = 1, ^IRX). Timings come only from `scripts/bench.py` (see the README).

## Justifications

### 1. Conventions (§2.1, §2.4)
- **Decimals throughout.** Returns, rates, weights and risk figures are decimals in the backend, and only the frontend formats percentages. Basis points appear only in the `cost_bps` input. One unit system leaves no room for factor-of-100 bugs between modules.
- **252 trading days a year.** Means annualize by ×252 and standard deviations by ×√252. It is the convention for US equities; the actual count varies from 250 to 253.
- **Simple returns** are C_t / C_{t−1} − 1 on adjusted closes, computed directly. `pct_change` changed its fill behaviour across pandas versions, and a silent fill would turn a missing day into a zero return. Log returns ℓ = ln(1 + r) are used where adding over time matters: the trend score and Monte Carlo.
- **Adjusted closes include dividends,** so a long position earns them and a short pays them, with no separate dividend accounting.
- **Trailing windows** include day t and need all L observations (`min_periods=L`). Every statistic dated t then uses data through the close of t and nothing later. **ddof = 1** (the pandas default) everywhere, so rolling, static and metric statistics agree.
- **Same-row tolerances.** A guard compares values from the same date, such as a window's variance with that window's own variance. A threshold built from a full-sample statistic would leak future data into earlier days, and the perturbation test (§12) would fail.
- **Undefined means `null`.** A ratio with a numerically zero denominator is reported as `null`, never NaN, inf or a huge number. Rounding leaves the standard deviation of a constant series tiny but not always zero (1.4e-17 for 252 copies of 0.1, §2.4), which would give a naive Sharpe ratio of about 7e15. Each statistic has its own threshold (§16 below).
- **JSON.** A plain JSON response containing NaN or inf makes Starlette answer HTTP 500, while Pydantic quietly writes them as `null`, even in fields that may not be null. `to_jsonable()` converts non-finite floats to `None` before the response model validates, so a nullable field gets `null` and a non-nullable one fails loudly.
- **One random generator per run,** `np.random.default_rng(seed)`, so Monte Carlo output depends only on the seed, never on global state.

### 2. Timing and look-ahead (§2.2)
A signal s_t may use data through the close of day t. The executed position is p_t = s_{t−k}, where k is the execution lag, and day t's return is earned by the position held into it: p_{t−1}·r_t.
- **k = 1 (default) fills at the next day's close.** The close that produced the signal is only known once the market has closed, so trading at that close needs an order sent before seeing the price. k = 0 is offered as the optimistic bound.
- **The lag is applied once, in the backtester.** Strategies return unshifted signals. A second shift would silently add a day of lag; a missing one would be look-ahead.
- **The last k signals are never executed,** because they would fill after the data ends.
- **The perturbation test checks this** (§12). Multiplying every price after a date by random noise must leave every score, signal, weight, position and return up to that date bit-identical. It covers every strategy, both hedge modes, both trend frequencies, the vol overlay off and on with each refresh schedule, the long-only switch, a borrow fee, the walk-forward's stitched series (§26 below), the Kalman pairs model and the trend portfolio, including each leg's score, signal, vol forecast and executed position (§27–28 below).

### 3. Data (§4)
- **Only final bars are cached:** those dated before `today_ny()`. During market hours Yahoo serves a partial bar for today, and caching it would freeze an intraday price as a close.
- **A cache miss refetches the whole range.** Yahoo back-adjusts: every dividend rescales all earlier adjusted closes, and every split rescales all earlier closes. §4.2 gives an example: KO's 2024-01-02 adjusted close, read in 2026, is 55.31 against a raw close of 59.82. Rows fetched on different dates sit on different bases, so appending new rows to old ones creates a fake return at the seam. A miss therefore downloads the union of the old and the requested range in one call and overwrites, keeping each ticker on one basis. Rows and coverage are written in one transaction, so coverage never claims rows that aren't there.
- **Coverage records the requested range,** not the dates Yahoo returned. Otherwise a ticker that listed after `start` would never count as covered and would be refetched on every request.
- **Upserts go in chunks of 1,000 rows,** because SQLite allows at most 32,766 bound parameters in one statement.
- **One network seam.** `fetch_from_yahoo` is the only code that imports yfinance, so the tests replace one function and can't reach the network. Each of its arguments matters in yfinance 1.7: `end` is exclusive, `auto_adjust` defaults to dropping `Adj Close`, and `multi_level_index` defaults to MultiIndex columns. Failures are never retried in a loop, which would only deepen a rate limit.
- **Cleaning (§4.4):**
  - A non-positive price is impossible for a listed stock and breaks log returns, so the row is dropped and reported.
  - A daily move above 50% may be real (a failed drug trial, GME in January 2021) or an unadjusted split. It is reported but kept: dropping a real crash would flatter every strategy.
  - Five or more identical adjusted closes suggest stale data or a halt. They are reported because the vol forecast shrinks during a stale run and sizes positions up toward the cap.
  - ^IRX is a rate, so the positivity check doesn't apply. A print outside [−1, 25] percent is treated as bad; the range allows the slightly negative yields of 2015–2025. Bad rows are dropped rather than kept as NaN: `merge_asof` returns a NaN row instead of skipping it, and one NaN would make every backtest spanning that date fail.
- **Alignment (§4.5).**
  - The run calendar is the intersection of every leg's dates and SPY's.
  - Returns are computed after the join, so a return correctly spans a date that one leg lacks, instead of inventing a flat day.
  - Prices are never forward-filled. A stale price is a fake zero return, which lowers measured volatility and inflates Sharpe ratios.
  - Every date index is normalized to `datetime64[ns]`. yfinance returns `[s]`, and pandas 3 refuses some operations across resolutions.

### 4. Cash and the risk-free rate (§2.3, §4.5)
- **The rate.** rf_t = IRX_{t−1} / 100 / 252: the 13-week T-bill yield (quoted in percent) observed at the previous close, accrued simply over day t. It is the rate known when the day starts, so using it involves no look-ahead.
- **"The previous close" is an as-of lookup, strictly before t.** ^IRX occasionally has no row on a stock trading day (once in 2015–2025, §4.5), and bad prints are dropped, so the rate for day t is the last valid print dated before t. `merge_asof(..., allow_exact_matches=False)` does exactly that.
  - Forward-filling a rate is harmless: a rate stays in force until the next print. Prices are never forward-filled.
  - ^IRX is loaded from start − 14 days, so the first trading day always has an earlier print. Nothing is back-filled, and a remaining gap is a 422.
- **Cash accounting.** With weights w_i as fractions of equity, the account holds 1 − Σw_i in cash, which earns rf; short sales add their proceeds to cash. Before costs, the day's total return is

  total_t = Σ_i w_{i,t−1} r_{i,t} + (1 − Σ_i w_{i,t−1}) rf_t = rf_t + Σ_i w_{i,t−1} (r_{i,t} − rf_t).

  Three consequences:
  - A flat day earns exactly rf, and its excess return is exactly 0.
  - Leverage needs no special case. With Σw > 1 the cash weight is negative, which is borrowing at rf (exact for w = 2, §12).
  - A pair whose legs offset in dollars earns rf on the whole account plus the spread's excess return.

  Before Phase 5 there is no borrow fee and no financing spread, which makes shorting and leverage slightly too cheap (see Limitations).

### 5. Strategy call order (§5.0)
`run_strategy` is the only code that runs the strategy steps, in one fixed order:
1. raw score;
2. vol forecast and mask;
3. signal;
4. unit weights;
5. sizing;
6. t₀, the split and the backtester.

"Signal first, size later" doesn't work, for two reasons:
- The vol mask has to hide the score before the entry rule runs. Otherwise a position could open during the forecast's warm-up, with no forecast to size it.
- For pairs, the forecast needs the rolling β that the strategy computes in step 1.

The API, the sensitivity grid and the tests all call this one entry point, so they can't drift apart.

### 6. The entry/exit rule (§5.1)
- **Hysteresis.** A position opens when |z| > z_entry and closes when z crosses back past ±z_exit, with 0 ≤ z_exit < z_entry. With a single threshold, positions would open and close every time z hovered around it, paying costs on noise.
- **The crossing exit.** A short closes once z < z_exit, and a long once z > −z_exit. The symmetric band rule ("flat when |z| < z_exit") fails when z jumps across the band in one bar.
  - In the §12 example, z goes from 0.7 to −0.7 in one bar.
  - The band rule keeps the short open while the spread sits below its mean. That position loses if the spread reverts, which is exactly the move the strategy bets on.
  - The crossing rule closes it on that bar.
- **Implementation.** Two independent hysteresis legs, each a forward-filled on/off state.
  - They can't both be on: the short leg turns on only above z_entry, where the long leg is necessarily off (z > −z_exit), and vice versa.
  - A direct flip in one bar is allowed.
  - The tests check the vectorized rule bit for bit against a reference state machine, on 1,200 random sequences that include NaNs, values exactly at the thresholds and direct flips (§12).
- **Strict comparisons, and NaN means flat.** A z exactly at a threshold triggers nothing. A NaN z means the window is warming up or degenerate: there is no signal, so any position closes.

### 7. Strategy A: single-asset mean reversion (§5.2)
z_t = (ln P_t − mean) / std over the trailing L days of the log price: a Bollinger-band z-score.
- **Log prices make z scale-free.** A 1% move counts the same at $10 and at $1,000.
- **Rebasing.** The log price is rebased to its first value, ln(P/P₀). z is unchanged, because it doesn't depend on the level, but the rolling sums stay small and accurate.
- **The guard.** A window of identical prices has a variance of exactly 0.0 (§5.2). The guard `v > 1e-12` turns it into a NaN z, and so a flat signal, instead of a division by zero.

### 8. Strategy B: pairs trading (§5.3)
- **The model.** Over each trailing window, regress y = ln(Y/Y₀) on x = ln(X/X₀) by OLS: β_t = cov(x, y) / var(x). Today's residual is e_t = (y_t − ȳ) − β_t(x_t − x̄).
- **The z-score.** With the window's sums of squares S_xx, S_yy and S_xy, and β = S_xy / S_xx, the residual sum of squares is

  SSR = Σ[(y − ȳ) − β(x − x̄)]² = S_yy − 2β S_xy + β² S_xx = S_yy − β² S_xx.

  Dividing by L − 1 gives SSR / (L − 1) = v_y − β² v_x, in the rolling ddof = 1 variances.
  - So z_t = e_t / √(v_y − β² v_x) is today's residual in units of the window's residual standard deviation.
  - It needs only rolling moments, with no loop over windows.
  - It agrees with an explicit `np.polyfit` on sampled windows within 1e-8 (§12).
- **Why divide by L − 1, not L − 2.** The textbook OLS standard error divides by L − 2 for the two fitted parameters. L − 1 is consistent with every other variance in the engine. The difference is a factor √((L − 1)/(L − 2)): 0.86% at L = 60, well below any meaningful change in z_entry.
- **Why not a rolling z of y − β_t·x.** That series mixes spreads built with a different hedge ratio each day. Its rolling mean and standard deviation describe no portfolio that was ever held.
- **The degeneracy guard.** (v_y − β² v_x) / v_y is 1 − R² of the window's regression. So the guard `resid_var > 1e-8 * vy` treats a window as exactly collinear when R² > 1 − 10⁻⁸.
  - Being relative, it doesn't depend on the price level.
  - For KO/PEP (2015–2025, L = 60) the smallest 1 − R² over all windows is 0.036 *(engine)*, more than six orders of magnitude above the guard.
  - Exactly collinear inputs leave rounding noise far below it: the §12 test with Y = X² gets an all-NaN z.
  - Slightly negative values from rounding are not clipped to zero. That would produce ±inf z-scores, which open positions.
- **The hedge ratio is held fixed (`fixed_at_entry`, default).** β is captured on the day a position opens and held until it closes, so the trade's P&L is that of one well-defined portfolio. Re-hedging daily (`rolling`) turns every wiggle in the rolling β into turnover and cost.
- **Exits use the rolling z,** even though the frozen-β spread is what's held. An exit rule on the frozen-β spread would depend on each trade's entry date and couldn't be vectorized. The two spreads differ by (β_t − β_entry)(x_t − x̄), which is small while β is stable (see Limitations).
- **Unit weights.** w_Y = s / (1 + |β|) and w_X = −sβ / (1 + |β|), so the gross exposure |w_Y| + |w_X| is 1.
  - A long spread (s = +1) holds, per unit, $1 of Y long and $β of X short, because the spread's daily change is Δy − βΔx ≈ r_Y − β r_X.
  - Weights are set to exactly 0 when flat, because in pandas a NaN β times a zero signal is still NaN.

### 9. Pairs diagnostics (§5.4)
- **Engle–Granger, not an ADF test on the residuals.** OLS chooses β to minimize the residual variance, so the fitted residual looks more stationary than the true spread. Dickey–Fuller critical values assume β is known and are too lenient. `statsmodels.tsa.stattools.coint` uses MacKinnon's critical values for an estimated cointegrating vector.
  - Simulated *(engine)* on 1,000 pairs of independent random walks (750 days each), at the 5% level:
    - `coint` called 5.4% of them cointegrated;
    - plain `adfuller` on the OLS residuals called 16.0% cointegrated.
  - This reproduces it, run from `backend\` with the venv's Python:

  ```python
  import numpy as np
  from statsmodels.tsa.stattools import adfuller

  from app.services import stats

  g = np.random.default_rng(2026)
  eg, adf = [], []
  for _ in range(1000):
      y, x = np.cumsum(g.normal(0, 0.01, (2, 750)), axis=1)  # two independent random walks
      eg.append(stats.engle_granger(y, x).p_value < 0.05)
      beta, alpha = np.polyfit(x, y, 1)
      adf.append(adfuller(y - alpha - beta * x, autolag="AIC", result_object=False)[1] < 0.05)
  print(np.mean(eg), np.mean(adf))  # 0.054 0.16
  ```
- **The IS window starts at the first loaded date,** warm-up included. Those days are genuine in-sample data, and the test needs every observation it can get. The OOS p-value is a stability check: it is displayed and never used for a decision.
- **Collinear inputs.** On exactly collinear legs `coint` returns p = 0 ("perfectly cointegrated") with a `CollinearityWarning`. The engine reports `null` plus a warning instead. A constant leg also gives `null`, because the test is undefined.
- **Half-life.** Fit Δs_t = a + b·s_{t−1} on the static residual.
  - Then s_t − μ = (1 + b)(s_{t−1} − μ) + ε_t, with μ = −a/b. A deviation shrinks by the factor 1 + b each day.
  - It halves after h days, where (1 + b)^h = ½, so h = −ln 2 / ln(1 + b).
  - That needs 0 < 1 + b < 1, i.e. −1 < b < 0. With b ≥ 0 there is no mean reversion. With b ≤ −1 the residual overshoots and oscillates, and has no half-life.

### 10. Strategy C: time-series momentum (§5.5)
- **The rule.** Moskowitz, Ooi & Pedersen (2012) found that an asset's own trailing 12-month excess return predicts its next month's return, across 58 futures and forward markets. Strategy C goes long when the trailing L-day excess log return

  m_t = ln(C_t / C_{t−L}) − Σ_{u=t−L+1..t} rf_u

  is positive, and short when it is negative.
- **Excess, not raw.** A long position earns r − rf above cash. When rates are high, an asset can rise by less than cash and still show a positive raw return. Subtracting the summed daily rf compares the asset with the alternative actually on offer. Log returns make the L-day return a sum, so the two terms subtract cleanly.
- **Monthly evaluation (default).** The paper rebalances monthly. Checked daily, the sign of a 12-month return flips on noise whenever the return is near zero, and every flip is two one-way trades. Measured *(engine)* on SPY from 2006-01-03 to 2025-12-31, overlay off:

  | `signal_frequency` | Position changes a year | Turnover a year | Cost drag a year | Sharpe (full) |
  |---|---:|---:|---:|---:|
  | daily | 4.75 | 9.55× | 0.95% | 0.23 |
  | monthly | 0.95 | 1.96× | 0.20% | 0.32 |

  - Request: `{"strategy": "trend", "tickers": ["SPY"], "start": "2006-01-03", "end": "2025-12-31", "signal_frequency": "daily"}`, and the same with `"monthly"`.
  - "Position changes" counts the evaluated days whose executed position differs from the day before, per 252 days.
  - These agree with the review prototype's figures in §5.5 to within rounding.
- **The first-of-month helper compares each row's month with the previous row's** (`Series.shift`). `PeriodIndex.shift` moves every period one month forward instead, so every day would compare unequal and count as a "month start", silently turning monthly into daily rebalancing (§5.5). A test checks that the monthly signal changes only on first trading days of months.
- **No hysteresis.** The sign is the position. The score stays NaN until the first month start after a full lookback, so t₀ falls on a month start.

### 11. Volatility targeting (§5.6)
- **What is sized.** u is the daily return of one unit of the strategy's position.
  - For A and C, u is the asset's return.
  - For pairs, u_t = (r_Y − β_{t−1} r_X) / (1 + |β_{t−1}|): one unit of long spread, formed with yesterday's rolling β.
  - The rolling β, unlike `beta_used`, exists while the strategy is flat, so the forecast is ready when a trade opens.
- **The forecast.** σ_t = √(252 · EWMA_t(u²)): a zero-mean, exponentially weighted forecast, as in RiskMetrics.
  - Daily mean returns are tiny next to their standard deviation, so dropping the mean costs almost nothing and avoids a second noisy estimate.
  - With `adjust=True`, pandas computes the normalized weighted average Σ(1 − α)^i u²_{t−i} / Σ(1 − α)^i over the available history, with α = 1 − 2^(−1/H). The weights halve every H days. A test checks this against an explicit loop to 1e-12.
- **A half-life of 42 days** matches Moskowitz, Ooi & Pedersen's 60-day centre of mass, α = 1/(1 + 60). The equivalent half-life is H = ln 2 / ln(61/60) = 41.9 days.
- **Scale and cap.** scale = min(target / σ, max_leverage). A zero forecast gives inf, which the cap turns into max_leverage rather than an error. The cap (default 2) bounds the damage when the forecast is too low, for example during a stale-price run.
- **The warm-up mask.** The score is masked wherever σ is undefined, before the entry rule runs (§5.0, step 2). So t₀ is the first day with both a score and a forecast, and no position ever opens unsized.
- **Refresh.** A signal change always resizes, so every entry uses the current forecast. Between signal changes, the size updates only on refresh days (weekly by default), which trades tracking accuracy for turnover. Measured *(engine)* on SPY trend with the monthly signal and a 10% target:

  | Overlay | Turnover a year | Realized vol (full) | Yearly realized vol, 2008–2025 | Sharpe (full) |
  |---|---:|---:|---:|---:|
  | off | 1.96× | 19.8% | 6.7–41.3% | 0.32 |
  | daily refresh | 2.62× | 10.4% | 6.9–13.7% | 0.45 |
  | weekly (default) | 2.08× | 10.6% | 6.8–14.0% | 0.44 |
  | monthly | 1.74× | 11.3% | 6.6–17.0% | 0.44 |

  - Request: the monthly trend request above plus `"vol_target": 0.10` and `"vol_refresh": "daily"`, `"weekly"` or `"monthly"`.
  - The yearly range runs from the lowest to the highest annualized standard deviation of daily total returns, over the calendar years fully inside the evaluation window.
  - The leverage cap never bound (`pct_days_at_cap` = 0).
  - These agree with the prototype's figures in §5.6 to within 0.1 percentage point.
- **Why it can raise risk-adjusted returns.** Volatility is persistent, so it can be forecast, while expected returns don't rise in proportion to it. Scaling exposure inversely to forecast volatility therefore cuts more risk than return when markets are turbulent.
  - Moreira & Muir (2017) document this for the market and for equity factors.
  - Harvey et al. (2018) find that it raises Sharpe ratios for risk assets such as equities and credit, where returns and volatility are negatively correlated (the leverage effect), with little effect on bonds, currencies and commodities. Across asset classes it makes extreme returns less likely.
  - On SPY trend the overlay roughly halved full-period volatility and raised the Sharpe ratio from 0.32 to 0.44 (table above).
- **Financing.** Leverage above 1 comes from the cash identity (§4 above): the negative cash weight is borrowed at rf, with no spread (see Limitations).
- **`pct_days_at_cap`.** It is measured on the executed exposure held into each day, the same way as `pct_days_in_market`. It uses a 1e-9 relative tolerance, because scaled weights can land one rounding step below the cap. When the cap binds on more than 25% of invested days, the target is effectively unreachable, and the engine warns.

### 12. Backtest accounting (§6.1–§6.2)
- **Inputs must line up by label.** pandas aligns on labels, so weights labelled Y/X times returns labelled KO/PEP give all-zero P&L with no error (§6.1). The backtester requires identical indexes and columns. `t0` has no default, because a forgotten t0 would quietly put the warm-up days back into the metrics.
- **The daily arithmetic.** The executed weights are w = W.shift(k), and the weights held into the day are w_prev = w.shift(1). With c = cost_bps / 10,000, one-way per unit of notional traded, per leg:
  - gross excess = Σ_i w_prev,i (r_i − rf);
  - turnover = Σ_i |w_i − w_prev,i|, counting every leg;
  - cost = c × turnover, charged on the trade day;
  - borrow = (b / 10,000 / 252) × Σ_i max(−w_prev,i, 0), the borrow fee on the short notional held into the day (§22 below);
  - excess = gross excess − borrow − cost, and total = rf + excess.

  A round trip costs 2c × gross exposure, and a pair pays on both legs.
- **Two approximations (§6.1):**
  1. Cost is charged as a fraction of the day's starting equity, which ignores the term cost × same-day return. At 10 bps of cost and a 1% move, that term is 0.1 bp.
  2. Weights are fractions of current equity, held at target every day, and only changes in the target cost money. A real holder of fixed share counts sees the weights drift with prices without trading; charging for that drift would charge for trades the strategy never makes.
- **The evaluation window.** t₀ is the first day with a valid score. The first evaluated day is e = t₀ + k: the first day a position can be executed and a cost incurred. Metrics start at e. The warm-up days earn exactly rf, and including them would add zero-excess days that dilute every statistic.
- **Equity** compounds from a baseline E₀ dated e − 1: E_t = E₀ ∏(1 + total_u). The drawdown's running maximum includes the baseline, so a loss on day e shows as a drawdown.
- **Ruin.** A short can lose more than 100% in a day (GME rose 134.8% on 2021-01-27, §6.2), and a −50% day wipes out a long levered 2×. On the first day with total ≤ −1:
  - the day is clipped to −1 (limited liability, no margin call), and the window ends there;
  - open positions are reported as open trades;
  - clipping raises that day's excess return, and the difference goes to the trade held into the day, so the trades' P&L still sums to the total excess (§7.2);
  - Monte Carlo refuses any window containing that day, because ln(1 + R) is undefined at R = −1.

### 13. In-sample and out-of-sample (§6.3)
- **The split.** By default the split date is the evaluated day at index ⌊0.7·n⌋, computed as `7 * n // 10` in integers (see Deviations, Phase 1, item 2). So 70% of evaluated days are in-sample. A requested date that isn't a trading day rolls forward to the next one.
- **Minimums of 252 IS and 126 OOS days.** By the Lo (2002) standard error (§16 below), an annualized Sharpe ratio from n days has a standard error of about √(252/n). That is 1.0 on a year of IS data and 1.4 on half a year of OOS data. The minimums are floors below which the statistics are noise; meeting them doesn't make the statistics precise.
- **Signals are computed on the full series, then sliced.** Trailing windows are causal, so an OOS day's signal correctly uses pre-split history, as it would have in real time.
- **Each segment is rebased** to E₀ at its own baseline, and its drawdown is measured within it. The OOS figures don't inherit the IS path.
- **What OOS protects.** Parameter choice uses only IS data: the sensitivity grid scores only IS days, and trade statistics never drive a selection.
  - A trade belongs to the segment in which it was entered, even if it closes after the split. Per-segment trade P&L therefore doesn't add up to per-segment returns (§7.2), which is harmless for the same reason.
  - OOS stays clean only if the user doesn't tune parameters on it (see Limitations).

### 14. Benchmarks (§6.4)
- **Mean reversion and trend** are compared with buy-and-hold of the same ticker. The question is whether timing the asset beat owning it.
- **Pairs** are roughly market-neutral and are compared with SPY: the return given up by holding the pair instead of the market.
- **The trend portfolio** is compared with an equal-weight, daily-rebalanced, long-only basket of its own legs (§28 below): the same assets owned without the trend signals or the vol scaling.
- **Benchmarks are the tougher comparison.** They use the same evaluated window, rebased to E₀, fully invested (w = 1, no cash) and with no costs.

### 15. Trades (§7)
- **Trades come from the executed position.** A trade is a maximal run of non-zero, same-sign executed position `pos`. Using the unlagged signal would date every trade k days early.
- **Attribution follows P&L timing.** Day t's return belongs to the trade held into day t (`held_id`, yesterday's trade), and day t's cost to the trades being exited and entered. Masking returns by `pos != 0` would drop each trade's last day, which is the day its exit fills.
- **Splitting a day's cost.** Let a = Σ|w_prev| (exposure being closed) and b = Σ|w| (exposure being opened). The exiting trade gets the fraction a/(a + b) of the day's cost, and the entering trade the rest. This is exact in every case:
  - entry from flat: a = 0, so the new trade pays all of it;
  - exit to flat: b = 0, so the old trade pays all of it;
  - direct flip: every leg's weight changes sign, so turnover = Σ|w − w_prev| = a + b. The split charges c·a to the closing trade and c·b to the opening one;
  - rebalancing inside a trade (`rolling` hedge mode, or vol resizing): both parts land on the same trade.

  The borrow fee is a holding cost, like the day's return, so it goes to `held_id`.
- **Reconciliation.** Each day's excess return is split between `held_id` and `trade_id`, and a fully flat day contributes exactly 0. So the trades' P&L, open and closed, sums to the total excess return over the window. The tests require agreement within 1e-9 on 144 random runs, covering both hedge modes, the vol overlay with in-trade resizing, lags of 0 to 2, borrow fees up to 2,000 bps and the long-only switch (§12).
- **Bookkeeping (§7.3).**
  - `exit_date` is the last day the trade is held into: the day its exit order fills. The last day it appears in `trade_id` is the day before.
  - A trade entered on the final day is open, with 0 holding days and a P&L equal to its entry cost.
  - `signal_date` and `exit_signal_date` are k trading days before entry and exit. That is where the dashboard draws the markers, on the score that triggered them.
- **The trend portfolio's trades are per leg** (§7.4), with the same rules applied to each leg's own exposure (§28 below).

### 16. Metrics (§8)
- **Excess returns for the risk-adjusted ratios, total returns for the account.** Sharpe, Sortino and beta measure return above cash per unit of risk. Total return, CAGR, volatility and drawdown describe the account itself.
- **Sharpe** = mean(excess) / std(excess) · √252.
- **95% CI (Lo 2002).** For iid normal returns, the daily Sharpe estimate has asymptotic variance (1 + SR²/2)/n. The interval is SR_d ± 1.96·√((1 + SR_d²/2)/n), annualized by √252. Autocorrelation and fat tails make the true interval wider (see Limitations).
- **Sortino** = mean(excess) / DD · √252, with DD = √(mean(min(excess, 0)²)) over all n days: the downside deviation at a target of zero excess return. Averaging the squared losses over the losing days only would inflate DD by √(n / n_losing) and penalize strategies with few losses.
- **CAGR** = (E_n/E₀)^(252/n) − 1: the constant annual rate that compounds to the same growth. It is `null` after ruin (E_n ≤ 0), where the total return of −100% already says everything.
- **Max drawdown** is the minimum of E_t / max_{s≤t} E_s − 1: the worst peak-to-trough loss, counting the baseline.
- **Beta and correlation** regress the strategy's excess return on SPY's excess return, r_SPY − rf. Using excess returns on both sides removes the shared rf term, which would otherwise create spurious co-movement between any two return series.
- **Exposure statistics.**
  - `pct_days_in_market` counts the days with a position held into them.
  - `avg_gross_exposure` averages Σ|w_prev| over all days, with flat days counting 0, so it shows leverage.
  - `ann_turnover`, `ann_cost_drag` and `ann_borrow_drag` annualize the daily means.
- **Null thresholds.** A Sharpe ratio, Sortino ratio or beta needs n ≥ 2, a denominator above 1e-12, and some exposure.
  - A strategy that never trades has excess exactly 0 and a `null` Sharpe ratio, not 0 and not 7e15.
  - 1e-12 is far below any real daily standard deviation (a volatility of 1% a year is 6e-4 a day) and far above rounding noise (about 1e-17).
- **Warnings** flag what the numbers alone would hide:
  - an IS Engle–Granger p ≥ 0.05: the pair may not mean-revert at all;
  - fewer than 10 closed trades in a segment: a win rate on 3 trades means nothing;
  - a non-positive hedge ratio at entry: the "hedge" adds exposure instead of offsetting it;
  - the data problems of §4.4;
  - ruin;
  - a leverage cap that binds on more than a quarter of invested days (of invested asset-days, against each leg's cap, for the trend portfolio);
  - a full-period break-even cost below twice the request's cost (§23 below).

### 17. Parameter sensitivity (§9)
- **One split.** The split is resolved once, for the request's own configuration, exactly as `/api/backtest` would. Every configuration then uses that date; otherwise each L would compute its own 70% point.
- **Each configuration keeps its own t₀.** Shorter lookbacks start signalling earlier. Forcing the grid's latest t₀ on them would break §6.1's "no signal before t₀" check.
- **A common scoring window,** [the latest first-evaluated day, the split). Comparing an L = 20 Sharpe ratio over seven years with an L = 120 one over six would mix the effect of the parameter with the effect of the period.
- **IS only.** OOS returns exist internally but are never scored or returned, so looking at the grid can't contaminate the OOS segment.
- **Enough history.** The common window must be at least 252 days. The grid can fail where a single backtest succeeds, so the error names the lookback that is too long.
- **Multiple testing.** The best of 20 configurations is biased upwards by the selection itself. The heatmap outlines the request's own cell, not the best one, and reports the deflated Sharpe ratio of both (§25 below).

### 18. Monte Carlo (§10)
- **Input checks (§10.1):**
  - at least 252 days, so μ̂ and σ̂ rest on at least a year;
  - every R > −1, so ln(1 + R) exists;
  - the strategy must have held a position. A cash-only series varies with the T-bill rate and passes any σ̂ floor, but projecting it says nothing about the strategy;
  - σ̂ ≥ 1e-8, which rejects degenerate input and keeps the histogram edges strictly increasing;
  - `block_len` ≤ n/2, which keeps blocks short relative to the history, so resampling actually mixes it.
- **Estimation** uses log returns ℓ: μ̂ = mean(ℓ) and σ̂ = std(ℓ, ddof = 1), per day.
- **Zero drift** sets μ̂ = 0 for GBM and demeans ℓ for the bootstrap. It is a risk view that doesn't extrapolate a noisy historical mean: a year of data estimates the annual mean return with a standard error equal to the annual volatility.
- **GBM without −½σ².** The model is ℓ_t = μ + σZ_t, so ln(E_T/E₀) ~ N(μT, σ²T).
  - μ̂ is already the mean log return, so it is the right drift for the log price.
  - The Itô correction converts an arithmetic drift into a log drift: if E[R] = m, then E[ln(1 + R)] ≈ m − σ²/2. Applying it to μ̂ would subtract the variance a second time.
  - The test detects a stray −½σ̂² term at about 16 standard errors (§12).
- **The circular moving-block bootstrap.** Resampling blocks of consecutive historical days keeps what GBM ignores: fat tails (every value is a real day), and volatility clustering and short-range autocorrelation within each block (Künsch 1989).
  - The circular version wraps around the end of the history, so every day appears in exactly `block_len` of the n possible blocks and is equally likely to be drawn. The plain moving-block bootstrap under-samples the first and last days (Politis & Romano 1992).
  - Longer blocks keep more dependence but leave fewer independent blocks per path. The default of 10 days keeps two weeks of dependence, while a one-year path still combines about 25 independent blocks.
- **Memory.** Each model's paths are built in place in one (T + 1) × N array. The bootstrap builds its indices as int64 and uses `np.take(..., mode="wrap")`, which avoids copying the indices and buffering the output (§10.3). At the API caps, the bench measured a peak of 216 MB, 2.13 times the 102 MB path array (README).
- **Outputs (§10.4):**
  - **Draw order is fixed:** GBM increments, GBM sample paths, bootstrap increments, then bootstrap sample paths, and from Phase 7 the stationary bootstrap's increments and sample paths (§29 below). A seed reproduces every output, and the model added later draws last, so it doesn't change the existing ones (tested).
  - **Bands** are quantiles of the log paths, exponentiated. Quantiles commute with any increasing function, so this equals the quantiles of the equity multiples, apart from the linear interpolation between neighbouring paths. `overwrite_input=True` avoids copying the array.
  - **VaR and CVaR** are positive loss fractions of the terminal return: VaR_α = −Q_{1−α}, and CVaR_α = −mean(terminal | terminal ≤ Q_{1−α}), the average loss in the worst 1 − α of outcomes. Unlike VaR, CVaR is coherent (subadditive), and it measures the tail that VaR only marks the start of.
  - **The histogram's 50 bins are shared by every model** (three from Phase 7), with edges at the pooled 0.1% and 99.9% quantiles, so the distributions compare bar by bar. Pooling the third model moves the edges, so a seed's counts differ from Phase 6's, while its paths, bands and risk figures don't. Values beyond the edges go into the end bins, because `np.histogram` silently drops them, and those values are the tail the chart exists to show. Each model's counts sum to N.
- **Closed form (§10.5).** Under GBM the terminal log return is X ~ N(m, s²), with m = μ̂T and s = σ̂√T.
  - With p = 1 − α and q = m + sΦ⁻¹(p), the loss 1 − e^X exceeds 1 − e^q with probability p, so VaR_α = 1 − e^q.
  - For CVaR, the partial expectation of a lognormal (complete the square in the normal integral) is E[e^X; X ≤ q] = e^{m + s²/2} Φ((q − m − s²)/s). So

    CVaR_α = E[1 − e^X | X ≤ q] = 1 − e^{m + s²/2} Φ((q − m − s²)/s) / p.

  - The simulated GBM figures must agree within 4 standard errors: the delta-method error for VaR, and for CVaR the full asymptotic error, which includes the error in the estimated quantile (§12). This checks the simulation, the estimators and the formulas against one another.
- **Seeds.** Without a seed, the API draws one with `secrets.randbelow(2**53)` and returns it, so any run can be repeated. 2^53 bounds the integers a JavaScript number holds exactly. `SeedSequence().entropy` is 128 bits and would be rounded in the browser.

### 19. API (§11)
- **Handlers are plain `def`.** The work is CPU-bound and yfinance blocks. FastAPI runs `def` handlers in a thread pool, whereas blocking inside an `async def` would stall every other request.
- **One error shape,** `{"detail": "..."}`: 422 for invalid input or too little data, 404 for an unknown ticker, and 502 or 503 for upstream failures (Deviations, Phase 2, item 5).
- **Fields that don't apply are ignored, not rejected,** and echoed as `null`. The dashboard can switch strategies without clearing fields, and the echo shows exactly what ran.
- **Validation ranges** keep requests inside what the engine is tested on (for example, L from 10 to 252 for mean reversion and pairs, 21 to 504 for trend and the trend portfolio, and δ from 1e-7 to 1e-2 for the Kalman model), and keep Monte Carlo inside its memory budget: n_paths × (horizon + 1) ≤ 12.7 million.
- **The vol target's range depends on the strategy:** 1–50% a year for the optional overlay, and 5–60% per asset for the trend portfolio, whose target resolves to 40% when it is `null`. The check therefore runs after the strategy is known, in the model validator, rather than as a field constraint.

### 20. Performance and the diagnostics cache (§14)
- **Timings come only from `scripts/bench.py`.** It records the hardware and library versions, uses its own database, and stops if a timed request reaches the network. Each figure is the median of 10 runs after a warm-up. Requests go in process through FastAPI's TestClient; the stage breakdown calls each step of the route directly.
- **Why a cache.** On a warm 15-year KO/PEP backtest, the two Engle–Granger tests take about half the request (bench: 184 ms of 361 ms), and the target is 300 ms (§14).
  - The diagnostics depend only on the two legs' adjusted closes and the split date. They don't depend on z_entry, z_exit, costs or the hedge mode.
  - The API keeps the last 256 results, evicting the least recently used. The key is the tickers, the IS and OOS windows, and a SHA-256 of the dates and both adjusted-close arrays.
  - Any change to the data changes the key, including a refetch onto a new adjustment basis.
  - A hit returns the same immutable result, so responses are identical with or without the cache (tested).
  - `stats.py` stays pure. The API passes the cache's lookup into `build_report`.
- **What it can't speed up.**
  - The first request on new data still runs both tests.
  - Under the default 70% split, L and the execution lag move t₀, and with it the split date, so changing them misses too. With `split_date` fixed, they hit.
  - The README gives the bench figures for each case.
- **Speed without changing a result (2026-10-20).** A refactor cut the first pairs request from 414 to 299 ms, the sensitivity grid from 689 to 423 ms, the walk-forward from 728 to 385 ms and the Monte Carlo request from 682 to 384 ms (bench medians, one machine, one day). Every change was checked to leave the API's responses byte-identical on 27 requests covering every strategy and option, apart from the Monte Carlo's wall-clock timings:
  - **One OpenBLAS thread.** The Engle–Granger tests' regressions are small, and OpenBLAS's thread pool costs more than it saves on them. The setting is made in `app/__init__.py`, before NumPy loads.
  - **The grids skip the trade bookkeeping.** `run_backtest(trades=False)` leaves out §7's attribution and trade records, which the sensitivity grid and the walk-forward never read; every other output is the same (tested).
  - **Whole-array code** for the §5.1 entry and exit rule, the trend portfolio's sizing (§5.8) and the JSON conversion of the series. These compute the same values in the same order, so the floats are identical, not just close.
  - **Monte Carlo.** The bands' quantiles are computed in four blocks of days on threads, which run in parallel because NumPy releases the interpreter lock while it partitions; each day's quantiles are computed exactly as before. The stationary bootstrap's running maximum of block starts (§10.7) is a loop of whole-row maxima, which gives the same integers about 20 times faster than an accumulate down the columns of a row-major array.
  - **What was left alone.** The Engle–Granger tests stay statsmodels' `coint`, as §5.4 specifies; reimplementing them could change the last digits of the p-values. The sensitivity grid still computes the pairs score once per cell rather than once per L, and the daily returns stay in pandas. Under the profiler each costs a few milliseconds per grid cell, and saving them would complicate `run_strategy`'s interface or risk the last bits of multi-leg sums.

### 21. Dashboard conventions (§13)
- **Only the Run button sends requests,** and hidden controls are never sent, so what runs is exactly what the sidebar shows. The walk-forward has its own button, and the run history reads stored runs only when its panel is opened (§31); loading a URL sends nothing.
- **Every synchronized chart uses the same `series` rows.** Nulls render as gaps rather than being dropped, because dropping rows would shift the shared index and misalign the charts.
- **Trade markers sit at signal dates,** when the decision was made, not at fill dates, so they line up with the score that triggered them.
- **The heatmap's colour scale is centred on a Sharpe of 0,** so colour shows the sign. The request's own cell is outlined, not the best one.
- **The histogram's red line is at −VaR95:** the loss that the worst 5% of outcomes exceed.
- **An assumptions footer** restates the cost, the borrow fee, the shorting setting, the lag, the rf source, daily rebalancing and the vol settings beside the results they qualify.
- **The cost sweep follows the KPI grid's IS / OOS / Full toggle,** so the break-even line and the KPI tile always describe the same days.
- **The rolling charts share the equity chart's rows and `syncId`,** so hovering a date shows the equity, the drawdown and the trailing-year Sharpe and beta of that day together.
- **Calendar months use the heatmap's diverging scale,** bounded by the largest |monthly return| but at least 2%, so a quiet strategy's months aren't all saturated. Partial periods are hatched.
- **The walk-forward has its own button,** because it backtests the whole grid. Its result stays when the sidebar changes, marked as stale until the sidebar builds the same request again, so a slow run is never discarded silently.
- **Alternate test folds are shaded** in the walk-forward chart, so each fold's boundaries are visible, and the tooltip names each day's fold. The fold table marks every change of choice.
- **The trend portfolio's signal panel has one small score chart per leg,** each with only that leg's trade markers, then the legs' executed weights, the gross exposure against the cap, and one vol-forecast line per leg against the per-asset target. They share the `syncId` with the other charts, and each leg keeps its colour across them.
- **Benchmark labels name what the benchmark is:** "Buy & hold SPY" for a single asset or a pair, and "Equal-weight basket" for the trend portfolio, in the equity legend, the KPI tiles, the calendar and the walk-forward.

### 22. Borrow fee and the long-only switch (§5.9, §6.1)
- **What a short costs.** Shorting means borrowing the shares, and the lender charges an annual fee on their value. The engine charges b basis points a year on the short notional held into each day: borrow_t = (b / 10,000 / 252) · Σ_i max(−w_i,t−1, 0). It accrues daily on the same 252-day basis as rf. The sale proceeds still earn rf (§4 above), so the fee is the only extra cost.
  - It covers a single-asset short and the short leg of a pair. A long-only run, and every flat day, pay exactly 0 (tested).
  - It is a holding cost, so it is attributed to the trade held into the day (§15), and the trades still reconcile with the total excess return.
  - *(engine)* `{"strategy": "pairs", "tickers": ["KO", "PEP"], "start": "2015-01-02", "end": "2025-12-31", "borrow_bps": 50}`: the pair holds a position on 42.1% of days, about half of it short, so the fee costs 0.11% a year (`ann_borrow_drag` 0.00106) and moves the full-period Sharpe ratio from −0.28 to −0.29.
- **A zero fee changes nothing.** With b = 0 the term is +0.0, and x − 0.0 = x in floating point, so every output is bit-identical to the earlier phases. This was checked two ways:
  - a test that excess equals gross excess − cost bit for bit;
  - a one-off comparison of 11 API responses on real data (6 backtests, 3 grids and 2 Monte Carlo runs), from the Phase 4 code and from this code. With the Phase 5 fields removed, they were identical as JSON text, which also compares the sign of zero.
- **The long-only switch** (`allow_short=false`) clips the signal at 0 after the entry rule, so a short signal means flat.
  - The §5.1 rule runs two hysteresis legs, long and short, that are never on together. The clipped signal is therefore exactly the long leg. A long-only run has exactly the long trades of the long/short run: the same dates, and the same P&L to the last bit (tested).
  - The P&L matches because a direct flip from short to long costs 2c, split c/c between the two trades (§15), and an entry from flat costs c.
  - Pairs return 422: a pair's hedge is its short leg.
  - *(engine)* SPY trend, monthly signal, 2006-01-03 to 2025-12-31: long/short has a Sharpe ratio of 0.32, a CAGR of 5.9% and a maximum drawdown of −45.2%; long-only (`allow_short: false`) has 0.57, 9.2% and −33.7%, invested on 80.2% of days. Over these 20 years, shorting SPY when its trend turned negative lost money relative to holding cash.

### 23. Cost sweep and break-even cost (§8.1)
- **Why one run is enough.** Positions are decided before costs. Signals, weights and turnover don't depend on the cost c, so the daily excess return is exactly linear in it:

  excess_t(c) = gross excess_t − borrow_t − c · turnover_t.

  The whole sweep comes from one run. The test re-runs the backtest at every grid cost and matches the sweep within 1e-12, and the sweep at the request's own cost equals the run's Sharpe exactly (the same expression on the same numbers).
- **The break-even cost.** A Sharpe ratio is 0 exactly when the mean excess return is 0, so

  c* = mean(gross excess − borrow) / mean(turnover), in basis points (× 10,000).

  It is `null` in two cases, and the dashboard says which:
  - no turnover: the strategy didn't trade in the segment;
  - a numerator ≤ 0: the strategy loses money before costs, so no cost makes it break even.
- **The grid,** 0, 2, 5, 10, 20, 50 and 100 bps plus the request's own cost, runs from frictionless to the cost of an illiquid stock.
- **Ruin.** A ruined run has no sweep: clipping the ruin day at −100% makes returns non-linear in c, and another cost would move the ruin day. A higher grid cost can also cause ruin that the run's own cost doesn't. That cost's Sharpe is then `null` for every segment containing its ruin day or starting after it (§6.2's rule, applied per cost).
- **The warning** fires when the full-period break-even is below twice the request's cost. Real costs are uncertain by at least that factor once spreads and market impact are counted, so such an edge may not exist after costs.
- *(engine)* KO/PEP pairs, 2015-01-02 to 2025-12-31, defaults: the in-sample Sharpe ratio falls from 0.27 at zero cost to 0.04 at the request's 10 bps and −0.19 at 20 bps; the in-sample break-even is 11.6 bps, so the edge barely covers the cost assumed. Out of sample the strategy loses money before costs (a Sharpe ratio of −0.71 at 0 bps), so the out-of-sample and full-period break-evens are `null`.

### 24. Rolling and calendar statistics (§8.2)
- **Rolling 1-year Sharpe and beta.** The full-period Sharpe ratio hides when a strategy worked. Each day's value uses the trailing 252 evaluated days, with §8's rules:
  - the window must hold a position on at least one day. An entry day pays the cost before any position is held, so a window ending on one can have a non-zero standard deviation and no exposure; it is `null` (tested);
  - neither standard deviation may be below 1e-12.

  The windows are computed with sliding-window views, which give the same numbers as computing each window separately, bit for bit (checked with numpy 2.5). Consecutive windows share 251 of their 252 days, so the series is smooth by construction. A one-year Sharpe ratio has a standard error of about 1.0 (§13 above), so a dip below zero for a few months is weak evidence of anything.
- **Calendar returns** compound total returns within each month and year of the evaluation window: prod(1 + r) − 1. The months of a year therefore compound exactly to its annual figure (tested). The benchmark is compounded the same way.
- **Partial periods.** A period is partial when the window doesn't cover every run-calendar day of it. That happens to the first period whenever the window starts after its first trading day.
  - The last period is also partial after ruin, or when the request's `end` falls before the period's last calendar day.
  - The rule is conservative: a run ending on Friday 29 October marks October partial unless `end` is 31 October, though no trading day was missed.

### 25. Probabilistic and deflated Sharpe ratios (§9.1)
- **The problem.** The best of N configurations shows a higher Sharpe ratio than its true one, even when no configuration has any skill: the maximum of N noisy estimates is biased upwards. The deflated Sharpe ratio (DSR, Bailey & López de Prado 2014) asks whether the best result beats what luck alone would produce.
- **The probabilistic Sharpe ratio (Bailey & López de Prado 2012).** For n iid daily returns with skewness γ₃ and kurtosis γ₄, the estimated daily Sharpe ratio ŜR has standard error

  σ(ŜR) = √((1 − γ₃·ŜR + (γ₄ − 1)/4 · ŜR²) / (n − 1)).

  So PSR(SR*) = Φ((ŜR − SR*) / σ(ŜR)) is the probability that the true Sharpe ratio exceeds SR*.
  - **Why skewness and kurtosis.** The Sharpe estimate is noisier for negatively skewed and fat-tailed returns. A strategy that earns small, steady gains with occasional crashes looks better than it is, and this formula widens its uncertainty. The engine uses the bias-corrected sample moments, `scipy.stats.skew(x, bias=False)` and `kurtosis(x, fisher=False, bias=False)`.
  - **Why daily.** The formula counts n daily observations, so ŜR must be the daily Sharpe ratio. Annualizing first would multiply the γ₃·ŜR term by √252 and the ŜR² term by 252.
- **The expected maximum under no skill.** For N independent standard normal variables, extreme-value theory gives E[max] ≈ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)), with γ the Euler–Mascheroni constant. Scaling by the spread of the trials' daily Sharpe ratios, √V, gives SR₀*: the Sharpe ratio the best trial would show if none had any skill.
  - It grows with N (tested for N = 2 to 200): more trials, a higher bar.
  - It is `null` with fewer than 2 trials, or when the trials' Sharpe ratios don't vary (V < 1e-24), because the spread of the grid is all it has to go on.
- **DSR = PSR(SR₀*),** reported for the request's own cell and for the best cell (the highest IS Sharpe, the first in grid order on a tie).
  - *(engine)* KO/PEP pairs, 2015-01-02 to 2025-12-31: 20 trials and an SR₀* of 0.42 (annualized). The request's cell (L 60, z 2.0) has a DSR of 0.15 and the best cell (L 20, z 3.0, Sharpe 0.23) 0.30, far below 0.95, the usual bar for calling a result significant. On SPY trend, 2006-01-03 to 2025-12-31, the five trend cells are so alike that SR₀* is only 0.09, and the best cell's DSR (0.65) is little more than its plain PSR against zero.
- **Calibration** *(prototype, 2026-10-17)*. On 200 seeds of pure-noise prices, with simple returns iid N(0, 1.5%²), rf = 0 and the default 20-cell grid run through `run_strategy` exactly as `run_sensitivity` does, the share of seeds whose best cell had a DSR above 0.95 was:

  | Grid | 0 bps | 10 bps |
  |---|---:|---:|
  | Mean reversion | 3.0% | 1.5% |
  | Pairs (two independent random walks) | 2.5% | 1.5% |

  With 20 independent iid normal trials it was 0.5%. The specification's bound is 10%. The permanent test runs the same mean-reversion grid, vectorized, on its own 200 seeds: 2 pass (1.0%).
- **Correlated cells pull both ways.** Neighbouring cells are strongly correlated.
  - That makes N overstate the number of independent trials, which raises the bar, as §9.1 notes.
  - It also shrinks V, the spread of the trials' Sharpe ratios, which lowers the bar. As the cells approach perfect correlation, V shrinks towards 0 and the DSR approaches the plain PSR against 0, which passes 5% of the time with no skill. (At V < 1e-24 the DSR is `null`.)
  - Measured, the correlated grids pass more often than independent trials (1.5–3.0% against 0.5%) but stay below the nominal 5%. The DSR remains conservative here, though less so than the N-overstatement argument alone suggests.

### 26. Walk-forward analysis (§9.2)
- **Why.** The sensitivity grid (§17 above) chooses on in-sample data, and the OOS segment is then a fair test only once: a user who re-tunes after looking at it has used it for selection. Walk-forward makes the selection part of the protocol. In each fold the grid's best configuration is chosen on a training window and traded on the test window that follows. Stitched together, the test windows form a track record in which every day was traded with parameters chosen before it.
- **Folds** start at s, the grid's common start (§17): from s on every configuration is trading, so all of them compete on the same training days.
  - Rolling windows (the default) use the last T days, so the choice can follow a change of regime, but each estimate rests on T days only. Anchored windows grow from s: the estimates steady and the choice changes less often.
  - The defaults are 756 training days (three years) and 63 test days (a quarter). A last test window under 21 days is dropped, because a Sharpe ratio over a few days means nothing. At least two folds are required.
- **Selection is causal.** Fold j ranks the configurations by their own §8 Sharpe ratio on the training window, which ends at the close of b_j − 1, the day before the test window. A configuration is eligible if it was evaluating from the window's first day (always true from s on), was not ruined by its last day, and has a defined Sharpe ratio there (one that never held a position has none). Ties go to grid order. With nothing eligible the fold holds cash.
  - Tested by changing prices after each fold's training window: the fold's choice never moves. The stitched series also passes the §12 look-ahead test, like any backtest.
- **Stitching by executed weights.** From the close of b_j − 1 to the close before the next test window, the walk-forward holds exactly the weights the chosen configuration holds at each close, which already include its execution lag k. A position that configuration opened before the fold is inherited, as it would be by a trader switching to it.
  - §6.1 then applies to the stitched weights: returns, the borrow fee, turnover and costs. A switch is an ordinary trade at the close of b_j − 1, from the position held into that close to the new configuration's weights. The walk-forward enters from flat at the close of b₀ − 1, its first evaluated day.
  - Each day belongs to one fold: fold j owns its test days, and fold 0 also its entry day. The cost of a switch falls on the last day of the fold being left, the way a trade's exit cost belongs to that trade (§15 above). A fold's test Sharpe is therefore what the walk-forward actually earned on its days, after switching costs, and the folds' excess returns add up to the total (tested to 1e-12).
  - With a one-configuration grid, the stitched series is that configuration's own, apart from the entry from flat on the first day (tested).
- **Outputs.** The §8 metrics of the stitched series. Its positions span configurations, so it has no trades, and the four trade fields (closed and open trades, win rate, holding days) are `null`. Turnover and the cost and borrow drags are reported, because the walk-forward pays them. A day counts as invested when a position was held into it. The comparisons are the request's own configuration held fixed over the same days, and the benchmark.
- **What it shows** *(engine)*. Each figure is from `/api/walk-forward` with every walk-forward default:
  - KO/PEP pairs, `{"strategy": "pairs", "tickers": ["KO", "PEP"], "start": "2015-01-02", "end": "2025-12-31"}`: 30 folds from 2018-06-25. The walk-forward chose 10 of the 20 configurations and changed its choice 14 times. The configurations it chose had an average training Sharpe ratio of 0.29 and an average test Sharpe ratio of −1.03 (over the 19 folds that held a position). The stitched Sharpe ratio was −0.97 (95% CI −1.69 to −0.26), against −0.43 for L 60 / z 2.0 held fixed and 0.66 for SPY. Re-choosing followed noise: the best Sharpe ratio of the last three years did not predict the next quarter.
  - The same with `"anchored": true`: 7 changes and 3 distinct configurations, and a stitched Sharpe ratio of −0.90.
  - SPY trend with a 10% vol target, `{"strategy": "trend", "tickers": ["SPY"], "start": "2006-01-03", "end": "2025-12-31", "vol_target": 0.1}`: 62 folds from 2010-08-02, with every one of the 5 lookbacks chosen at some point. The stitched Sharpe ratio was 0.23, against 0.49 for L 252 held fixed and 0.80 for buy-and-hold.
- **Why it remedies OOS reuse, and what it can't fix.** Each choice is computed before its test window, so re-running the walk-forward on new data can't leak a test window into its own selection. The grid, the window lengths and the strategy are still chosen by the user, though. Choosing them after seeing a walk-forward result makes it in-sample again (Limitations).

### 27. The Kalman-filter hedge ratio (§5.7)
- **Why.** Rolling OLS (§8 above) weighs each of the last L days equally and then drops a day abruptly, so β jumps when an unusual day leaves the window, whatever the market is doing. The Kalman filter treats the regression's coefficients as a state that drifts slowly, and old information fades gradually at a rate set by δ instead of through a hard window (Chan 2013).
- **The model.** The state s_t = (α_t, β_t)ᵀ is a random walk, s_t = s_{t−1} + w_t with w_t ~ N(0, q·I₂), observed through y_t = α_t + β_t x_t + e_t with e_t ~ N(0, R), on the rebased log prices of §8. With q = δ/(1 − δ), δ = 0 is a constant state and larger δ lets β move faster.
- **The OLS start and the δ = 0 identity.** The filter starts from the OLS fit on days 0…L−1: ŝ = OLS of y on (1, x), R = SSR/(L − 2) and P = R(HᵀH)⁻¹.
  - That is the posterior of the state after those L days under a flat prior, with R treated as known: N(ŝ, R(HᵀH)⁻¹).
  - With q = 0 each later step is the recursive least-squares update, which adds one observation to the regression. So the filtered state on day t is OLS on days 0…t. The test checks this against an explicit expanding-window OLS within 1e-8 on 20 sampled days; the prototype measured 4.2e-15 over 1,500 days *(prototype, 2026-10-19)*.
  - With q > 0, the predict step P + q·I adds uncertainty every day, which down-weights older observations geometrically.
  - The start is written out in closed form with centred sums (the 2×2 inverse), not with `np.linalg.solve`, and the guards run first. On a constant window, `solve` returned large, meaningless coefficients for the constants 0.0123, 0.1, 0.3 and 1/3, and raised only for an exact 0 *(prototype)*, so it can't serve as the flat-window check (§5.7).
- **The innovation z-score.** e_t = y_t − (α̂_{t−1} + β̂_{t−1} x_t) is the error of a prediction made with data through t − 1, and S_t = h P hᵀ + R, with P after the predict step, is its variance under the model. Under the model z_t = e_t/√S_t is N(0, 1) and independent across days, so z_entry = 2 keeps the meaning of §6 above: a two-standard-deviation surprise.
  - **The start-up step biases it in practice.** R comes from an OLS fit that holds the state fixed over days 0…L−1. If the true β drifts during those days, the residuals absorb the drift, R is overstated, and z is too small. On data simulated from the model with a random-walk state from day 0, over 200 seeds for each of eight settings (δ 1e-4 or 1e-5, R 1e-4 or 1e-6, L 60 or 252), the std of z ranged from 0.23 to 1.14. No seed met the §12 bounds (mean within 0.05 of 0, std within 0.05 of 1) in seven of the settings, and 12% did in the eighth *(prototype)*. Given the true R, the std was 0.97–1.13 and 98–100% of seeds met the bounds.
  - With the state constant over the start-up window, as the OLS start assumes, only the sampling error of the estimated R remains: 80–95% of seeds met the bounds at L = 252, and all 200 at L = 2,000. The §12 test uses that setup (Deviations, Phase 7, item 1).
- **The Joseph-form update,** P = (I − Kh) P (I − Kh)ᵀ + R KKᵀ, equals the short form (I − Kh) P in exact arithmetic. It is a sum of a congruence and a positive semidefinite term, so it stays symmetric positive definite under rounding. The tests check symmetry and positive eigenvalues on every day.
- **Speed.** The recursion is unrolled into scalar arithmetic on the 2×2 matrices and matches the matrix form within 1e-12 (§12). The bench puts `run_strategy` at 31.6 ms for 15 years of KO/PEP, against 20.0 ms with rolling OLS (README, 2026-10-20).
- **The default δ is 1e-6** (Deviations, Phase 7, item 2). On real data the dispersion of z depends on δ. Too large a δ makes the model expect more drift than there is, so S overstates the actual innovations and z shrinks. Too small a δ does the reverse. Measured *(engine)* on KO/PEP from 2015-01-02 to 2025-12-31, every other setting default, over the evaluated days (from 2015-04-01, or 2015-03-31 for rolling OLS, whose first score comes a day earlier):

  | δ | std of z | Share of days with \|z\| > 2 | Closed trades | Days in market | β range | Sharpe (full) |
  |---|---:|---:|---:|---:|---:|---:|
  | 1e-4 | 0.35 | 0.3% | 4 | 0.5% | 0.45–0.91 | 0.02 |
  | 1e-5 | 0.63 | 1.1% | 10 | 2.2% | 0.53–0.96 | −0.08 |
  | 1e-6 (default) | 1.13 | 7.1% | 19 | 17.4% | 0.53–0.95 | −0.35 |
  | 1e-7 | 1.88 | 22.9% | 27 | 42.5% | 0.58–1.14 | 0.05 |
  | rolling OLS, L = 60 | 1.40 | 15.4% | 67 | 42.1% | −0.44–1.97 | −0.28 |

  - Request: `{"strategy": "pairs", "tickers": ["KO", "PEP"], "start": "2015-01-02", "end": "2025-12-31", "pairs_model": "kalman", "kalman_delta": 1e-6}`, with the other δ values, and without the two Kalman fields for rolling OLS.
  - At 1e-6 the innovations have roughly the spread the model predicts. At the spec's original 1e-4, z almost never reaches 2, and the model makes 4 trades from 2015 to 2025.
  - δ was chosen on the dispersion of z, which doesn't look at returns, on KO/PEP and on the test pair (std 1.14 there). It is still a parameter chosen on this sample (Limitations). Maximum likelihood on the innovations would be the principled way to estimate it, and isn't implemented.
- **What it shows.** The filtered β stays between 0.53 and 0.95, while the 60-day rolling β ranges from −0.44 to 1.97 and was ≤ 0 at 3 entries (the §8 warning). The steadier hedge doesn't make the pair profitable: the full-period Sharpe ratio is −0.35, against −0.28 for rolling OLS, and both lose money out of sample.
- **Everything downstream is §8's,** with the filtered β in place of the rolling one: the entry/exit rule, both hedge modes, the unit weights and the overlay's u. The §5.4 diagnostics test the pair, not the model, so they are unchanged.

### 28. The multi-asset trend portfolio (§5.8)
- **Why.** Moskowitz, Ooi & Pedersen (2012) traded time-series momentum across 58 instruments, each scaled to the same ex-ante volatility (40% a year), and averaged the positions. One asset's trend signal is weak and noisy: 0.32 on SPY over 20 years (§10 above). Signals on loosely correlated assets diversify, which is where most of the published Sharpe ratio comes from.
- **Equal risk budgets.** Leg i holds w_i = s_i · min(σ*/σ_i, max_leverage) / N, where σ* is the per-asset target and σ_i the leg's own §11 forecast. Uncapped, each leg's ex-ante volatility is σ*/N, whatever the asset's own volatility, so a bond leg and an equity leg carry the same risk. For N uncorrelated legs the portfolio's volatility is √N · σ*/N = σ*/√N; correlated legs give more.
  - Tested: with constant-volatility assets (8–30% a year), a constant signal and a cap that never binds, each leg's realized volatility is within 20% of σ*/N after 3 half-lives (§12). The prototype measured 0.981–1.031 times σ*/N over 200 seeds × 4 legs *(prototype)*.
- **The cap breaks the budgets for calm assets.** Each leg is capped at max_leverage/N, so a leg reaches its share of the target only when σ_i ≥ σ*/max_leverage: 20% under the defaults (40%, 2×). On the diversified preset the legs' median forecasts ran from 5.7% (IEF) to 20.5% (IWM), so the legs sat at their cap on 76% of invested asset-days. At the cap every leg holds 0.2 of equity, whatever its volatility: equal notional rather than equal risk, with the equity legs carrying most of the risk. The original paper had no cap; it levered calm instruments such as bond futures as far as the target required. Measured *(engine)* on the diversified preset, 2016-01-05 to 2025-12-31:

  | Setting | Sharpe (full) | Realized vol | Max drawdown | Asset-days at cap | Average gross exposure |
  |---|---:|---:|---:|---:|---:|
  | defaults: 40% per asset, 2× | 0.01 | 15.2% | −36.6% | 76% | 1.90 |
  | `max_leverage` 4 | 0.04 | 19.3% | −45.8% | 15% | 2.66 |
  | 10% per asset | 0.04 | 5.0% | −12.3% | 3% | 0.74 |
  | long-only (`allow_short` false) | 0.45 | 13.8% | −24.7% | 81% | 1.16 |
  | equal-weight basket (benchmark) | 0.66 | 11.9% | −23.1% | — | 1 |

  - Request: `{"strategy": "trend_portfolio", "tickers": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"], "start": "2015-01-02", "end": "2025-12-31"}`, then with `"max_leverage": 4`, `"vol_target": 0.1` or `"allow_short": false`. The first evaluated day is 2016-01-05: a 252-day lookback, then the next month start.
  - The default run has a 95% CI of −0.61 to 0.63 on its Sharpe ratio, 147 closed and 10 open leg trades, and a break-even cost of 12.4 bps, so both the cap and the break-even warnings fire.
  - The four trend ETFs (SPY, QQQ, GLD, TLT) over the same window: 0.30, against 0.94 for their basket.
  - Over this decade, shorting the assets whose 12-month trend had turned negative lost money, as on single-asset SPY (§22 above): the long-only portfolio had a Sharpe ratio of 0.45. Ten years of one portfolio is one sample, though, and the CI spans −0.6 to 0.6.
- **All legs start together.** t₀ is the first date on which every leg's score is valid, and every signal before it is 0. So the portfolio holds its N legs from its first day, each with its 1/N share, and §6.1's "no signal before t₀" check holds for the whole portfolio. A leg with a short history delays everything (Limitations).
- **The backtester's signal is the "invested" flag,** 1 when any leg's signal is non-zero. §6.1's checks then apply to the portfolio as a whole. Two checks are added for the legs: each leg's weight is exactly 0 where its own signal is flat, and the flag is exactly what the legs imply.
- **Trades are per leg (§7.4).** The legs enter and exit independently, so a run of same-sign position of the whole portfolio means nothing. §15's identification, attribution and bookkeeping run on each leg's own exposure, return, borrow fee and cost. Every daily term of §6.1 is a sum over legs, so the leg trades together reconcile with the total excess return.
  - Tested over 54 random portfolios (2–5 legs, lags 0–2, costs, borrow fees, the long-only switch and all three refresh schedules): the trades' P&L sums to the total excess within 1e-9, and a leg's trades never overlap.
  - Each leg's trades have exactly the dates of that leg backtested alone with its portfolio weights, and the same P&L within 1e-15 (tested).
  - Ids run 1… across the portfolio by entry date and then ticker, so the trades table reads in time order.
- **A ruin-day credit is split by gross weight** (Deviations, Phase 7, item 3). §6.2 credits the clipped amount to "the trade held into r", but several legs can be held into r. The credit is the limited-liability floor, not any leg's own P&L, so it is split in proportion to the exposure each leg held into the day, |w_{i,r−1}|. That keeps the reconciliation exact and needs no rule about which leg caused the ruin. The alternative considered, in proportion to each leg's loss that day, needs a rule for legs that gained.
- **The benchmark** is the portfolio's own assets held long in equal weights, rebalanced daily, with no costs. The comparison then isolates what the trend signals and the vol scaling add to owning the same universe.
- **Leverage at the cap is counted per asset-day.** `pct_days_at_cap` is the share of invested asset-days (leg i held into day t) on which |w_{i,t−1}| reached max_leverage/N, so its warning says "invested asset-days".

### 29. The stationary bootstrap (§10.7)
- **Why.** The circular block bootstrap's blocks have a fixed length b, so block boundaries fall at fixed positions in every path: days b and b + 1 are always independent, while days b − 1 and b never are. The resampled series is therefore not stationary. The stationary bootstrap of Politis & Romano (1994) starts a new block with probability p = 1/b each day, so block lengths are geometric with mean b, the boundaries fall at random, and the resampled series is stationary.
- **Implied dependence.** Conditional on the data, the stationary bootstrap's autocovariance at lag k is (1 − p)^k times the circular sample autocovariance. Its weights decay smoothly and reach well past b: at b = 40, lag 100 still keeps 8% of its autocovariance. The circular block bootstrap keeps lags below b only, with weights falling linearly to 0.
- **Is it less sensitive to the block length?** Politis & Romano expected the random lengths to make results less sensitive to the exact choice. Here it was the other way round. Measured *(engine)* on SPY from 2015-01-02 to 2025-12-31 (`source: "ticker"`, seed 42, 10,000 paths over 252 days):

  | block_len | 1 | 5 | 10 | 20 | 40 | auto |
  |---|---:|---:|---:|---:|---:|---:|
  | VaR95, circular block bootstrap | 16.1% | 14.3% | 14.0% | 14.3% | 13.6% | 14.8% (14 days) |
  | VaR95, stationary bootstrap | 15.2% | 14.5% | 14.0% | 12.8% | 10.9% | 13.4% (mean 12 days) |
  | VaR95, GBM (no blocks) | 15.6% | 15.6% | 15.6% | 15.6% | 15.6% | 15.6% |

  - The simulation error of each VaR95 is roughly 0.3 percentage point (the delta-method error of §18 above), so the differences between block lengths are real.
  - They come from the data: SPY's daily autocorrelations over 2015–2025 sum to −0.13 over lags 1–20 and −0.20 over lags 21–100 *(prototype)*. Longer blocks keep more of that negative dependence, which narrows the spread of 252-day returns, and the stationary bootstrap's long tail of block lengths keeps the most.
  - A check confirmed both models do what the theory says: on 20,000 paths, each model's standard deviation of the 252-day log return matched its theoretical value within 1% for every block length above (for b = 40: 0.140 for the stationary and 0.149 for the circular bootstrap, against 0.178 with no dependence) *(prototype)*.
  - Lahiri (1999) shows that, at their best block lengths, the stationary bootstrap's variance estimates are less accurate than the moving-block bootstrap's. The engine reports both, and the gap between them is a rough measure of how much the answer depends on modelling choices.
- **Vectorized generation.** For a chunk of paths, the engine draws the day's restart flags and a candidate start for every day, then finds, for each day, the row where its block began: last = the running maximum of t over the restart rows. The index is start[last] + (t − last), modulo n. Each path consumes 2T draws, including starts that are never used, which is the price of vectorizing.
- **The tests (§12).**
  - Every increment is a historical value, and with ℓ = 0, 1, …, n − 1 every path splits into runs that are contiguous modulo n.
  - The share of steps that continue the previous index is within 5 binomial standard errors of (1 − p) + p/n. The p/n term counts new blocks that start, by chance, at the next index.
  - The mean run length is not tested against b, because runs are cut off at the horizon: its expectation is T/(1 + (T − 1)p), 9.655 at T = 252 and b = 10 (§12).
  - Chunked and unchunked generation give identical paths when they consume the generator in the same order.
- **Memory and time.** Chunks of 2,048 paths keep the temporaries at about 3 × T × 2,048 values. The peak at the API caps is still 216 MB, set by the block bootstrap (§18 above). The stationary model takes 141 ms of a default request, against 65 ms for the block bootstrap and 81 ms for GBM, and the three-model request takes 384 ms against its 1.5 s target (README, 2026-10-20).

### 30. Automatic block length (§10.8)
- **The estimator.** Politis & White (2004) choose the block length that minimizes the mean squared error of the bootstrap's estimate of the variance of the sample mean. To first order, b_opt = (2G²/D)^(1/3) · n^(1/3), where G = Σ_k |k| γ(k) measures how far the autocovariances γ reach and D depends on the method, D_SB = 2g² for the stationary and D_CB = (4/3)g² for the circular bootstrap, with g = Σ_k γ(k). Patton, Politis & White (2009) corrected D_SB. The sums are estimated with a flat-top lag window whose bandwidth is chosen from the data.
- **The `arch` check (§10.8).** `arch` 8.0.0 installs with the project's pins on Python 3.14 on Windows, and pip resolves it for 3.12 on Windows and for 3.12 and 3.14 on Linux, where CI installs it. Its output columns are named `stationary` and `circular`. On 20 reference series it matched an independent implementation of the formulas within 6e-16: AR(1) series with φ from 0 to 0.9 and n = 500 or 2,520, iid Student-t, a GARCH-like series, and SPY's daily log returns and their absolute values. The ratio circular/stationary was (3/2)^(1/3) = 1.1447 on every series, as D_SB/D_CB implies, and on long AR(1) series the estimates came within 4% of the theoretical optimum *(prototype, 2026-10-19)*.
- **Rounding up and clipping.** Block lengths are whole days. Rounding up means an estimate of 1.3 gives 2, keeping some dependence, rather than 1, an iid bootstrap. The upper bound is the API's maximum of 63 days. The other bound in §10.8, ⌊n_hist/2⌋, never binds, because n_hist ≥ 252 makes it at least 126.
- **It runs on ℓ itself.** With `zero_drift`, the bootstraps resample ℓ − mean(ℓ), and the estimate is the same either way, because it depends only on autocovariances. Computing it before demeaning keeps the rounding of the subtraction out of it.
- **What it chooses** *(engine)*: 14 days (circular) and 12 (stationary) for SPY over 2015–2025 (table above), 2 and 2 for the KO/PEP pairs strategy's returns, and 1 and 1 for the diversified trend portfolio's. Request: the table's, or `{"source": "strategy", "backtest": {...}, "block_len": "auto", "seed": 42}` with the §27–28 backtest requests.
- **What it optimizes.** The criterion is the variance of the mean. Volatility clustering, dependence in squared returns, doesn't enter it at all, and neither does the tail of a compounded 252-day return. In the `arch` check, SPY's daily log returns gave a stationary block length of 7.4 days, while their absolute values, which carry the clustering, gave 139 *(prototype)*. So "auto" is a defensible, data-driven default, not the best block length for VaR (Limitations).

### 31. Run history, URL state and CSV export (§11.4, §13)
- **What is stored, and why that is enough.** Every backtest stores its *resolved* request (every default filled in, the split date and the clamped end date) and its metrics, not its curves. The engine is deterministic (§14), so the resolved request is the run: re-running it on the same cached prices gives the same curves. Storing ~4,000-point arrays per run would buy nothing that a re-run can't give back.
- **Old rows are read, never rewritten.** A field added in a later phase, such as `ann_borrow_drag` (Phase 5) or `pairs_model` (Phase 7), reads back as `null` from a run stored before it existed. Rewriting old rows would put numbers in them that their engine never computed. "Load settings" fills such a field with the default the field was introduced with, which reproduces the old behaviour: no borrow fee, shorting allowed, rolling OLS. A run from another engine version is flagged, because the same settings may now compute different numbers.
- **"Load settings" loads the run's split date,** not a blank split. Under a blank split the 70% split moves with t₀ and the end date (§20 above), so the explicit date is what re-evaluates the same IS and OOS windows.
- **The comparison is of metrics only,** per segment, with the settings that distinguish the runs (strategy, tickers, period, split, L, z, pairs model, vol target, cost and engine) above them. The comparison takes up to four runs, the spec's cap, side by side.
- **The URL holds settings, never results.** A URL that reproduces the sidebar can be bookmarked or shared and always means the same run. One that held results would go stale and could be edited to show numbers the engine never produced. Only settings that differ from the defaults are written, so a URL stays short and says what was changed. L is compared with the chosen strategy's default, because the default switches with the strategy (§13).
- **A URL sends nothing on load.** Loading a page must not start a slow or cold-cache request the user didn't ask for: only the Run button sends (§21 above). Each parameter is checked only for its form (a number, a date, a ticker, an allowed value), like the sidebar's own fields; ranges are left to the API's 422, which names the rule. An invalid or unknown parameter is ignored with a notice, rather than silently, so a mistyped URL can't quietly run something else.
- **CSV files carry full precision.** Numbers are written with `String(x)`, the shortest text that reads back as the same double, so a file re-analysed elsewhere gives the dashboard's numbers exactly. The formatted values on screen are rounded for reading. Nulls are empty fields, not "—" or 0, so a spreadsheet or pandas reads them as missing. Quoting follows RFC 4180, the common CSV standard, with CRLF line endings.
- **The log scale shows growth as distance.** On a log axis equal vertical distances are equal percentage changes, so a 20-year equity curve's early years aren't flattened against its later ones. The ticks fall at 1, 2 and 5 × 10^k dollars and the axis extends to the nearest such values beyond the data, so it always has at least two labels. Over more than nine such values only the powers of ten are kept. A ruined run's equity reaches $0, whose logarithm is undefined, so the toggle is disabled there.

### 32. Offline mode and the browser tests (§4.3, §12)
- **Offline mode makes a cache miss an error.** With `QUANT_OFFLINE=1` a request whose data isn't in the cache returns 503, "Offline mode: {ticker} is not in the local cache", and never calls the network seam. A test that would otherwise download data fails loudly instead of depending on Yahoo being up, on today's prices and on a rate limit. The check runs where the seam is called, so the tests' fake network can count that it was never called.
- **The browser tests run on the synthetic market.** `scripts/seed_e2e.py` caches the test suite's synthetic data (`tests/synthetic.py`) under the tickers SYNY and SYNX (a cointegrated pair), SYNT (a trending asset) and SPY, with ^IRX at 2%, for 2012-01-03 to 2019-12-31. It writes them through the app's own `ensure_cached`, so the rows and coverage are exactly what the API would cache. The results are the same on every machine and every day, with no data licence question.
- **They test the dashboard as a user meets it:** the production build, served by `vite preview`, against a real API. They check what the unit tests can't: that the panels render with no console errors, that the error states show the API's detail, that a blank field sends nothing, that the history and "Load settings" round-trip through the database, that the trades CSV downloads, and that the URL round-trips without a request.
- **Their servers are their own.** The API runs on port 8001 against a temporary database, and the build is served from `dist-e2e/` on port 4173, so a developer's API on 8000, its price cache and a Vite server on 5173 are never used or changed. The tests share one database and run one at a time, because the history test reads what earlier runs stored.

### 33. The static demo (§13, "Static demo")
- **It replays, it doesn't compute.** GitHub Pages serves static files only, so the demo can't run the engine. It shows responses recorded from the real API, for a fixed list of scenarios, and says so in a banner with the recording date and the last date of data. Running the engine in the browser instead would mean shipping Python, NumPy, pandas, statsmodels and arch to every visitor, to reproduce numbers that a recording already holds exactly.
- **The recorded files are the API's own bytes.** `scripts/record_demo.py` sends each request through FastAPI's `TestClient` and writes the response body unchanged. Nothing is recomputed, rounded or reformatted, so the demo shows exactly what the API returned for those settings on that day's cache. The dashboard code that draws them is the same code that draws a live response.
- **A response is found by its request, never by a nearest match.** The manifest maps each request's JSON, with its keys sorted, to the file holding its response, per endpoint, because the backtest and the sensitivity grid are sent the same body. The dashboard builds its requests exactly as it does live and looks them up. A request that wasn't recorded gets "This configuration isn't in the demo. Run the project locally for live data.", rather than the result of some other settings. Python writes 2.0 where JavaScript writes 2, so the dashboard parses each manifest key and re-serializes it before comparing: the lookup compares values, not two languages' number formatting.
- **The scenarios are recorded once and read by the dashboard.** The list lives in `record_demo.py`, and the manifest carries each scenario's four requests. The demo's sidebar shows a scenario by filling the form from those requests, and a test checks that the form then builds exactly the recorded requests. A scenario added to the recorder therefore reaches the demo without a frontend change, and one the form couldn't reproduce fails the test instead of failing silently in the browser.
- **The recording doesn't touch the run history.** The requests run on a temporary copy of the price cache with the run history emptied. Missing prices are first downloaded into the real cache, as the API would do. So a recording adds no runs to the history, and the recorded run ids are 1 to 8 whatever the history holds. The demo shows "recorded" in place of a run id, because the ids belong to no database the visitor can see.
- **A replay is shown at once.** Live, a URL sends nothing on load, because a request can be slow or start a download (§31 above). A replay is one small static file, so the demo shows its first scenario on load, and choosing a scenario shows it.
- **It goes stale, and it needs a licence decision.** The results are fixed at the recording date: the dates and the seed are fixed, so a re-recording sends the same requests, but Yahoo revises and re-adjusts its history, so it may not reproduce the same numbers. The recorded series are derived from Yahoo data, and `benchmark_equity` is a rescaled adjusted price, so publishing them is a redistribution question. Until the owner answers it, `pages.yml` builds the demo but its deploy job doesn't run.

## Limitations
- **Selection and survivorship bias.** The preset pairs and ETFs were chosen with hindsight: they exist today and are known to be related. A pair that stopped moving together, or a stock that was delisted, never enters the sample, which flatters every pairs result.
- **Execution.**
  - Fills happen at the close.
  - The cost is a fixed number of basis points per unit traded: there is no spread model, no market impact, and no limit on trade size relative to volume.
  - That is reasonable for small trades in liquid large caps, and wrong for illiquid names or large size.
- **A flat borrow fee, and no financing spread for leverage.**
  - The borrow fee is one rate for every short and every day. Real fees vary by name and over time: hard-to-borrow stocks can cost several percent a year, and a lender can recall the shares. Short-sale proceeds earn the full T-bill rate, with no rebate spread.
  - Leverage is financed exactly at the T-bill rate. There are no margin calls, and ruin has limited liability.
- **The cost sweep holds positions fixed** (§8.1). A trader facing higher costs would trade less, so at high costs the sweep understates what a re-optimized strategy could earn.
- **The T-bill rate.** ^IRX is based on the 13-week bill's discount rate, which is slightly below its investment yield (by about 0.1 percentage point at a 5% rate), so cash and short proceeds earn slightly too little. Daily accrual is simple (a/252), not compounded.
- **Daily rebalancing to target weights,** with no cost for drift (§6.1): between signal changes, the model's weights stay constant without trading.
- **Pairs exits use the rolling z** while the frozen-β spread is held (§5.3), so an exit can fire slightly early or late relative to the spread actually held. With the Kalman model, they use the filtered z.
- **The Kalman model's δ and start-up window are tuned parameters.**
  - The default δ = 1e-6 was chosen on the dispersion of z for KO/PEP over 2015–2025 and for the test pair (§27 above). That uses no returns, but it is still a choice made on this sample, and the right δ differs between pairs.
  - R is estimated once, from the first L days, and never updated. If the hedge ratio drifted during those days, R is overstated and z understated for the whole run (§27).
  - The model's δ is not estimated by maximum likelihood.
- **Multiple testing.** The sensitivity grid tests 20 configurations (5 for trend), and a user may try many more by hand. The best cell's Sharpe ratio is biased upwards by the selection itself.
  - The deflated Sharpe ratio corrects for the grid's own trials only. Every configuration tried before, by hand or in earlier runs, is a hidden trial it can't count.
  - It treats the grid's correlated cells as independent, an approximation whose net effect was conservative in the calibration above (§25).
- **Rolling statistics overlap.** Consecutive one-year windows share all but one day, so the rolling series looks smoother and more persistent than the evidence behind it.
- **Tuning while watching OOS results turns OOS into IS.** The engine keeps OOS out of its own selection. But a person who changes parameters until the OOS panel looks good has used OOS for selection. The honest OOS result is the one recorded before looking.
  - The walk-forward (§26) removes this for the choice it makes itself. Choosing the grid, the window lengths or the strategy after seeing a walk-forward result reintroduces it.
- **Walk-forward selection chases noisy training Sharpe ratios.** Over three years an annualized Sharpe ratio has a standard error of about 0.6 (√(252/756), §13 above), so the best of 20 is largely the luckiest. Many distinct choices mean unstable parameters, not adaptivity: on KO/PEP the walk-forward chose 10 of 20 configurations, and their test Sharpe ratios averaged −1.03 against 0.29 in training (§26). The ranking also ignores the cost of switching, which the stitched series then pays.
- **A walk-forward switch trades at the close that completes its training window.** Fold j's choice uses returns through the close of b_j − 1 and trades at that same close, as an execution lag of 0 would (§2 above). Ranking the configurations needs that close, so in practice the switch would fill a day later. Each configuration's own signals keep their lag k; only the switch, at most once per fold, is optimistic.
- **The Sharpe CI assumes iid returns** (Lo 2002). Autocorrelation, volatility clustering and fat tails make the true interval wider, especially for strategies that hold positions for weeks.
- **Trend following on one asset is much weaker than the published results,** which come from diversified portfolios of dozens of futures: 0.32 on SPY (§10 above).
- **The trend portfolio is a small, short version of the published one.**
  - It holds 2 to 12 ETFs, not 58 futures. The presets are chosen with hindsight, and so are their histories (survivorship, above).
  - Its calendar is the intersection of its legs', so it starts after its youngest ETF's first lookback. ETFs are younger than the futures markets they track.
  - Under the default 40% per asset and 2× leverage, most legs sit at their cap, which turns the equal risk budgets into equal notional weights (§28 above).
  - Equal volatility budgets ignore correlations: two highly correlated legs, such as SPY and QQQ, carry nearly twice the risk of one.
  - Its 10-year Sharpe ratio on the diversified preset has a 95% CI of −0.6 to 0.6 (§28), so it can't confirm or reject the published result.
- **The vol forecast lags.** The EWMA reacts to a volatility spike only after it starts, so realized vol overshoots the target in crashes: the weekly overlay's yearly range above still reaches 14.0% against a 10% target.
- **Vol targeting sizes up when measured volatility falls,** including during stale-price runs. The leverage cap limits this, and the data warning flags the runs.
- **Leverage is financed at rf,** with no spread and no margin constraints beyond `max_leverage`.
- **GBM assumes iid normal log returns:** no fat tails, no volatility clustering, and constant drift and volatility.
- **The bootstraps assume stationarity** and can't produce a day worse than the worst day in the history.
- **The block length matters, and "auto" doesn't settle it.** On SPY the stationary bootstrap's VaR95 ranged from 10.9% to 15.2% across block lengths of 1 to 40 days (§29 above). The automatic estimate minimizes the error in the variance of the mean, ignoring volatility clustering and the tail (§30 above).
- **Monte Carlo ignores parameter uncertainty,** treating μ̂ and σ̂ as known, and it inherits any overfitting in the backtest it resamples: projecting an overfit strategy projects the overfitting.
- **A stored run can't always be re-run exactly.** Its curves aren't stored, and a re-run reproduces them only on the same cached prices. Extending a ticker's cached range refetches its whole history on Yahoo's current adjustment basis (§4.2), so a later re-run of an old request can differ slightly. The stored metrics keep the numbers as first computed.
- **The static demo is a snapshot.** Its results were recorded on one date from that day's cached Yahoo data, so they go stale as markets move and as Yahoo revises its history. It offers eight scenarios, each with the dashboard's default settings, and can't run anything else.
- **Yahoo data quality.** The data is unofficial and has occasional bad prints, missing days and revisions. Adjusted closes depend on Yahoo's dividend and split records. The cleaning rules catch the obvious errors, not the subtle ones.

## Deviations
Approved departures from CLAUDE.md, with reasons (§0, rule 7).

### Phase 0 (approved 2026-10-15)
1. **Frontend linter: oxlint instead of ESLint (§3).** The spec says "ESLint from the Vite template", but the current template (create-vite 9.2) ships oxlint, not ESLint. We kept the template's linter, which preserves the intent: use whatever the template provides rather than a hand-rolled config. `npm run lint` runs oxlint with the template's `.oxlintrc.json`.
2. **Test client dependency: `httpx2` instead of `httpx` (§3, `requirements-dev.txt`).** FastAPI's `TestClient` needs an HTTP client library. Starlette 1.8 deprecates `httpx` for this and emits a `StarletteDeprecationWarning` that asks for `httpx2`. `requirements-dev.txt` pins `httpx2` alongside `pytest` and `ruff`.
3. **`npm test` runs `vitest run --passWithNoTests` until Phase 3.** The frontend has no pure helpers to test yet, and Vitest exits with an error when it finds no test files, which would fail CI. **Phase 3 removes the flag** once the first Vitest tests exist.
4. **CI also runs the backend on `windows-latest` (§3, addition).** The spec asks for Python 3.12 and 3.14. The backend job runs both versions on Ubuntu *and* Windows, because the project is Windows-first (§0, rule 6). The frontend job runs on Ubuntu only.

### Phase 1 (approved 2026-10-16)
1. **Extra input checks in `run_backtest` (§6.1, addition).** Besides the §6.1 list, the backtester rejects:
   - a signal outside {−1, 0, +1};
   - weights that are non-finite, or non-zero on a day the signal is flat;
   - non-finite returns or rf (§6.1 only rules out NaN).

   Trades are identified from the executed position (§7), but P&L comes from the executed weights. A weight held on a flat day would earn P&L that no trade owns and would silently break the §7.2 reconciliation. The §6.1 snippet's `assert` is implemented as `raise ValueError`, because Python drops `assert` statements under `-O`.
2. **`resolve_split` computes ⌊0.7·n⌋ in integer arithmetic (§6.3).** `7 * n // 10` is the exact floor. The float expression `int(0.7 * n)` is one too small for n = 90, 170, 180, 650 and 37,443 values of n up to 2,000,000, because 0.7 has no exact binary representation: 0.7 × 650 evaluates to 454.99999999999994. The §12 counts (880 evaluated days → 616 IS) are the same either way.
3. **The Monte Carlo core requires a seed (§10.6).** `monte_carlo.simulate` takes an explicit seed, so the core stays deterministic (§0, rule 8). When a request has no seed, the API layer (Phase 2) draws one with `secrets.randbelow(2**53)` and returns it. `mu_daily` in the result is the drift that GBM and the closed form actually used, so it is 0 with `zero_drift`.
4. **`static_alpha_is` is fitted on raw log prices (§5.4; the spec doesn't say).** The diagnostics regress ln Y on ln X without rebasing, so α̂ reads directly as ln Y = α + β ln X. Rebasing would change α̂ only; β̂, both Engle–Granger p-values and the half-life stay the same.
5. **Warning texts add context (§8).** Each warning keeps the spec's wording and adds the segment, a count or a date. For example: "Out-of-sample: 3 closed trades (fewer than 10), so trade statistics are unreliable."
6. **`run_strategy` and `build_report` are separate functions (§5.0, §9).**
   - `run_strategy` runs §5.0 steps 1–6 and is still the only code that does.
   - `build_report` computes the metrics, benchmark, sizing summary, pairs diagnostics, warnings and chart series from its result.
   - The sensitivity grid calls only `run_strategy`, so it doesn't run two Engle–Granger tests for each of up to 64 configurations.
7. **Two more `null` cases (§2.4).** The Engle–Granger p-value is `null` when a leg is constant over the test window, because statsmodels raises an error on constant input. CAGR is `null` if (E_n / E₀)^(252/n) overflows a float.
8. **The one-day-segment ruin test uses the split date (§12, `test_backtester.py` item 9).** The spec's original example, ruin on the first evaluated day, can't happen: no position is held into that day, because the first position is entered at its close. The test instead puts the ruin day on the split date. That leaves a one-day OOS segment, and the test checks the same `null` fields. CLAUDE.md §12 now says this.
9. **The Pydantic part of that test waits for Phase 2.** "The full response still validates against the Pydantic response model" needs `schemas.py`, which Phase 2 adds; the assertion is added then.
10. **pytest treats every warning as an error (`pyproject.toml`, addition).** The config sets `filterwarnings = ["error"]`, so any warning fails the test that raised it, locally and in CI. In numerical code a `RuntimeWarning` usually signals a silent NaN or inf.
11. **The Phase 1 `conftest.py` has synthetic data only (§12).** It provides the 1,000-day market (a cointegrated Y/X pair and SPY, from a fixed seed) and rf from a constant 2% ^IRX. The fake network, the frozen clock, the yfinance-layout frames and the temp-file database need the loader and the API, so they arrive in Phase 2.

### Phase 2 (approved 2026-10-16)
1. **Yahoo errors are read from yfinance's log (§4.3).** In yfinance 1.7, `yf.download` never raises. It catches every failure, logs it on the `yfinance` logger as `['TICKER']: <error>`, and returns an empty frame. (Verified with a stubbed `Ticker` driving the real `yf.download`.)
   - `fetch_from_yahoo` keeps the spec's exact call, collects that log line for its own ticker, and re-raises.
     - A rate limit becomes `YFRateLimitError` (503).
     - Any other exception, including a timeout, becomes `DownloadError` (502).
   - An unknown or delisted ticker is logged as a plain "possibly delisted" message, not an exception, so it stays an empty frame (404). A real request for an unknown ticker confirmed this.

   CLAUDE.md §4.3 now says so.
2. **Cleaning is split between storage and read time (§4.4).**
   - Before rows are stored, those missing `close` or `adj_close` are dropped and duplicate dates are removed. Those columns are NOT NULL, and (ticker, date) is the key.
   - Every time prices are read, the other checks run:
     - non-positive prices are dropped with a warning;
     - days above 50% and runs of 5 or more identical closes produce warnings;
     - out-of-range ^IRX prints are dropped.

   So the warnings appear on cache hits too, not only on the request that downloaded the data.
3. **`open`, `high` and `low` are nullable (§4.1).** The spec lists them as Float without saying NOT NULL. Yahoo occasionally has a valid close with a missing open, high or low, and the engine uses only `close` and `adj_close`.
4. **A failed or empty download stores nothing (§4.2).** An empty answer raises 404 before anything is written, so neither the rows nor the coverage change. Otherwise one transient empty answer from Yahoo would be recorded as coverage and never refetched.
5. **Validation errors are one string (§11.1).** FastAPI's default 422 body is a list of error objects. The API returns `{"detail": "field: message; …"}` instead, the same `{"detail": "..."}` shape as every other error, so the frontend handles one format.
6. **Request bodies are strict (§11.2, addition).** Unknown fields return 422, which catches typos such as `z_entri`. NaN or Infinity in a request body is rejected (`allow_inf_nan=False`).
7. **Fields that don't apply are dropped before validation (§11.2).** A "before" validator removes them from the input, so none of their rules run. Even a nonsense value, such as `z_entry: 99` for trend, is accepted and then echoed as `null` in `resolved`.
8. **Only successful backtests are stored (§4.1).** `backtest_runs` gets a row when `/api/backtest` returns 200. A failed request has no metrics to store.
9. **Two small wording choices (§11.3).**
   - `rf_source` reads "13-week T-bill (^IRX)" or "constant 3.00% a year".
   - A ticker-source Monte Carlo run returns no data-cleaning warnings, because `MonteCarloResponse` has no warnings field.

### Phase 3 (approved 2026-10-16)
1. **A Run sends the backtest first, then the other two requests (§13).** The Run button is the only thing that sends requests. It posts `/api/backtest`, and only if that succeeds does it post `/api/sensitivity` and `/api/monte-carlo`, in parallel.
   - Three simultaneous requests on a cold cache would each download the same tickers from Yahoo. §4.2 makes that safe, but it triples the downloads and the rate-limit risk.
   - A request the backtest rejects would fail the same way at the other two endpoints, so they are skipped, and their panels say so.
   - A new Run aborts any request still in flight, so a slow old response can't overwrite a newer one.
2. **The dashboard's Monte Carlo projects the strategy over the full evaluation window (§13).** The sidebar has only the §13 settings (paths, horizon, block length, seed, zero drift), so requests use `source: "strategy"` with the default `window: "full"`. The KPI grid shows VaR and CVaR next to the strategy's metrics, so the projection has to be of the strategy. `source: "ticker"` and `window: "is"` remain available through the API.
3. **Two API options have no control (§13).** The sensitivity grid overrides (`L_values`, `z_entry_values`) and `initial_capital` are not in the §13 sidebar list, so the dashboard never sends them: the default grid (§9) and $100,000 apply. Both remain available through the API.
4. **The KPI grid shows a few more figures (§13, view 1, addition).**
   - Besides the listed metrics: total return, annualized vol, average holding days, cost drag, and correlation with SPY.
   - The pairs diagnostics also show the static β̂ and α̂.
   - "VaR95 and CVaR95 from the latest Monte Carlo run" are shown for both models, GBM and bootstrap.
5. **The vol forecast has its own chart (§13, view 3).** It sits under the gross-exposure chart, with the target as a dashed line, and the exposure chart marks the leverage cap. A second y-axis on the exposure chart would narrow its plot area, and the synchronized charts would no longer line up date for date.
6. **The sidebar checks only that inputs are well formed (§13).**
   - A blank or non-numeric field stops the Run with a message naming the field. `JSON.stringify(NaN)` is `null`, and a null `L` would silently mean "use the default".
   - The seed must be a whole number below 2^53: a larger integer is rounded before it leaves the browser (§10.6).
   - Every range rule is left to the API, whose 422 `detail` names the rule, so the rules live in one place.
7. **Two marker edge cases in the signal panel (§13, view 3).**
   - An exit forced by a NaN score (a degenerate window means flat) has no score to sit on, so its marker is drawn at 0.
   - A marker whose signal date is not in the series is counted and reported under the chart instead of being dropped. The s0 rule (§11.3) means this never happens.
8. **Each view has its own error boundary (addition).** A rendering error in one panel shows an error box in that panel instead of blanking the dashboard. The boundaries reset on every new run.
9. **The production build puts the libraries in their own chunks (`vite.config.ts`, addition).** The single bundle was 657 kB, over Vite's 500 kB advisory limit, so every `npm run build` printed a warning. Rolldown's `codeSplitting` now puts React (219 kB) and Recharts with its dependencies (387 kB) in separate chunks next to the 50 kB app, rather than the limit being raised.
10. **Screenshots are in `docs/screenshots/` (§16).** There are 11 PNGs (1.5 MB) of the dashboard running against the local API:
    - KO/PEP pairs, 2015–2025;
    - SPY trend, 2006–2025, with a 10% vol target;
    - an unknown ticker (404);
    - a trend history too short for the sensitivity grid (422).

    They were captured by driving headless Edge over the Chrome DevTools Protocol. That script is not in the repo.

As Phase 0, item 3 planned, `npm test` is now plain `vitest run`.

### Phase 4 (approved 2026-10-16)
1. **The diagnostics cache is a module of its own, `app/diagnostics_cache.py` (§3, §14, addition).** The bench missed the 300 ms target, so §14's cache was built: a warm 15-year KO/PEP backtest took 361 ms, 184 ms of it in the two Engle–Granger tests.
   - The cache sits in the API layer, in its own module so it can be tested directly.
   - `pipeline.build_report` gained a keyword argument, `pairs_diagnostics`, which defaults to `stats.pairs_diagnostics`. The `/api/backtest` route passes the cache's lookup instead.
   - Every other caller is unchanged, and the pipeline stays pure.
2. **The cache key also hashes the dates (§14, addition).** The spec keys on the tickers, the IS and OOS windows, and a SHA-256 of the two adjusted-close arrays. The digest also covers the calendar dates. That pins how many days fall on each side of the split, even if the same prices ever sat on different dates.
3. **CLAUDE.md §14 now says which changes miss the cache.** It said the diagnostics "don't depend on L, z or costs". Under the default 70% split, though, L, `execution_lag` and, with the overlay on, `vol_halflife` move t₀, and the split date moves with it. Changing them therefore misses the cache unless `split_date` is set.
   - Measured with the bench: an L sweep under the default split got 0 hits in 10 requests.
   - With `split_date` fixed, the same sweep got 10 hits in 10.
4. **The first-request target is still missed (§14).**
   - A first request on new data takes 361 ms against the 300 ms target. As §14 anticipates, the cache can't help it.
   - A request that changes only z_entry takes 150 ms.
   - The README reports both numbers. No optimization beyond the spec was attempted.
5. **Bench design (§14 leaves these open).**
   - The bench uses its own SQLite cache, `backend/bench.db` (git-ignored), so its runs never enter the app's database or run history. Its first run downloads KO, PEP, SPY and ^IRX. After that, a guard stops the bench if any timed request reaches the network.
   - The 15-year window is KO/PEP from 2011-01-01 to 2025-12-31: 3,772 trading days.
   - Requests go in process through FastAPI's TestClient. Each figure is the median of 10 runs after one warm-up run, and `--repeats` changes the count. Parsing the response JSON on the client side is outside the timed region.
   - The output is plain ASCII. Redirecting it to a file in Windows PowerShell 5.1 would otherwise garble characters such as "§".
6. **The bench reports more than §14 asks (addition).** Besides the targets, the stage breakdown and the repeats with z_entry or L changed, it reports:
   - an L sweep with `split_date` fixed;
   - the Monte Carlo time per model;
   - the Monte Carlo peak memory at the API caps, measured with tracemalloc: 216 MB, 2.13 times the 102 MB path array. §14 says "about 200 MB".
7. **Ruff ignores E402 in `backend/scripts/` (`pyproject.toml`).** Scripts run as `python scripts\bench.py`, so they have to put `backend\` on `sys.path` before importing the app, and those imports can't sit at the top of the file.
8. **Tests beyond §12 for the cache (addition).**
   - `tests/test_diagnostics_cache.py` checks that:
     - a hit returns what a fresh computation gives;
     - any change to the inputs misses;
     - the least recently used entry is evicted first;
     - the default size is 256 entries.
   - An API test checks that:
     - z_entry, z_exit, costs and the hedge mode reuse the cached result;
     - a new L under the default split does not;
     - cached and fresh responses are identical apart from the run id.
   - The `client` fixture empties the cache before each test, so tests stay independent.
   - Each of five deliberate bugs planted in the cache made at least one of these tests fail.
9. **METHODOLOGY covers Phases 0–4 (§15).**
   - The Phase 5–9 justifications and limitations that §15 lists come with those phases.
   - The figures marked *(engine)* were measured on the app's own price cache, `backend/quant.db`, read-only and without storing any runs. Each comes with the request or script that reproduces it.
   - The trend and vol-targeting tables reproduce the review prototype's figures in §5.5 and §5.6 to within rounding.

### After Phase 4 (approved 2026-10-16)
1. **The benchmark can be hidden (addition to §13, views 1–2).**
   - A "Buy & hold {ticker}" toggle sits in the header of the Performance panel and of the Equity and drawdown panel.
   - Both buttons share one state. It starts on, and it is kept across runs.
   - Turning it off hides the benchmark line and legend entry in the equity chart. It also hides the grey "B&H" figures under the KPI tiles, including realized vol in the vol-targeting group.
   - When the line is hidden, the equity axis rescales to the strategy alone. That makes a low-volatility strategy readable next to a fast-rising benchmark.
   - It only changes what is shown. The request, the response and `benchmark_metrics` are unchanged.
   - It was requested by the user and checked in the browser on a KO/PEP pairs run.

### Phase 5 (approved 2026-10-17)
1. **The DSR calibration test runs a vectorized copy of the mean-reversion grid (§12, `test_sensitivity.py`).**
   - Running `run_sensitivity` on 200 seeds takes about 70 s (0.36 s a seed in the prototype), more than the whole suite's 60 s budget.
   - The test instead builds the default 20-cell grid for all 200 seeds at once, with run_strategy's arithmetic: the rolling z-score, the §5.1 rule, a one-day lag, rf = 0 and no costs. It feeds the result to the same `deflated_sharpe` function the API uses.
   - It checks the copy against `run_sensitivity` on seed 0: the deflated results compare equal. In the prototype, the copy matched the engine's window returns exactly on 5 seeds × 20 configurations.
   - It takes 1.5 s.
2. **"At the break-even cost the Sharpe is 0 or null" is tested as |Sharpe| < 1e-9 or null (§12, item 13).** The mean excess at c* is 0 only up to rounding (the test checks it is below 1e-15), so the Sharpe ratio comes out near 1e-15, not exactly 0.
3. **The long-only test also covers the vol overlay (§12, item 11, addition).**
   - With unit weights the long trades match exactly, dates and P&L, as specified.
   - With sized weights, a direct flip's cost is split by the fraction a/(a + b), which is exact only up to rounding. The overlay case checks the dates exactly and the P&L within 1e-15.
4. **The KPI grid gains a "Costs" group (§13, view 1).**
   - Turnover and cost drag move there from the strategy group, next to the new borrow drag and break-even cost.
   - The break-even tile turns amber below twice the run's cost in every segment. The §8 warning applies to the full period only.
5. **The cost sweep's axis stretches to a break-even up to 4× the grid's top cost (§13, view 8, addition).** A break-even further out is stated in words instead, so the 0–100 bps sweep isn't squashed into one corner.
6. **The benchmark toggle also hides the calendar's buy-and-hold column.** This extends "After Phase 4", item 1.
7. **Page order (§13).** The new views sit next to the views they relate to: rolling statistics and calendar returns after equity and drawdown, and the cost sweep before the sensitivity grid.

### Phase 6 (approved 2026-10-17)
1. **`fixed_metrics` can be `null` (§11.3).** It is `null` when the request's own configuration was ruined before the walk-forward's first day, by the same rule as §6.2 for a segment that starts after ruin; `fixed_equity` is then all `null`. Tested. CLAUDE.md §11.3 now says so.
2. **The switching cost follows §6.1 (§9.2 wording).** §9.2 said a switch is charged |w^(j)_{b−1} − w^(j−1)_{b−1}|. Applying §6.1 to the stitched weights, which §9.2 also requires, charges |w^(j)_{b−1} − w^(j−1)_{b−2}|: the trade from the position actually held into that close. The two differ only when the outgoing configuration itself trades at that close. CLAUDE.md §9.2 now gives the §6.1 form.
3. **The last fold runs through the last close (§9.2).** The span [b_j − 1, b_{j+1} − 1) has no b_{J+1} for the last fold; it runs to the end of the stitched window. The one-configuration identity test (§12) requires this. CLAUDE.md §9.2 now says so.
4. **Invested days of the stitched series.** It has no trade ids, so a day counts as invested when a position was held into it (gross exposure > 0). This drives `pct_days_in_market`, the Sharpe's no-exposure rule and the folds' test Sharpe ratios. For a single configuration it is the same as `held_id` not null.
5. **Only the four trade fields are `null` for the stitched series (§9.2).** Closed and open trades, win rate and holding days are `null`; turnover, cost drag and borrow drag are reported, because the walk-forward pays them.
6. **The fixed run's trades.** `fixed_metrics` counts the trades entered in the window. A trade still held after the window's last day counts as open; this matters only when the walk-forward is ruined before the end. Tested.
7. **Walk-forward warnings (§8).**
   - The response also carries the data-cleaning warnings (§4.4) and the stitched series' ruin warning (§6.2).
   - Fold numbers in the warnings are the 0-based indexes of the fold table and `series.fold`.
   - "Changed in more than half the folds" counts the folds whose choice differs from the previous fold's, a flat fold counting as a choice, and warns when that count exceeds half the folds.
   - Folds that start after a stitched ruin stay in the fold table, with a `null` test Sharpe.
8. **Dashboard (§13, view 11).**
   - The walk-forward settings are checked only by Run walk-forward, which sends them, so a blank one never blocks Run.
   - A result survives later Runs and is marked "Settings changed since this ran" when the sidebar would build a different request.
   - The buy & hold toggle also hides the walk-forward chart's benchmark line and its metrics column, extending "After Phase 4", item 1. The README screenshot shows that view, so the axis fits the two strategies.
   - Additions: a summary row (out-of-sample days, folds and changes of choice, distinct configurations, both Sharpe ratios), ↻ marks on changed choices, test Sharpe ratios coloured by sign, each day's fold in the tooltip, and the trade rows in the metrics table ("—" for the walk-forward).
   - The panel sits after the sensitivity grid, whose grid it reruns, and before Monte Carlo.
9. **Tests and bench beyond §12 and §14.**
   - Tests of the selection rules on crafted inputs (ties, the ruin-day boundary, exposure), two ruin scenarios, the outputs on a window shorter than the calendar, and the warnings.
   - The perturbation test's walk-forward cases check the stitched weights and total return as well as the excess.
   - The bench also times `run_walk_forward` alone and reports the folds and distinct choices.
   - Planted bugs: 18 of 18 in the backend caught by tests; 12 of 13 in the frontend helpers, the 13th caught by the TypeScript check in `npm run build`.

### Phase 7 (approved 2026-10-19)
Items 1–4 were approved while the phase was being built, on 2026-10-19.
1. **The Kalman standardization test holds the state constant over the start-up window, with L = 2,000 (§12, item 9).** As written, the test simulated a state that is a random walk from day 0, and it failed on nearly every seed. The §5.7 start estimates R from an OLS fit that assumes a fixed state, so drift in those days overstates R and shrinks z (§27). The test now holds the simulated state constant over days 0…L−1, which is what the start assumes. It uses L = 2,000, outside the API's 10–252 range, so the sampling error of the estimated R stays well inside the bounds. The 0.05 bounds are unchanged, and the test runs for δ = 1e-4 and 1e-5. All 200 prototype seeds passed with this setup. CLAUDE.md §12 now says so.
2. **The default `kalman_delta` is 1e-6, not 1e-4 (§5.7, §11.2).** At 1e-4 the std of z on KO/PEP over 2015–2025 was 0.35, and the model made 4 trades; on the test pair it made none. At 1e-6 the std is 1.13 on KO/PEP and 1.14 on the test pair (§27). The valid range, 1e-7 to 1e-2, is unchanged. CLAUDE.md §5.7 and §11.2 now give 1e-6.
3. **A ruin-day credit is split across the legs by gross weight (§6.2, §7.4).** The spec credits the clipped amount to "the trade held into r", and a trend portfolio can hold several legs into r. Each leg's trade gets the share |w_{i,r−1}| / Σ_j |w_{j,r−1}|, so the reconciliation stays exact (§28). Tested. CLAUDE.md §7.4 now says so.
4. **pytest-xdist runs the suite in parallel (§3, §12).** One process took 71.6 s, against the 60 s budget. `requirements-dev.txt` pins `pytest-xdist==3.8.0`, and `pyproject.toml` adds `-n auto` to pytest's options, so `python -m pytest` is still the command. With one worker per core (16 here) the suite takes about 30 s; `-n 0` runs it in one process. CLAUDE.md §3 and §12 now list it.
5. **`run_backtest` has one argument with a default, `leg_signals=None` (§6.1).** §6.1 says no input has a default. The trend portfolio passes each leg's signal, and the backtester then identifies and attributes trades leg by leg (§7.4). It also checks that `s` is the invested flag and that each leg's weight is exactly 0 where its own signal is flat. With the default, every other call site, in the pipeline and the tests, is unchanged. A portfolio that forgot it would number its trades on the invested flag; `_run_trend_portfolio` is the only caller that passes it, and its tests cover it.
6. **The per-leg bookkeeping is written with NumPy (§7.1, §7.3).** For the trend portfolio, the §7.1 attribution and the §7.3 bookkeeping run on arrays, not pandas group-bys, because they run once per leg. That brought a 4-leg backtest from 69 ms to 40 ms. Single-trade strategies keep the pandas code of §7.3, so their outputs are unchanged bit for bit. A test checks that each leg's trades match that leg backtested alone through the pandas code: the same dates, and P&L within 1e-15.
7. **Per-leg outputs where the spec is silent (§5.8, §7.4, §11.3).**
   - `BacktestResult` gains `held`: true on the days a position is held into, by any leg. It drives `pct_days_in_market`, the Sharpe ratio's no-exposure rule, the rolling statistics, walk-forward eligibility and the Monte Carlo exposure check, for every strategy. For single-trade strategies it equals `held_id` not null.
   - For the trend portfolio, `trade_id` and `held_id` are frames with one column per leg, holding the portfolio-wide ids, and `leg_pos` holds each leg's executed position.
   - `series.position` is the executed invested flag, and `series.gross_exposure` sums the legs.
   - In `series.legs`, `score` and `vol_forecast` are as of each close, like the single-asset `series.score` and `series.vol_forecast`; `position` and `weight` are executed. §11.3 said "executed values" for all four, and now says which are which.
   - `beta_at_entry` is `null` for leg trades.
   - The cap warning reads "Legs were at their leverage cap (max_leverage / N) on 76% of invested asset-days …", because the figure counts asset-days.
   - The sensitivity grid and the walk-forward work for the trend portfolio, with trend's L axis (§9); their benchmark is the basket.
8. **The Kalman model (§5.7).**
   - The start is computed in closed form with centred sums, rather than by a matrix solve. The guards run first.
   - `StrategyParams` requires L ≥ 3 for the Kalman model, because R divides by L − 2. The API's minimum is 10.
9. **Monte Carlo (§10.7, §10.8, §11.3).**
   - The automatic block lengths are computed on ℓ before `zero_drift` demeans it, which gives the same estimate without the subtraction's rounding (§30).
   - If the estimate is undefined (NaN), the request returns 422. No real series has produced one.
   - `resolved.block_len` echoes the request, a number or `"auto"`, and `block_len_used` gives the lengths actually used: the request's number for both bootstraps, or the two estimates.
   - The ⌊n_hist/2⌋ bound of §10.8 never binds, because n_hist ≥ 252 makes it at least 126, above the cap of 63.
   - CLAUDE.md §10.7 said the stationary bootstrap's results are "less sensitive to the exact block length". On SPY they were more sensitive (§29), and Lahiri (1999) finds the stationary bootstrap less accurate. CLAUDE.md §10.7 now says what the random lengths do achieve, a stationary resampled series, and notes this finding.
10. **API (§11.2).** The vol target's range depends on the strategy (1–50%, or 5–60% per asset for the trend portfolio, where `null` becomes 40%), so it is checked in the model validator instead of as a field constraint. Its 422 message names the range for the strategy.
11. **Dashboard (§13).**
    - The strategy switch uses short labels, "Mean rev.", "Pairs", "Trend" and "Portfolio", with the full names as tooltips, so four fit across the sidebar.
    - The portfolio's tickers are one text field, separated by spaces or commas, with a count ("4 of 2–12"). The count and uniqueness are left to the API's 422, like every range (Phase 3, item 6). The presets are "Trend ETFs" (SPY, QQQ, GLD, TLT) and "Diversified".
    - Pairs get a "Pairs model" switch and a "State noise δ" field. Under the Kalman model the hedge-mode switch calls its daily re-hedging "Filtered" rather than "Rolling".
    - Monte Carlo has an "auto" checkbox beside the block length. The table gains a stationary-bootstrap column and a "Block length used" row, and the summary gives the lengths used, such as "auto: 14 days circular, mean 12 days stationary".
    - The benchmark is labelled "Equal-weight basket" for the portfolio ("EW basket" in column headers, "EW" in tiles), and the buy & hold toggle shows and hides it the same way.
    - The KPI grid's vol group shows "Target per asset" with each leg's cap ("each leg ≤ 0.20× of 2.0×"), and "Days at cap" counts invested asset-days. Its strategy group notes that the trade figures count every leg's trades, and its Monte Carlo group shows VaR95 and CVaR95 for all three models, extending Phase 3, item 4.
    - The trades table gets a "Leg" column after the id, and the assumptions footer names the pairs model and the portfolio's sizing.
12. **Tests, bench and checks beyond §12 and §14.**
    - Extra tests: each leg's trades equal that leg backtested alone (item 6); the ruin-credit split; each leg resizing on refresh days and on its own signal changes; the basket benchmark; the portfolio's input checks; the Kalman model feeding the unchanged pairs pipeline; a nearly flat (not exactly flat) X window; automatic block lengths rounding up and clipping, with a stubbed estimator; and the third model leaving GBM and the block bootstrap unchanged, at a block length of 10 and at "auto".
    - The portfolio vol-tracking test (§12, item 10) calls the sizing and the backtester directly, because a trend signal can't be held constant. It uses zero-drift assets, a 20% target per asset, a 4× cap that never binds, and 20 seeds.
    - The bench adds `run_strategy` with rolling OLS and with the Kalman filter, the 10-ETF portfolio, each model's time inside a Monte Carlo request, a request with `block_len: "auto"`, and the memory peak with three models.
    - Planted bugs: 38 of 38 in the backend and 20 of 20 in the frontend helpers were caught. The first backend pass missed two, a long-only per-leg case and automatic block lengths that round instead of rounding up; a test was added for each, and both are caught now.
    - Eleven earlier API responses on real data (6 backtests, 3 grids and 2 Monte Carlo runs), from the Phase 6 code and from this code, are identical apart from the new fields and the Monte Carlo histogram, whose bins now pool three models (§18).
    - Screenshots: every dashboard view was captured again on 2026-10-19, and five views were added (README). Seven files changed, through the new labels and the Monte Carlo panel's third model; the rest came out byte-identical. The two error views are still the Phase 3 captures.

### Phase 8 (approved 2026-10-19)
1. **The offline check runs in `_download`, the network seam's only caller, not inside `fetch_from_yahoo` (§4.3).** §4.3 puts the check inside `fetch_from_yahoo`, but §12 asks the test to prove it with the fake network's call counter, and the fake replaces `fetch_from_yahoo` whole: a check inside it would never run in the tests. In `_download` it runs before the seam is called, so the counter stays at zero (tested). `yfinance` is still imported when the module loads, as before; importing it makes no network call. `QUANT_OFFLINE` is read on every call, so tests set it with `monkeypatch.setenv`; only the value `1` turns it on.
2. **`seed_e2e.py` also works with `QUANT_OFFLINE=1` set.** It writes through `ensure_cached` with the seam replaced, and switches offline mode off for itself, because `playwright.config.ts` sets `QUANT_OFFLINE=1` for the whole server command, seed included. It restores both afterwards. SYNT is the 15%-vol asset of the trend-portfolio universe with a fixed drift of 0.03% a day; ^IRX is seeded from 2011-12-01, for the 14-day buffer.
3. **Stored runs are read through nullable copies of the response models (§11.4).** `StoredMetrics` and `StoredResolvedRun` have the same fields as `MetricsOut` and `ResolvedBacktestRun`, each nullable with a default of `null`, generated from them so the two can't drift apart. In `types.ts`, `StoredRun` is `RunDetail` with every stored field nullable, and the dashboard reads runs as that. `created_at` is written as ISO 8601 with a `+00:00` offset: SQLite returns the column without a time zone, and it was written in UTC. An unknown id's 404 comes from FastAPI's `HTTPException`, which already has the `{"detail": …}` shape.
4. **"Load settings" (§13 view 12).**
   - It loads the run's resolved split date, so a re-run evaluates the same windows (§31).
   - A field an older engine didn't store loads as the default it was introduced with: borrow fee 0, shorting allowed, rolling OLS.
   - Settings a run doesn't store, Monte Carlo and walk-forward, keep their current values, as do the hidden controls of other strategies.
   - A notice says which run was loaded and that Run re-runs it.
5. **The comparison table (§13 view 12)** shows the settings that tell runs apart (strategy, tickers, period, split, L, z, pairs model, vol target, cost and engine), then the chosen segment's strategy metrics, then five benchmark metrics. The history table also shows when each run was stored, in the viewer's time zone, and the history lists the 50 most recent runs.
6. **CSV files (§13 view 13), where the spec is silent.**
   - In file names the tickers are joined by "_", because tickers can contain "-" (BRK-B): `pairs-KO_PEP-2015-01-02-2025-12-31-trades.csv`.
   - The trades CSV has a `ticker` column for every strategy, empty except for the trend portfolio, and is in id order whatever the table's sort.
   - The series CSV leaves out the arrays the API sends as `null` for the strategy, and adds `{ticker}_score`, `_position`, `_weight` and `_vol_forecast` for each trend-portfolio leg.
   - The calendar CSV holds the monthly rows, then the annual ones, with a `period` column (`month` or `year`) and an empty month on annual rows.
   - The three buttons sit in one "Export CSV" bar under the run header.
7. **The log axis (§13 view 2)** extends to the nearest 1, 2 or 5 × 10^k beyond the data, so it always has at least two ticks, and keeps only powers of ten past nine ticks (§31). It spans the benchmark only while the benchmark is shown, and the Linear | Log choice is kept across runs.
8. **URL state (§13).** Parameters are the sidebar's own field names (`pairY`, `zEntry`, `volOn`, …), and booleans are `true` and `false`. The strategy is applied first, so L starts from its default. A ticker list is normalized to spaces. Notices for ignored parameters, and the "Load settings" notice, appear in one dismissible banner above the results.
9. **The browser tests' servers (§12).**
   - The API runs on port 8001 and `vite preview` on 4173, and neither is ever reused, so the tests never touch a developer's servers on 8000 and 5173 or their price cache.
   - The build is made for the tests into `dist-e2e/`, with `VITE_API_URL` pointing at port 8001 and `CORS_ORIGINS` allowing port 4173. That uses the two existing settings, where the spec's same-origin proxy would have needed a new one for the port. The build is still `tsc -b` followed by `vite build`, as in `npm run build`.
   - The database is a temporary file, recreated on every run. The tests run one at a time on it, because the history test reads what was stored.
   - The CI job seeds the database through the same `webServer` command as a local run, not in a step of its own, and uploads the report when a test fails.
   - `npm run e2e` is an alias for `npx playwright test`.
10. **Two more browser tests, beyond the eight of §12:** an invalid URL parameter is ignored with a notice naming it, and Run is not blocked by the browser's own form validation (item 11). The pairs test also runs the walk-forward and switches the equity axis to log, and the history test compares the loaded run.
11. **The sidebar's form no longer lets the browser block Run (a bug from Phases 3–7).** The number fields' `min`, `max` and `step` were meant only for the spinner arrows, as their code comment said, but the browser enforced them on submit: Run silently did nothing for valid values off a field's step, such as a 2.5 bps cost, a 10 bps borrow fee or 2,500 paths, and an out-of-range value never reached the API's 422. The form now has `noValidate`. The URL round-trip browser test found it, and the new test covers both cases.
12. **`npm run screenshots` (§12)** runs `scripts/screenshots.ts` with Node's built-in TypeScript support, against the dev servers on 8000 and 5173, and sets each scenario up through the URL. It writes every file in `docs/screenshots`, including the two error views, which until now were the Phase 3 captures, and two new ones: the equity chart on the log scale and the run-history comparison.
13. **Tests and checks beyond §12.**
    - Backend: the history's summary fields are read from the stored metrics; `limit` outside 1–500 returns 422; offline mode lets a cache hit through, and `QUANT_OFFLINE=0` stays online; the seed script writes a database on which the offline API serves the browser tests' requests and answers 503 past its end.
    - Frontend: tests for CSV quoting, line endings and precision, each CSV table, URL parsing and round trips (including that the rebuilt requests are identical), "Load settings" for each strategy and for an old run, the comparison rows, the log axis and the GET wrappers.
    - Planted bugs: 17 of 17 in the backend and 37 of 37 in the frontend helpers were caught. The first frontend pass missed two. One was an equivalent mutant (it set the strategy twice to the same value) and was replaced by a real bug, which is caught. The other, ticks with floating-point noise, appears only at 5e-6 and 5e-11; a test at that range now catches it.

### Phase 9 (approved 2026-10-20)
1. **The scenarios' fixed dates and settings (§13), where the spec is silent.** Every scenario runs 2015-01-01 to 2025-12-31, whole calendar years, so a re-recording on a later day sends the same requests. Every other setting is the dashboard's default: L 60 or 252, z 2.0 / 0.5, 10 bps, a one-day lag, the T-bill rate, the 10% vol target with the trend overlay on, 40% per asset for the portfolio; the Monte Carlo has 10,000 paths, 252 days, blocks of 10 and seed 42; the walk-forward 756 / 63 days, rolling.
2. **The recording runs on a temporary copy of the price cache (§13, "Recording").** `record_demo.py` first caches the scenarios' prices in the real cache, downloading what's missing as the API would (on 2026-10-20 that was XOM, CVX, GOOGL and GOOG; `QUANT_OFFLINE=1` forbids it). It then sends the requests to a copy with the run history emptied, so it adds no runs to the app's history and the recorded run ids are always 1 to 8. The demo's run header shows "recorded" in place of the id. The response bodies are written byte for byte, and the script removes the output folder first, so a dropped scenario leaves no file behind. The 10 MB limit is checked as 10,000,000 bytes. A second recording, made offline on the same cache, reproduced all 33 files byte for byte except the Monte Carlo responses' `elapsed_ms` timings.
3. **The manifest (§13, "Replay").** It has one table per endpoint, because the backtest and the sensitivity grid send the same body, and it also holds the recording date, the last data date, the engine version and each scenario's four requests. The dashboard parses each key and re-serializes it with sorted keys before comparing, because Python writes 2.0 where JavaScript writes 2 (§33).
4. **The scenario list is defined once, in `record_demo.py`.** The demo's sidebar reads the scenarios from the manifest and fills the form from their requests; `demo.test.ts` checks, for every scenario, that the form then builds exactly the recorded requests. Recorded numbers keep the default's text where they are equal (2.0, not 2), so a scenario's URL holds only real differences.
5. **The demo shows a scenario at once.** It replays the first scenario, or the URL's, on load, and choosing a scenario in the picker replays it; live, a URL sends nothing on load (§31). A replay is one static file, with no computation or download to avoid (§33).
6. **URL state works in the demo,** with the first scenario as the defaults, so its URL is the bare page and every other scenario's URL holds its differences. A URL that isn't a recorded scenario fills the sidebar as usual; the picker then reads "Not a recorded scenario", and the replay shows the spec's message. That message has no HTTP status, since no request was sent.
7. **The build (§13, "Build").** `npm run build:demo` runs `vite build --mode demo`, the same build as `VITE_DEMO=1 npm run build`, because PowerShell can't set a variable for one command that way. The base path is `/Quantative-Backtesting-Engine/`, the repository's current name (the spec's example is `/quant-engine/`), and `DEMO_BASE` overrides it; `pages.yml` sets it from the repository's name. `npm run preview:demo` serves the build under that path.
8. **Live builds leave the recorded data out.** Vite copies all of `public/` into every build, so a small plugin removes `demo/` from the output of every build but the demo's. `VITE_DEMO` is a compile-time constant, so the replay code is compiled out of the live bundle too (checked: the live bundle doesn't mention the manifest).
9. **`pages.yml`.** The build job runs on every push to `main` and on manual dispatch, and uploads the Pages artifact. The deploy job has `if: false` until the data-licence decision; turning it on also needs Pages set to deploy from GitHub Actions in the repository's settings.
10. **The banner adds a "Source and setup" link** to the repository's README after the spec's text, so a visitor can find how to run the project locally.
11. **`resolvedToForm` accepts a request as well as a stored run,** reading a missing field as it reads a null, so the demo fills the sidebar with the same code as "Load settings".
12. **Browser tests of the demo (§12).** `playwright.config.ts` starts a third server: the demo build in `dist-demo/`, served by `vite preview --mode demo` on port 4174 under the GitHub Pages path. Four tests check, with no request outside the demo's own files and no console errors, that the first scenario replays on load with the banner, no run history and a read-only sidebar; that every scenario replays with its walk-forward; that a scenario's URL reopens it; and that an unrecorded configuration shows the demo's message. The phase's "local static server" check was also made by hand with Python's `http.server` serving a copy of the build.
13. **Tests and checks beyond §12.**
    - Vitest: canonical keys, the number-format normalization, every scenario's requests and lookups, that every named file exists and none is unnamed, the size limit, URL round trips of every scenario, matching, and the replay (a hit, a lookup by endpoint, a miss that fetches nothing, a missing file, and a retry after a failed manifest load).
    - Planted bugs: 16 of 16 were caught, 8 in the replay helpers by Vitest and 8 in the dashboard and the build by the demo browser tests (the replay without the base path, the recorded data dropped from the demo build, the scenario not shown on load or on picking, the history or the run id shown, an editable sidebar and the banner's wording).
    - Final checks: 329 pytest in 35.9 s (no backend test changed: the recorder is a script, and its output is checked by the Vitest tests above), 208 Vitest, 14 browser tests in 69.7 s with both builds, lint and both builds clean.
