# Methodology

Why the engine computes what it computes, what it can't tell you, and every approved departure from the specification, [CLAUDE.md](../CLAUDE.md) (§15). Section signs (§) refer to CLAUDE.md. This covers everything built through Phase 4; Phases 5–9 add their own sections as they land.

**Measured figures.** A figure marked *(engine)* was measured on 2026-10-16 with this code, through the same loader and pipeline as `/api/backtest`, on Yahoo Finance data cached that day. Each comes with the request that reproduces it; fields not shown are the defaults (10 bps one-way, k = 1, ^IRX). Timings come only from `scripts/bench.py` (see the README).

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
- **The perturbation test checks this** (§12). Multiplying every price after a date by random noise must leave every score, signal, weight, position and return up to that date bit-identical. It covers every strategy, both hedge modes, both trend frequencies, and the vol overlay off and on with each refresh schedule.

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
  - excess = gross excess − cost, and total = rf + excess.

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
- **Benchmarks are the tougher comparison.** They use the same evaluated window, rebased to E₀, fully invested (w = 1, no cash) and with no costs.

### 15. Trades (§7)
- **Trades come from the executed position.** A trade is a maximal run of non-zero, same-sign executed position `pos`. Using the unlagged signal would date every trade k days early.
- **Attribution follows P&L timing.** Day t's return belongs to the trade held into day t (`held_id`, yesterday's trade), and day t's cost to the trades being exited and entered. Masking returns by `pos != 0` would drop each trade's last day, which is the day its exit fills.
- **Splitting a day's cost.** Let a = Σ|w_prev| (exposure being closed) and b = Σ|w| (exposure being opened). The exiting trade gets the fraction a/(a + b) of the day's cost, and the entering trade the rest. This is exact in every case:
  - entry from flat: a = 0, so the new trade pays all of it;
  - exit to flat: b = 0, so the old trade pays all of it;
  - direct flip: every leg's weight changes sign, so turnover = Σ|w − w_prev| = a + b. The split charges c·a to the closing trade and c·b to the opening one;
  - rebalancing inside a trade (`rolling` hedge mode, or vol resizing): both parts land on the same trade.
- **Reconciliation.** Each day's excess return is split between `held_id` and `trade_id`, and a fully flat day contributes exactly 0. So the trades' P&L, open and closed, sums to the total excess return over the window. The tests require agreement within 1e-9 on 144 random runs, covering both hedge modes, the vol overlay with in-trade resizing, and lags of 0 to 2 (§12).
- **Bookkeeping (§7.3).**
  - `exit_date` is the last day the trade is held into: the day its exit order fills. The last day it appears in `trade_id` is the day before.
  - A trade entered on the final day is open, with 0 holding days and a P&L equal to its entry cost.
  - `signal_date` and `exit_signal_date` are k trading days before entry and exit. That is where the dashboard draws the markers, on the score that triggered them.

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
  - `ann_turnover` and `ann_cost_drag` annualize the daily means.
- **Null thresholds.** A Sharpe ratio, Sortino ratio or beta needs n ≥ 2, a denominator above 1e-12, and some exposure.
  - A strategy that never trades has excess exactly 0 and a `null` Sharpe ratio, not 0 and not 7e15.
  - 1e-12 is far below any real daily standard deviation (a volatility of 1% a year is 6e-4 a day) and far above rounding noise (about 1e-17).
- **Warnings** flag what the numbers alone would hide:
  - an IS Engle–Granger p ≥ 0.05: the pair may not mean-revert at all;
  - fewer than 10 closed trades in a segment: a win rate on 3 trades means nothing;
  - a non-positive hedge ratio at entry: the "hedge" adds exposure instead of offsetting it;
  - the data problems of §4.4;
  - ruin;
  - a leverage cap that binds on more than a quarter of invested days.

### 17. Parameter sensitivity (§9)
- **One split.** The split is resolved once, for the request's own configuration, exactly as `/api/backtest` would. Every configuration then uses that date; otherwise each L would compute its own 70% point.
- **Each configuration keeps its own t₀.** Shorter lookbacks start signalling earlier. Forcing the grid's latest t₀ on them would break §6.1's "no signal before t₀" check.
- **A common scoring window,** [the latest first-evaluated day, the split). Comparing an L = 20 Sharpe ratio over seven years with an L = 120 one over six would mix the effect of the parameter with the effect of the period.
- **IS only.** OOS returns exist internally but are never scored or returned, so looking at the grid can't contaminate the OOS segment.
- **Enough history.** The common window must be at least 252 days. The grid can fail where a single backtest succeeds, so the error names the lookback that is too long.
- **Multiple testing.** The best of 20 configurations is biased upwards by the selection itself (see Limitations). The heatmap outlines the request's own cell, not the best one.

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
  - **Draw order is fixed:** GBM increments, GBM sample paths, bootstrap increments, then bootstrap sample paths. A seed reproduces every output, and a model added later (Phase 7) draws last, so it doesn't change the existing ones.
  - **Bands** are quantiles of the log paths, exponentiated. Quantiles commute with any increasing function, so this equals the quantiles of the equity multiples, apart from the linear interpolation between neighbouring paths. `overwrite_input=True` avoids copying the array.
  - **VaR and CVaR** are positive loss fractions of the terminal return: VaR_α = −Q_{1−α}, and CVaR_α = −mean(terminal | terminal ≤ Q_{1−α}), the average loss in the worst 1 − α of outcomes. Unlike VaR, CVaR is coherent (subadditive), and it measures the tail that VaR only marks the start of.
  - **The histogram's 50 bins are shared by both models,** with edges at the pooled 0.1% and 99.9% quantiles, so the two distributions compare bar by bar. Values beyond the edges go into the end bins, because `np.histogram` silently drops them, and those values are the tail the chart exists to show. Each model's counts sum to N.
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
- **Validation ranges** keep requests inside what the engine is tested on (for example, L from 10 to 252 for mean reversion and pairs, 21 to 504 for trend), and keep Monte Carlo inside its memory budget: n_paths × (horizon + 1) ≤ 12.7 million.

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

### 21. Dashboard conventions (§13)
- **Only the Run button sends requests,** and hidden controls are never sent, so what runs is exactly what the sidebar shows.
- **Every synchronized chart uses the same `series` rows.** Nulls render as gaps rather than being dropped, because dropping rows would shift the shared index and misalign the charts.
- **Trade markers sit at signal dates,** when the decision was made, not at fill dates, so they line up with the score that triggered them.
- **The heatmap's colour scale is centred on a Sharpe of 0,** so colour shows the sign. The request's own cell is outlined, not the best one.
- **The histogram's red line is at −VaR95:** the loss that the worst 5% of outcomes exceed.
- **An assumptions footer** restates the cost, the lag, the rf source, daily rebalancing and the vol settings beside the results they qualify.

## Limitations
- **Selection and survivorship bias.** The preset pairs and ETFs were chosen with hindsight: they exist today and are known to be related. A pair that stopped moving together, or a stock that was delisted, never enters the sample, which flatters every pairs result.
- **Execution.**
  - Fills happen at the close.
  - The cost is a fixed number of basis points per unit traded: there is no spread model, no market impact, and no limit on trade size relative to volume.
  - That is reasonable for small trades in liquid large caps, and wrong for illiquid names or large size.
- **No borrow cost for shorts and no financing spread for leverage.**
  - Both are financed exactly at the T-bill rate. Phase 5 adds a flat borrow fee.
  - There are no hard-to-borrow names, recalls or margin calls, and ruin has limited liability.
- **The T-bill rate.** ^IRX is based on the 13-week bill's discount rate, which is slightly below its investment yield (by about 0.1 percentage point at a 5% rate), so cash and short proceeds earn slightly too little. Daily accrual is simple (a/252), not compounded.
- **Daily rebalancing to target weights,** with no cost for drift (§6.1): between signal changes, the model's weights stay constant without trading.
- **Pairs exits use the rolling z** while the frozen-β spread is held (§5.3), so an exit can fire slightly early or late relative to the spread actually held.
- **Multiple testing.** The sensitivity grid tests 20 configurations (5 for trend), and a user may try many more by hand. The best cell's Sharpe ratio is biased upwards by the selection itself. The grid outlines the request's own cell, and Phase 5 adds the deflated Sharpe ratio.
- **Tuning while watching OOS results turns OOS into IS.** The engine keeps OOS out of its own selection. But a person who changes parameters until the OOS panel looks good has used OOS for selection. The honest OOS result is the one recorded before looking.
- **The Sharpe CI assumes iid returns** (Lo 2002). Autocorrelation, volatility clustering and fat tails make the true interval wider, especially for strategies that hold positions for weeks.
- **Trend following here trades one asset at a time.** The published results come from diversified portfolios of dozens of futures. A single asset's trend Sharpe ratio is much lower and noisier: 0.32 on SPY (§10 above). Phase 7 adds a multi-asset portfolio.
- **The vol forecast lags.** The EWMA reacts to a volatility spike only after it starts, so realized vol overshoots the target in crashes: the weekly overlay's yearly range above still reaches 14.0% against a 10% target.
- **Vol targeting sizes up when measured volatility falls,** including during stale-price runs. The leverage cap limits this, and the data warning flags the runs.
- **Leverage is financed at rf,** with no spread and no margin constraints beyond `max_leverage`.
- **GBM assumes iid normal log returns:** no fat tails, no volatility clustering, and constant drift and volatility.
- **The bootstrap assumes stationarity** and can't produce a day worse than the worst day in the history. The block length is a tuning choice; Phase 7 adds an estimated one.
- **Monte Carlo ignores parameter uncertainty,** treating μ̂ and σ̂ as known, and it inherits any overfitting in the backtest it resamples: projecting an overfit strategy projects the overfitting.
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
