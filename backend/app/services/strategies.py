"""Scores, entry/exit rules and unit weights for strategies A-D (§5.1-§5.5, §5.7-§5.8).

Pure (§0, rule 8). Every function takes pandas objects indexed by the run calendar and returns
new ones on the same index. A score or signal "at date t" uses data up to and including the
close of day t. Nothing here shifts a signal: the execution lag is applied only in the
backtester (§2.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from app.services import stats

# §5.2/§5.3: a rolling variance of rebased log prices at or below this is a degenerate window.
VAR_FLOOR = 1e-12
# §5.3: the pairs residual variance must exceed this fraction of the rolling Var(y).
RESID_VAR_RATIO = 1e-8


def hysteresis_signal(z: pd.Series, z_entry: float, z_exit: float) -> pd.Series:
    """Entry/exit crossing rule shared by strategies A and B (§5.1).

    Args:
        z: score; the value at date t uses closes up to and including day t. NaN means flat.
        z_entry: entry threshold (dimensionless, in standard deviations), > 0.
        z_exit: exit threshold, 0 <= z_exit < z_entry.

    Returns:
        The raw signal s_t in {-1.0, 0.0, +1.0}: the position wanted at the close of day t.
        Short when z > z_entry, long when z < -z_entry; a short exits once z < z_exit and a
        long once z > -z_exit. Comparisons are strict, and a direct flip in one bar is allowed.
    """
    if not 0 <= z_exit < z_entry:
        raise ValueError(f"need 0 <= z_exit < z_entry, got z_exit={z_exit}, z_entry={z_entry}")
    # Two independent hysteresis legs; they can't both be on when 0 <= z_exit < z_entry.
    # NaN compares False, so a NaN day turns neither leg on, and it turns both off.
    v = z.to_numpy(dtype=float)
    flat = np.isnan(v)
    short = _latch(on=v > z_entry, off=(v < z_exit) | flat)
    long = _latch(on=v < -z_entry, off=(v > -z_exit) | flat)
    return pd.Series(long - short, index=z.index, name="signal")


def _latch(*, on: np.ndarray, off: np.ndarray) -> np.ndarray:
    """1.0 from each `on` day up to the next `off` day, which is 0.0; 0.0 before the first `on`.

    Each day takes the value of the latest day that set the latch (1.0 if on, 0.0 if off; off
    wins when both are true), so this is a forward fill of those days.
    """
    setting = on | off
    latest = np.maximum.accumulate(np.where(setting, np.arange(on.size), -1))  # -1: none yet
    return np.where((latest >= 0) & ~off[latest] & on[latest], 1.0, 0.0)


def first_of_period(index: pd.DatetimeIndex, freq: Literal["W", "M"]) -> pd.Series:
    """True on the first trading day of each calendar week ("W") or month ("M") (§5.5).

    Uses a positional Series.shift. PeriodIndex.shift would move each period value forward by
    one period instead of lagging by one row, which marks every day as a period start.
    """
    p = pd.Series(index.to_period(freq), index=index)
    return p.ne(p.shift(1))


# ---------------------------------------------------------------- Strategy A: mean reversion


def mean_reversion_score(close: pd.Series, L: int) -> pd.Series:
    """Bollinger z-score of the rebased log adjusted close (§5.2).

    Args:
        close: adjusted close of one ticker.
        L: trailing window in trading days; the window includes day t.

    Returns:
        z_t = (lc_t - mean_L(lc)) / std_L(lc), with lc = ln(close / close[0]) and ddof=1.
        NaN during the first L - 1 days and on windows whose variance is <= 1e-12.
    """
    lc = np.log(close / close.iloc[0])  # rebased log price: causal and translation-invariant
    r = lc.rolling(L, min_periods=L)
    m, v = r.mean(), r.var()
    z = (lc - m) / np.sqrt(v.where(v > VAR_FLOOR))  # a constant window has var exactly 0.0 -> NaN -> flat
    return z.rename("z")


# ---------------------------------------------------------------- Strategy B: pairs


@dataclass(frozen=True)
class PairsScore:
    """Step 1 of §5.0 for pairs.

    z is the residual z-score of today's point against the trailing OLS fit of y on x.
    beta is that fit's rolling slope: the hedge ratio, the change in ln(Y) per unit change
    in ln(X). Both use closes up to and including day t.
    """

    z: pd.Series
    beta: pd.Series


def pairs_score(y_close: pd.Series, x_close: pd.Series, L: int) -> PairsScore:
    """Rolling-OLS residual z-score of Y (= tickers[0]) on X (= tickers[1]) (§5.3).

    Args:
        y_close, x_close: adjusted closes on the same calendar.
        L: trailing window in trading days, including day t.

    Returns:
        PairsScore. z is NaN during warm-up, where Var_L(x) <= 1e-12, and where the residual
        variance is <= 1e-8 * Var_L(y) (near-collinear legs). It is never +-inf.
    """
    y, x = np.log(y_close / y_close.iloc[0]), np.log(x_close / x_close.iloc[0])  # rebased logs
    ry, rx = y.rolling(L, min_periods=L), x.rolling(L, min_periods=L)
    my, mx, vy, vx = ry.mean(), rx.mean(), ry.var(), rx.var()
    beta = ry.cov(x) / vx.where(vx > VAR_FLOOR)  # rolling OLS slope of y on x
    resid_var = vy - beta**2 * vx  # = SSR / (L - 1)
    spread = (y - my) - beta * (x - mx)  # OLS residual of today's point
    z = spread / np.sqrt(resid_var.where(resid_var > RESID_VAR_RATIO * vy))
    return PairsScore(z=z.rename("z"), beta=beta.rename("beta"))


def kalman_pairs_score(y_close: pd.Series, x_close: pd.Series, L: int, delta: float) -> PairsScore:
    """Kalman-filter score and hedge ratio of Y (= tickers[0]) on X (= tickers[1]) (§5.7).

    Replaces pairs_score in step 1 of §5.0 when pairs_model="kalman"; everything downstream is
    unchanged. On the rebased log prices of §5.3, the state (alpha_t, beta_t) is a random walk
    with q = delta / (1 - delta), seeded by the OLS fit on days 0..L-1.

    Returns:
        PairsScore. z is the filter's standardized innovation: its prediction uses data through
        t - 1, and it is N(0, 1) under the model. beta is the filtered hedge ratio, using closes
        through t. Both are NaN for t < L, and everywhere when the warm-up window has a flat X or
        exactly collinear legs.
    """
    y, x = np.log(y_close / y_close.iloc[0]), np.log(x_close / x_close.iloc[0])  # rebased logs
    kf = stats.kalman_filter(y.to_numpy(), x.to_numpy(), L, delta)
    index = y_close.index
    return PairsScore(
        z=pd.Series(kf.z, index=index, name="z"), beta=pd.Series(kf.beta, index=index, name="beta")
    )


def hedge_ratio_used(
    beta: pd.Series, signal: pd.Series, hedge_mode: Literal["fixed_at_entry", "rolling"]
) -> pd.Series:
    """The hedge ratio actually held (§5.3).

    "fixed_at_entry" captures the rolling beta on the day a position opens (an entry from flat
    or a direct flip) and holds it for the whole trade; between trades the last captured value
    carries forward and is unused because the weights are 0. "rolling" re-hedges daily.
    The value at date t applies to the signal at date t (it is lagged with the weights).
    """
    if hedge_mode == "rolling":
        return beta.rename("beta_used")
    if hedge_mode == "fixed_at_entry":
        opens = signal.ne(signal.shift()) & signal.ne(0)
        return beta.where(opens).ffill().rename("beta_used")
    raise ValueError(f"unknown hedge_mode {hedge_mode!r}")


def pair_unit_weights(signal: pd.Series, beta_used: pd.Series, tickers: tuple[str, str]) -> pd.DataFrame:
    """Unit leg weights for pairs, with gross exposure 1 (§5.3).

    A long spread (s = +1) holds, per unit, long $1 of Y and short $beta of X, scaled by
    1 / (1 + |beta|). Columns are the tickers, so they match the returns frame (§6.1).
    Weights are exactly 0.0 when flat, including during warm-up.
    """
    s, beta = signal.to_numpy(dtype=float), beta_used.to_numpy(dtype=float)
    open_ = s != 0
    if np.isnan(beta[open_]).any():
        raise ValueError("beta_used is NaN on a day with an open pairs position")
    w_y = np.where(open_, s / (1 + np.abs(beta)), 0.0)
    w_x = np.where(open_, -s * beta / (1 + np.abs(beta)), 0.0)
    return pd.DataFrame({tickers[0]: w_y, tickers[1]: w_x}, index=signal.index)


# ---------------------------------------------------------------- Strategy C: time-series momentum


def trend_score(close: pd.Series, rf: pd.Series, L: int) -> pd.Series:
    """Raw trailing L-day excess log return m (step 1 of §5.0 for Strategy C, §5.5).

    Args:
        close: adjusted close of one ticker.
        rf: daily risk-free rate as a decimal (§4.5), a Series on the same calendar.
        L: lookback in trading days.

    Returns:
        m_t = ln(close_t / close_{t-L}) - sum of rf over days t-L+1..t. It is NaN for the
        first L rows. Uses data up to and including the close of day t (rf_t is the rate
        observed at the previous close).
    """
    if not isinstance(rf, pd.Series):
        raise TypeError(f"rf must be a pandas Series on the run calendar (§2.3), not {type(rf).__name__}")
    if not rf.index.equals(close.index):
        raise ValueError("rf must have the same index as close")
    m = np.log(close / close.shift(L)) - rf.rolling(L, min_periods=L).sum()
    return m.rename("m")


def trend_signal(m: pd.Series, signal_frequency: Literal["monthly", "daily"]) -> tuple[pd.Series, pd.Series]:
    """Step 3 of §5.0 for Strategy C: monthly sampling (if monthly), then the sign.

    Args:
        m: trailing excess return from trend_score, already masked by the vol forecast when
            the overlay is on.
        signal_frequency: "monthly" decides on the first trading day of each month and holds
            that value until the next month start; "daily" re-decides every day.

    Returns:
        (score, signal). The score is the sampled m that the API reports; the signal is its
        sign in {-1.0, 0.0, +1.0}, with exactly 0 or NaN meaning flat.
    """
    if signal_frequency == "monthly":
        m = m.where(first_of_period(m.index, "M")).ffill()
    elif signal_frequency != "daily":
        raise ValueError(f"unknown signal_frequency {signal_frequency!r}")
    signal = np.sign(m).fillna(0.0)
    return m.rename("m"), signal.rename("signal")


def long_only(signal: pd.Series) -> pd.Series:
    """The long-only switch (§5.9): clip the raw signal at 0, at the end of step 3 of §5.0.

    The §5.1 hysteresis legs are computed before the clip. They are never on together, so for
    strategies A and C the clipped signal is exactly the long leg: shorts become flat.
    """
    return signal.clip(lower=0.0).rename("signal")


def single_unit_weights(signal: pd.Series, ticker: str) -> pd.DataFrame:
    """Unit weight for strategies A and C: one leg, gross exposure 1, equal to the signal."""
    return signal.to_frame(ticker)
