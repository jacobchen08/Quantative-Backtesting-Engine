"""Volatility-targeting overlay (§5.6). Pure (§0, rule 8).

The overlay runs in two separate places in the §5.0 call order, so it is two functions:
forecast_and_mask() before any signal exists (step 2), and size_weights() after the signal
and unit weights exist (step 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from app.services.strategies import first_of_period


def pair_unit_return(r_y: pd.Series, r_x: pd.Series, beta: pd.Series) -> pd.Series:
    """Day-t return of a unit long-spread position formed with yesterday's rolling beta (§5.6).

    u_t = (r_Y,t - beta_{t-1} * r_X,t) / (1 + |beta_{t-1}|), with simple daily returns as
    decimals. Uses the rolling beta, not beta_used, so u exists even while the strategy is flat.
    """
    b1 = beta.shift(1)
    return ((r_y - b1 * r_x) / (1 + b1.abs())).rename("u")


def vol_forecast(u: pd.Series, vol_halflife: int) -> pd.Series:
    """Zero-mean EWMA volatility forecast, annualized (§5.6, RiskMetrics style).

    sigma_t = sqrt(252 * sum_i (1-a)^i u_{t-i}^2 / sum_i (1-a)^i), with a = 1 - 2^(-1/H).
    Uses u up to and including day t. NaN until H valid observations of u exist; a NaN u
    later on keeps the previous value.
    """
    if vol_halflife < 1:
        raise ValueError(f"vol_halflife must be >= 1, got {vol_halflife}")
    ewm = u.pow(2).ewm(halflife=vol_halflife, adjust=True, min_periods=vol_halflife)
    return np.sqrt(252 * ewm.mean()).rename("vol_forecast")


def forecast_and_mask(u: pd.Series, score: pd.Series, vol_halflife: int) -> tuple[pd.Series, pd.Series]:
    """Step 2 of §5.0: the vol forecast, then the score masked to NaN where it is undefined.

    Masking before the entry rule (and before Strategy C's monthly sampling) makes t0 the
    first day on which both the score and the forecast exist, so no position can carry over
    from the warm-up.

    Returns:
        (sigma, masked score).
    """
    sigma = vol_forecast(u, vol_halflife)
    return sigma, score.where(sigma.notna())


@dataclass(frozen=True)
class SizedWeights:
    """Step 5 of §5.0.

    weights: sized target weights W (fractions of equity), on the signal date, not lagged.
    scale: target / sigma capped at max_leverage, as of each day's close.
    scale_used: the scale actually applied: refreshed on vol_refresh days and on every signal
        change, held in between.
    """

    weights: pd.DataFrame
    scale: pd.Series
    scale_used: pd.Series


def size_weights(
    unit_weights: pd.DataFrame,
    signal: pd.Series,
    sigma: pd.Series,
    *,
    vol_target: float,
    max_leverage: float,
    vol_refresh: Literal["daily", "weekly", "monthly"],
) -> SizedWeights:
    """Step 5 of §5.0: scale the unit weights toward an annualized volatility target (§5.6).

    Args:
        unit_weights: gross-1 weights from the strategy, columns = tickers.
        signal: raw signal in {-1, 0, +1}.
        sigma: annualized vol forecast from forecast_and_mask (decimal, e.g. 0.15 = 15%).
        vol_target: annualized target vol as a decimal.
        max_leverage: cap on the scale (gross exposure multiple).
        vol_refresh: how often the size may change between signal changes.

    Returns:
        SizedWeights. Weights are exactly 0.0 whenever the signal is flat. Raises ValueError
        if any weight is NaN.
    """
    if not vol_target > 0:
        raise ValueError(f"vol_target must be > 0, got {vol_target}")
    if not max_leverage > 0:
        raise ValueError(f"max_leverage must be > 0, got {max_leverage}")
    scale = (vol_target / sigma).clip(upper=max_leverage)  # sigma == 0 -> inf -> clipped to the cap
    if vol_refresh == "daily":
        refresh = pd.Series(True, index=sigma.index)
    elif vol_refresh in ("weekly", "monthly"):
        refresh = first_of_period(sigma.index, "W" if vol_refresh == "weekly" else "M")
    else:
        raise ValueError(f"unknown vol_refresh {vol_refresh!r}")
    scale_used = scale.where(refresh | signal.ne(signal.shift())).ffill()
    W = unit_weights.mul(scale_used, axis=0)
    W.loc[signal == 0] = 0.0  # exactly 0 when flat (NaN x 0 is NaN)
    if W.isna().any().any():
        raise ValueError("sized weights contain NaN")
    return SizedWeights(weights=W, scale=scale.rename("scale"), scale_used=scale_used.rename("scale_used"))
