"""The synthetic market (§12): shared by conftest.py and scripts/seed_e2e.py.

Every series comes from a fixed seed, so the tests and the browser tests' database see the
same prices. Frames from yf_layout() are in yfinance 1.7's exact column layout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

N_DAYS = 1000
START = "2018-01-01"
SEED = 20261015
IRX_PERCENT = 2.0  # ^IRX held constant at 2.0, i.e. 2% a year


def make_prices(n: int = N_DAYS, seed: int = SEED, start: str = START) -> pd.DataFrame:
    """Adjusted closes on n business days: a cointegrated pair (Y on X) and SPY.

    ln Y = 0.3 + 1.2 ln X + u, where u is an AR(1) with phi = 0.9 and 1% daily shocks, and
    ln X is a random walk with 1.5% daily steps. SPY drifts up 0.04% a day with 1% daily vol.
    """
    g = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n).as_unit("ns")
    x = np.log(40.0) + np.cumsum(g.normal(0.0, 0.015, n))
    u = lfilter([1.0], [1.0, -0.9], g.normal(0.0, 0.01, n))
    y = 0.3 + 1.2 * x + u
    spy = np.log(250.0) + np.cumsum(g.normal(0.0004, 0.01, n))
    return pd.DataFrame({"Y": np.exp(y), "X": np.exp(x), "SPY": np.exp(spy)}, index=idx)


UNIVERSE_VOLS = {"T08": 0.08, "T15": 0.15, "T20": 0.20, "T30": 0.30}  # annual vol of each asset
UNIVERSE_SEED = 20261018


def make_universe(
    index: pd.DatetimeIndex, seed: int = UNIVERSE_SEED, drift: float | None = None
) -> pd.DataFrame:
    """The trend-portfolio universe (§12): four assets with constant volatilities of 8%, 15%,
    20% and 30% a year on the given calendar, from a fixed seed.

    Log returns are iid normal. Each asset's daily drift is random (between -0.03% and +0.06%)
    unless `drift` fixes it for all of them.
    """
    g = np.random.default_rng(seed)
    out = {}
    for ticker, vol in UNIVERSE_VOLS.items():
        mu = float(g.uniform(-0.0003, 0.0006)) if drift is None else drift
        out[ticker] = 50.0 * np.exp(np.cumsum(g.normal(mu, vol / np.sqrt(252), len(index))))
    return pd.DataFrame(out, index=index)


def constant_rf(index: pd.DatetimeIndex, percent: float = IRX_PERCENT) -> pd.Series:
    """The daily rate a constant ^IRX quote gives: percent / 100 / 252 on every day (§4.5)."""
    return pd.Series(percent / 100 / 252, index=index, name="rf")


def tbill_rf(index: pd.DatetimeIndex, seed: int = 0) -> pd.Series:
    """A varying, T-bill-style daily rate: an annual yield wandering between 0% and 5%."""
    g = np.random.default_rng(seed)
    yield_pct = np.clip(2.0 + np.cumsum(g.normal(0.0, 0.03, len(index))), 0.0, 5.0)
    return pd.Series(yield_pct / 100 / 252, index=index, name="rf")


def yf_layout(adj_close: pd.Series, close: pd.Series | None = None, seed: int = 0) -> pd.DataFrame:
    """A frame laid out like yfinance 1.7's yf.download(..., auto_adjust=False, multi_level_index=False).

    Columns "Adj Close", "Close", "High", "Low", "Open", "Volume" on a "Date" index at
    datetime64[s]. Close defaults to the adjusted close (no dividends or splits).
    """
    close = adj_close if close is None else close
    g = np.random.default_rng(seed)
    c = close.to_numpy(dtype=float)
    spread = np.abs(g.normal(0.0, 0.004, len(c)))
    return pd.DataFrame(
        {
            "Adj Close": adj_close.to_numpy(dtype=float),
            "Close": c,
            "High": c * (1 + spread),
            "Low": c * (1 - spread),
            "Open": c * (1 + g.normal(0.0, 0.002, len(c))),
            "Volume": g.integers(1_000_000, 5_000_000, len(c)),
        },
        index=pd.DatetimeIndex(adj_close.index, name="Date").as_unit("s"),
    )
