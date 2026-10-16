"""Shared test fixtures (§12).

Phase 1 needs only synthetic data. The fake network, the frozen clock, the yfinance-layout
frames and the temp-file database arrive in Phase 2 with the loader and the API.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
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


def constant_rf(index: pd.DatetimeIndex, percent: float = IRX_PERCENT) -> pd.Series:
    """The daily rate a constant ^IRX quote gives: percent / 100 / 252 on every day (§4.5)."""
    return pd.Series(percent / 100 / 252, index=index, name="rf")


def tbill_rf(index: pd.DatetimeIndex, seed: int = 0) -> pd.Series:
    """A varying, T-bill-style daily rate: an annual yield wandering between 0% and 5%."""
    g = np.random.default_rng(seed)
    yield_pct = np.clip(2.0 + np.cumsum(g.normal(0.0, 0.03, len(index))), 0.0, 5.0)
    return pd.Series(yield_pct / 100 / 252, index=index, name="rf")


_PRICES = make_prices()


@pytest.fixture
def prices() -> pd.DataFrame:
    """The 1,000-day fixture: adjusted closes of Y, X and SPY (a fresh copy per test)."""
    return _PRICES.copy()


@pytest.fixture
def rf(prices: pd.DataFrame) -> pd.Series:
    """rf from ^IRX held at 2.0 (§12): a constant-valued Series on the fixture calendar."""
    return constant_rf(prices.index)


@pytest.fixture
def synthetic_prices() -> Callable[..., pd.DataFrame]:
    """Factory for other lengths and seeds of the same synthetic market."""
    return make_prices


@pytest.fixture
def varying_rf() -> Callable[..., pd.Series]:
    """Factory for a T-bill-style varying rf on a given calendar."""
    return tbill_rf
