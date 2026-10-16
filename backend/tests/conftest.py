"""Shared test fixtures (§12).

Two autouse fixtures apply to every test:
- a fake network: data_loader.fetch_from_yahoo is replaced by FakeYahoo, which serves
  synthetic frames in yfinance's exact layout, counts its calls and fails the test for any
  ticker it doesn't know, so no test can reach Yahoo;
- a frozen clock: calendar.today_ny() returns FROZEN_TODAY, after the fixture data ends.
API tests get a temp-file SQLite database through a dependency override.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.signal import lfilter
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base, get_session, make_engine
from app.main import app
from app.services import calendar, data_loader

N_DAYS = 1000
START = "2018-01-01"
SEED = 20261015
IRX_PERCENT = 2.0  # ^IRX held constant at 2.0, i.e. 2% a year
FROZEN_TODAY = dt.date(2022, 1, 3)  # a Monday after the fixture data ends (2021-10-29)


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


_PRICES = make_prices()
assert _PRICES.index[-1].date() < FROZEN_TODAY
_IRX_DAYS = pd.bdate_range("2017-11-01", _PRICES.index[-1])  # covers the 14-day buffer before START
YF_FRAMES = {
    **{ticker: yf_layout(_PRICES[ticker], seed=i) for i, ticker in enumerate(_PRICES.columns)},
    "^IRX": yf_layout(pd.Series(IRX_PERCENT, index=_IRX_DAYS)),
}


class FakeYahoo:
    """Stands in for data_loader.fetch_from_yahoo (§12).

    frames maps a ticker to its whole history in yfinance layout; a call returns the rows in
    [start, end_inclusive], or every row from start on for tickers in ignore_end (Yahoo
    serving today's partial bar). errors maps a ticker to an exception to raise. Every call is
    recorded in calls as (ticker, start, end_inclusive).
    """

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = dict(frames)
        self.errors: dict[str, BaseException] = {}
        self.ignore_end: set[str] = set()
        self.calls: list[tuple[str, dt.date, dt.date]] = []

    def __call__(self, ticker: str, start: dt.date, end_inclusive: dt.date) -> pd.DataFrame:
        self.calls.append((ticker, start, end_inclusive))
        if ticker in self.errors:
            raise self.errors[ticker]
        if ticker not in self.frames:
            pytest.fail(f"the fake network has no data for {ticker!r}")  # BaseException: not mapped to 502
        frame = self.frames[ticker]
        keep = frame.index >= pd.Timestamp(start)
        if ticker not in self.ignore_end:
            keep &= frame.index <= pd.Timestamp(end_inclusive)
        return frame.loc[keep].copy()


@pytest.fixture(autouse=True)
def fake_yahoo(monkeypatch: pytest.MonkeyPatch) -> FakeYahoo:
    fake = FakeYahoo(YF_FRAMES)
    monkeypatch.setattr(data_loader, "fetch_from_yahoo", fake)
    return fake


@pytest.fixture(autouse=True)
def frozen_today(monkeypatch: pytest.MonkeyPatch) -> dt.date:
    monkeypatch.setattr(calendar, "today_ny", lambda: FROZEN_TODAY)
    return FROZEN_TODAY


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


@pytest.fixture
def yf_frame() -> Callable[..., pd.DataFrame]:
    """Factory for frames in yfinance layout, for tests that add tickers to the fake network."""
    return yf_layout


@pytest.fixture
def db_engine(tmp_path: Path) -> Iterator[Engine]:
    """A temp-file SQLite database with every table created."""
    engine = make_engine(f"sqlite:///{(tmp_path / 'quant.db').as_posix()}")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(db_engine: Engine) -> Iterator[Session]:
    with Session(db_engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture
def client(db_engine: Engine) -> Iterator[TestClient]:
    """A TestClient whose requests use the temp database. The app's lifespan doesn't run."""
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)

    def override() -> Iterator[Session]:
        with factory() as s:
            yield s

    app.dependency_overrides[get_session] = override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_session, None)
