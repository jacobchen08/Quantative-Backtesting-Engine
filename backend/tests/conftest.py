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

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base, get_session, make_engine
from app.main import app, diagnostics_cache
from app.services import calendar, data_loader
from tests.synthetic import (
    IRX_PERCENT,
    constant_rf,
    make_prices,
    make_universe,
    tbill_rf,
    yf_layout,
)

FROZEN_TODAY = dt.date(2022, 1, 3)  # a Monday after the fixture data ends (2021-10-29)


_PRICES = make_prices()
_UNIVERSE = make_universe(_PRICES.index)
assert _PRICES.index[-1].date() < FROZEN_TODAY
_IRX_DAYS = pd.bdate_range("2017-11-01", _PRICES.index[-1])  # covers the 14-day buffer before START
YF_FRAMES = {
    **{ticker: yf_layout(_PRICES[ticker], seed=i) for i, ticker in enumerate(_PRICES.columns)},
    **{ticker: yf_layout(_UNIVERSE[ticker], seed=10 + i) for i, ticker in enumerate(_UNIVERSE.columns)},
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
def portfolio_prices() -> pd.DataFrame:
    """The 1,000-day fixture plus the four-asset trend-portfolio universe (T08, T15, T20, T30)."""
    return _PRICES.join(_UNIVERSE)


@pytest.fixture
def universe() -> Callable[..., pd.DataFrame]:
    """Factory for the trend-portfolio universe on another calendar, seed or drift."""
    return make_universe


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
    """A TestClient whose requests use the temp database and start with an empty diagnostics
    cache. The app's lifespan doesn't run."""
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    diagnostics_cache.clear()

    def override() -> Iterator[Session]:
        with factory() as s:
            yield s

    app.dependency_overrides[get_session] = override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_session, None)
