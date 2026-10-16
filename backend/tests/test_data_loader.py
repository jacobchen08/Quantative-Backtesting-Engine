"""The Yahoo seam, the price cache, cleaning and alignment (§12, test_data_loader.py)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yfinance.multi
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from yfinance.exceptions import YFRateLimitError

from app.models import HistoricalPrice, TickerCoverage
from app.services import data_loader
from app.services.data_loader import PRICE_COLUMNS, DataSourceError, DownloadError
from app.services.data_loader import (
    fetch_from_yahoo as real_fetch_from_yahoo,
)  # bound before the autouse fake

START = dt.date(2018, 1, 1)
END = dt.date(2021, 10, 29)  # the last fixture day


def stored_rows(session: Session, ticker: str) -> pd.DataFrame:
    rows = session.execute(
        select(HistoricalPrice.date, HistoricalPrice.adj_close)
        .where(HistoricalPrice.ticker == ticker)
        .order_by(HistoricalPrice.date)
    ).all()
    return pd.DataFrame(rows, columns=["date", "adj_close"])


# ---------------------------------------------------------------- the network seam


def test_fetch_from_yahoo_includes_the_requested_end_date(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_download(ticker: str, **kwargs: object) -> pd.DataFrame:
        seen.update(ticker=ticker, **kwargs)
        days = pd.bdate_range(
            kwargs["start"], kwargs["end"], inclusive="left", name="Date"
        )  # end is exclusive
        return pd.DataFrame({"Adj Close": 1.0, "Close": 1.0, "Volume": 1}, index=days.as_unit("s"))

    monkeypatch.setattr(data_loader.yf, "download", fake_download)
    frame = real_fetch_from_yahoo("KO", dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    assert seen["end"] == dt.date(2024, 1, 6)
    assert frame.index[-1] == pd.Timestamp("2024-01-05")
    flags = ("auto_adjust", "actions", "multi_level_index", "progress", "threads")
    assert {key: seen[key] for key in flags} == dict.fromkeys(flags, False)


def stub_ticker(raised: BaseException | None = None, soft_error: str | None = None) -> type:
    """A yfinance Ticker whose history() raises, or returns nothing with a soft error, offline."""

    class StubTicker:
        def __init__(self, ticker: str, *args: object, **kwargs: object) -> None:
            self._price_history = type("PriceHistory", (), {"_last_error": soft_error})()

        def history(self, **kwargs: object) -> pd.DataFrame:
            if raised is not None:
                raise raised
            return pd.DataFrame()

    return StubTicker


@pytest.mark.parametrize(
    "raised, expected",
    [
        (YFRateLimitError(), YFRateLimitError),
        (ConnectionError("connection refused"), DownloadError),
        (TimeoutError("read timed out"), DownloadError),
    ],
)
def test_fetch_from_yahoo_reraises_what_yf_download_swallows(
    monkeypatch: pytest.MonkeyPatch, raised: BaseException, expected: type
) -> None:
    # The real yf.download catches the exception, logs it and returns an empty frame.
    monkeypatch.setattr(yfinance.multi, "Ticker", stub_ticker(raised=raised))
    with pytest.raises(expected):
        real_fetch_from_yahoo("KO", dt.date(2024, 1, 2), dt.date(2024, 1, 5))


def test_an_unknown_ticker_comes_back_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        yfinance.multi, "Ticker", stub_ticker(soft_error="possibly delisted; no timezone found")
    )
    assert real_fetch_from_yahoo("NOPE", dt.date(2024, 1, 2), dt.date(2024, 1, 5)).empty


@pytest.mark.parametrize(
    "error, status",
    [(None, 404), (YFRateLimitError(), 503), (DownloadError("ConnectionError('refused')"), 502)],
    ids=["empty-frame", "rate-limit", "download-error"],
)
def test_upstream_failures_map_to_http_statuses(
    session: Session, fake_yahoo, error: BaseException | None, status: int
) -> None:
    if error is None:
        fake_yahoo.frames["ZZZ"] = fake_yahoo.frames["SPY"].iloc[:0]
    else:
        fake_yahoo.errors["ZZZ"] = error
    with pytest.raises(DataSourceError) as err:
        data_loader.load_close(session, "ZZZ", START, END)
    assert err.value.status_code == status
    if status == 404:
        assert err.value.detail == f"No data for ZZZ between {START} and {END}"
    assert session.get(TickerCoverage, "ZZZ") is None  # a failure caches nothing


def test_normalize_flattens_a_multiindex_and_uses_ns_dates(yf_frame) -> None:
    raw = yf_frame(pd.Series([10.0, 11.0], index=pd.bdate_range("2024-01-02", periods=2)))
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["KO"]], names=["Price", "Ticker"])
    out = data_loader.normalize(raw)
    assert list(out.columns) == PRICE_COLUMNS
    assert out.index.dtype == "datetime64[ns]" and out.index.name == "date"
    assert out["adj_close"].tolist() == [10.0, 11.0]


# ---------------------------------------------------------------- cache hits and refetches


def test_a_repeat_request_makes_no_fetch(session: Session, fake_yahoo) -> None:
    data_loader.load_market(session, ["Y", "X"], START, END, "tbill")
    first = list(fake_yahoo.calls)
    assert sorted(ticker for ticker, *_ in first) == ["SPY", "X", "Y", "^IRX"]  # one call per ticker
    data_loader.load_market(session, ["Y", "X"], START, END, "tbill")
    data_loader.load_market(
        session, ["Y"], START + dt.timedelta(days=30), END - dt.timedelta(days=30), "tbill"
    )
    assert fake_yahoo.calls == first


def test_a_ticker_listed_after_start_is_not_refetched(session: Session, fake_yahoo) -> None:
    fake_yahoo.frames["LATE"] = fake_yahoo.frames["SPY"].loc["2019-06-03":]
    close, _ = data_loader.load_close(session, "LATE", START, END)
    assert close.index[0] == pd.Timestamp("2019-06-03")
    calls = len(fake_yahoo.calls)
    data_loader.load_close(session, "LATE", START, END)
    assert len(fake_yahoo.calls) == calls
    coverage = session.get(TickerCoverage, "LATE")
    assert (coverage.cov_start, coverage.cov_end) == (START, END)  # the requested range, not the data's


def test_an_extension_refetches_the_union_on_the_new_adjustment_basis(session: Session, fake_yahoo) -> None:
    original = fake_yahoo.frames["Y"]
    mid = dt.date(2019, 12, 31)
    data_loader.load_close(session, "Y", START, mid)
    # A dividend since then: Yahoo now serves every earlier adjusted close scaled by 0.98.
    fake_yahoo.frames["Y"] = original.assign(**{"Adj Close": original["Adj Close"] * 0.98})
    close, _ = data_loader.load_close(session, "Y", dt.date(2019, 6, 3), END)
    assert fake_yahoo.calls[-1] == ("Y", START, END)  # one download of the whole union
    stored = stored_rows(session, "Y")
    expected = original["Adj Close"].loc[str(START) : str(END)] * 0.98
    assert len(stored) == len(expected)
    np.testing.assert_allclose(stored["adj_close"].to_numpy(), expected.to_numpy(), rtol=1e-15)
    seam = pd.Timestamp(mid)
    nxt = close.index[close.index > seam][0]
    true_return = original["Adj Close"][nxt] / original["Adj Close"][seam]
    assert close[nxt] / close[seam] == pytest.approx(true_return, rel=1e-14)  # no jump at the seam


# ---------------------------------------------------------------- storage


def test_a_bar_dated_today_is_never_stored(
    session: Session, fake_yahoo, yf_frame, frozen_today: dt.date
) -> None:
    days = pd.bdate_range(frozen_today - dt.timedelta(days=60), frozen_today)
    assert days[-1].date() == frozen_today
    fake_yahoo.frames["LIVE"] = yf_frame(pd.Series(np.linspace(50, 51, len(days)), index=days))
    fake_yahoo.ignore_end.add("LIVE")  # Yahoo serves today's partial bar anyway
    data_loader.load_close(session, "LIVE", days[0].date(), frozen_today - dt.timedelta(days=1))
    stored = stored_rows(session, "LIVE")
    assert len(stored) == len(days) - 1 and max(stored["date"]) < frozen_today


def test_overlapping_requests_store_each_day_once(session: Session) -> None:
    data_loader.load_close(session, "Y", START, dt.date(2019, 6, 30))
    data_loader.load_close(session, "Y", dt.date(2019, 1, 1), dt.date(2020, 6, 30))
    count = select(func.count()).select_from(HistoricalPrice).where(HistoricalPrice.ticker == "Y")
    distinct = select(func.count(func.distinct(HistoricalPrice.date))).where(HistoricalPrice.ticker == "Y")
    n = session.scalar(count)
    assert n == session.scalar(distinct) == len(pd.bdate_range(START, "2020-06-30"))
    frame = data_loader.normalize(data_loader.fetch_from_yahoo("Y", START, dt.date(2020, 6, 30)))
    data_loader.upsert_prices(session, "Y", frame)  # the same rows again: overwritten, not duplicated
    session.commit()
    assert session.scalar(count) == n


def test_a_5000_row_upsert_works_on_sqlite(session: Session, yf_frame) -> None:
    days = pd.bdate_range("2000-01-03", periods=5000)  # 40,000 parameters in one INSERT would fail
    frame = data_loader.normalize(yf_frame(pd.Series(np.linspace(10.0, 20.0, 5000), index=days)))
    data_loader.upsert_prices(session, "BIG", frame)
    session.commit()
    assert session.scalar(select(func.count()).where(HistoricalPrice.ticker == "BIG")) == 5000


# ---------------------------------------------------------------- cleaning (§4.4)


def test_non_positive_prices_are_dropped_with_a_warning(session: Session, fake_yahoo) -> None:
    frame = fake_yahoo.frames["SPY"].copy()
    frame.iloc[10, frame.columns.get_loc("Close")] = 0.0
    frame.iloc[11, frame.columns.get_loc("Adj Close")] = -1.0
    fake_yahoo.frames["BAD"] = frame
    close, warnings = data_loader.load_close(session, "BAD", START, END)
    dropped = pd.DatetimeIndex(frame.index[[10, 11]]).as_unit("ns")
    assert len(close) == len(frame) - 2 and not close.index.isin(dropped).any()
    assert len(warnings) == 1 and "dropped 2 row(s) with a non-positive close" in warnings[0]
    assert f"{dropped[0]:%Y-%m-%d}, {dropped[1]:%Y-%m-%d}" in warnings[0]


def test_big_moves_and_stale_runs_are_kept_with_warnings(session: Session, fake_yahoo, yf_frame) -> None:
    adj = fake_yahoo.frames["SPY"]["Adj Close"].copy()
    adj.iloc[200:] *= 2.348  # a +134.8% day, like GME on 2021-01-27
    adj.iloc[301:306] = adj.iloc[300]  # six identical closes in a row
    fake_yahoo.frames["ODD"] = yf_frame(adj)
    close, warnings = data_loader.load_close(session, "ODD", START, END)
    assert len(close) == len(adj)  # nothing dropped
    assert any("daily return above 50%" in w and f"{adj.index[200]:%Y-%m-%d}" in w for w in warnings)
    assert any(
        "run(s) of 5 or more identical adjusted closes" in w and f"{adj.index[300]:%Y-%m-%d}" in w
        for w in warnings
    )


# ---------------------------------------------------------------- alignment


def test_mismatched_dates_are_inner_joined_never_forward_filled(session: Session, fake_yahoo) -> None:
    frame = fake_yahoo.frames["X"]
    gone = frame.index[[100, 101, 102, 500]]
    fake_yahoo.frames["GAPPY"] = frame.drop(gone)
    market = data_loader.load_market(session, ["Y", "GAPPY"], START, END, 0.02)
    assert frame.index.dtype == "datetime64[s]"  # the fake serves yfinance's resolution...
    assert market.prices.index.dtype == "datetime64[ns]" and market.rf.index.dtype == "datetime64[ns]"
    assert not market.prices.isna().any().any() and not market.prices.index.isin(gone).any()
    assert len(market.prices) == len(frame) - 4
    gappy = market.prices["GAPPY"]
    before, after = pd.Timestamp(frame.index[99]), pd.Timestamp(frame.index[103])
    assert gappy.index.get_loc(after) == gappy.index.get_loc(before) + 1  # the next row spans the gap


# ---------------------------------------------------------------- risk-free rate


@pytest.fixture
def varying_irx(fake_yahoo, yf_frame) -> pd.Series:
    """A ^IRX that changes every day, with no row on Veterans Day 2019 and a bad print of 99.0."""
    days = pd.bdate_range("2017-11-01", END)
    level = pd.Series(1.0 + 0.001 * np.arange(len(days)), index=days)
    level = level.drop(pd.Timestamp("2019-11-11"))  # bond market closed, stock market open
    level[pd.Timestamp("2020-03-10")] = 99.0
    fake_yahoo.frames["^IRX"] = yf_frame(level)
    return level


def test_rf_is_the_last_irx_close_strictly_before_each_day(session: Session, varying_irx: pd.Series) -> None:
    market = data_loader.load_market(session, ["Y"], START, END, "tbill")
    valid = varying_irx[varying_irx <= 25]
    prior = valid.iloc[valid.index.searchsorted(market.rf.index, side="left") - 1]
    np.testing.assert_array_equal(market.rf.to_numpy(), prior.to_numpy() / 100 / 252)
    rate = market.rf * 100 * 252
    assert rate[pd.Timestamp("2019-11-11")] == pytest.approx(varying_irx[pd.Timestamp("2019-11-08")])
    assert rate[pd.Timestamp("2019-11-12")] == pytest.approx(varying_irx[pd.Timestamp("2019-11-08")])
    assert rate[pd.Timestamp("2020-03-11")] == pytest.approx(
        varying_irx[pd.Timestamp("2020-03-09")]
    )  # 99.0 dropped
    assert not market.rf.isna().any()


def test_a_request_starting_on_a_day_without_irx_still_has_a_rate(
    session: Session, fake_yahoo, varying_irx: pd.Series
) -> None:
    start = dt.date(2019, 11, 11)
    market = data_loader.load_market(session, ["Y"], start, END, "tbill")
    assert market.prices.index[0] == pd.Timestamp(start)
    assert market.rf.iloc[0] * 100 * 252 == pytest.approx(varying_irx[pd.Timestamp("2019-11-08")])
    irx_call = next(call for call in fake_yahoo.calls if call[0] == "^IRX")
    assert irx_call[1] == start - dt.timedelta(days=14)


def test_a_constant_risk_free_rate_needs_no_irx(session: Session, fake_yahoo) -> None:
    market = data_loader.load_market(session, ["Y"], START, END, 0.03)
    assert (market.rf == 0.03 / 252).all()
    assert "^IRX" not in {ticker for ticker, *_ in fake_yahoo.calls}
