"""Price cache and the single network seam fetch_from_yahoo() (§4.2-§4.5).

All network access goes through fetch_from_yahoo(). Nothing else imports yfinance, and tests
replace that one function with a fake that serves frames in the same layout.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import String, select, type_coerce
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session
from yfinance.exceptions import YFRateLimitError

from app.config import offline_mode
from app.models import HistoricalPrice, TickerCoverage
from app.services import calendar
from app.services.backtester import InsufficientDataError
from app.services.pipeline import MARKET

logger = logging.getLogger(__name__)

RATE_TICKER = "^IRX"  # 13-week T-bill yield, quoted in percent
IRX_BUFFER_DAYS = 14  # load ^IRX from start - 14 days so the first day has an earlier rate
IRX_VALID = (-1.0, 25.0)  # percent; yields went slightly negative in 2015-2025, so no clip at 0
UPSERT_CHUNK = 1_000  # rows per statement; SQLite allows at most 32,766 bound parameters
BIG_MOVE = 0.5  # warn when |daily return| exceeds 50%
STALE_RUN = 5  # warn when this many consecutive adjusted closes are identical
YF_COLUMNS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close"}
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
_EXCEPTION_REPR = re.compile(r"^[A-Za-z_][\w.]*\(")  # e.g. "ConnectionError('...')"


class DataSourceError(Exception):
    """An upstream data failure with the HTTP status the API returns: 404, 502 or 503 (§4.3)."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class DownloadError(Exception):
    """yf.download failed with an exception (including a timeout) that it logged instead of raising."""


class _YahooErrorLog(logging.Handler):
    """Collects the per-ticker error lines that yf.download writes to the 'yfinance' logger."""

    def __init__(self, ticker: str) -> None:
        super().__init__(level=logging.ERROR)
        self.prefix = f"['{ticker.upper()}']: "
        self.errors: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith(self.prefix):
            self.errors.append(message[len(self.prefix) :])


def fetch_from_yahoo(ticker: str, start: dt.date, end_inclusive: dt.date) -> pd.DataFrame:
    """Download daily bars for [start, end_inclusive] in one yf.download call (§4.3).

    Returns the frame in yfinance's layout ("Adj Close", "Close", ... columns on a "Date"
    index), empty when Yahoo has no data. yfinance 1.7's download() never raises: it logs each
    failure and returns an empty frame. So this reads that log for the ticker and re-raises:
    YFRateLimitError for a rate limit, DownloadError for any other exception or a timeout.
    An unknown or delisted ticker is not an exception; it comes back as an empty frame.
    """
    log = _YahooErrorLog(ticker)
    yf_logger = logging.getLogger("yfinance")
    yf_logger.addHandler(log)
    started = time.perf_counter()
    try:
        frame = yf.download(
            ticker,
            start=start,
            end=end_inclusive + dt.timedelta(days=1),  # Yahoo's end is exclusive
            auto_adjust=False,
            actions=False,
            multi_level_index=False,
            progress=False,
            threads=False,
        )
    finally:
        yf_logger.removeHandler(log)
    frame = pd.DataFrame() if frame is None else frame
    logger.info(
        "fetch_from_yahoo %s %s..%s: %d rows in %.0f ms",
        ticker,
        start,
        end_inclusive,
        len(frame),
        (time.perf_counter() - started) * 1000,
    )
    for error in log.errors:
        if "YFRateLimitError" in error:
            raise YFRateLimitError()
        if _EXCEPTION_REPR.match(error):
            raise DownloadError(error)
    return frame


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """yfinance layout -> snake_case price columns on a tz-naive datetime64[ns] index named date."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):  # defensive: keep the price level
        level = next(
            (
                i
                for i in range(frame.columns.nlevels)
                if set(frame.columns.get_level_values(i)) & set(YF_COLUMNS)
            ),
            0,
        )
        frame.columns = frame.columns.get_level_values(level)
    frame = frame.rename(columns={**YF_COLUMNS, "Volume": "volume"})
    if not {"close", "adj_close"} <= set(frame.columns):
        raise DownloadError(f"unexpected columns from Yahoo: {list(raw.columns)}")
    for column in PRICE_COLUMNS:
        if column not in frame.columns:
            frame[column] = float("nan")
    frame.index = calendar.to_ns_index(frame.index)
    frame.index.name = "date"
    return frame[PRICE_COLUMNS]


def _storable(frame: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    """Rows that may be cached: close and adj_close present, one row per date, before today (§4.2, §4.4)."""
    frame = frame.dropna(subset=["close", "adj_close"])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame[frame.index < pd.Timestamp(today)]  # never store a bar dated today_ny() or later


def _insert(session: Session) -> Callable[..., Any]:
    name = session.get_bind().dialect.name
    if name == "sqlite":
        return sqlite.insert
    if name == "postgresql":
        return postgresql.insert
    raise RuntimeError(f"unsupported database dialect {name!r}")


def _number(value: object) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def upsert_prices(session: Session, ticker: str, frame: pd.DataFrame) -> None:
    """Insert or overwrite (ON CONFLICT DO UPDATE) the rows of a normalized frame, 1,000 per statement."""
    rows = [
        {
            "ticker": ticker,
            "date": day.date(),
            "open": _number(o),
            "high": _number(h),
            "low": _number(lo),
            "close": float(c),
            "adj_close": float(a),
            "volume": None if pd.isna(v) else int(v),
        }
        for day, o, h, lo, c, a, v in frame[PRICE_COLUMNS].itertuples()
    ]
    insert = _insert(session)
    for i in range(0, len(rows), UPSERT_CHUNK):
        stmt = insert(HistoricalPrice).values(rows[i : i + UPSERT_CHUNK])
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "date"],
            set_={column: stmt.excluded[column] for column in PRICE_COLUMNS},
        )
        session.execute(stmt)


def _set_coverage(session: Session, ticker: str, start: dt.date, end: dt.date) -> None:
    stmt = _insert(session)(TickerCoverage).values(
        ticker=ticker, cov_start=start, cov_end=end, fetched_at=dt.datetime.now(dt.UTC)
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker"],
        set_={
            "cov_start": stmt.excluded.cov_start,
            "cov_end": stmt.excluded.cov_end,
            "fetched_at": stmt.excluded.fetched_at,
        },
    )
    session.execute(stmt)


def _download(ticker: str, start: dt.date, end: dt.date, requested: tuple[dt.date, dt.date]) -> pd.DataFrame:
    """fetch_from_yahoo with the §4.3 error mapping. Called once per ticker per request at most.

    In offline mode (QUANT_OFFLINE=1) a cache miss raises 503 here, before the network seam is
    called, so the browser tests can never reach Yahoo (§4.3).
    """
    if offline_mode():
        raise DataSourceError(503, f"Offline mode: {ticker} is not in the local cache")
    try:
        raw = fetch_from_yahoo(ticker, start, end)
    except YFRateLimitError as exc:
        raise DataSourceError(
            503, f"Yahoo Finance rate-limited the download of {ticker}; try again later."
        ) from exc
    except Exception as exc:  # any other download failure, including a timeout
        raise DataSourceError(502, f"Downloading {ticker} from Yahoo Finance failed: {exc}") from exc
    if raw is None or raw.empty:  # yfinance returns an empty frame for unknown tickers instead of raising
        raise DataSourceError(404, f"No data for {ticker} between {requested[0]} and {requested[1]}")
    return raw


def ensure_cached(session: Session, ticker: str, start: dt.date, end: dt.date) -> None:
    """Make the cache cover [start, end] for one ticker (§4.2). end must already be end*.

    A hit (coverage contains the range) makes no network call. A miss refetches the whole
    union of the old coverage and the request in one download, overwrites every returned row
    and sets the coverage to that union, all in one transaction. Refetching everything keeps
    the ticker on one adjustment basis: Yahoo rewrites past adjusted closes after each
    dividend or split, so rows fetched at different times can't be stitched together.
    """
    coverage = session.get(TickerCoverage, ticker)
    if coverage is not None and coverage.cov_start <= start and coverage.cov_end >= end:
        return
    lo = start if coverage is None else min(start, coverage.cov_start)
    hi = end if coverage is None else max(end, coverage.cov_end)
    raw = _download(ticker, lo, hi, requested=(start, end))  # an empty answer raises before any write
    frame = _storable(normalize(raw), calendar.today_ny())
    try:
        if not frame.empty:
            upsert_prices(session, ticker, frame)
        _set_coverage(session, ticker, lo, hi)
        session.commit()
    except Exception:
        session.rollback()
        raise


def read_prices(session: Session, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Cached close and adj_close for [start, end], on a datetime64[ns] index named date.

    The dates are read as the stored ISO text and parsed by numpy in one call: turning each
    row into a datetime.date first took about 40% of the read. The query runs on the session's
    connection, which skips the ORM's row handling; it reads plain columns, not objects.
    """
    stmt = (
        select(type_coerce(HistoricalPrice.date, String), HistoricalPrice.close, HistoricalPrice.adj_close)
        .where(HistoricalPrice.ticker == ticker, HistoricalPrice.date >= start, HistoricalPrice.date <= end)
        .order_by(HistoricalPrice.date)
    )
    rows = session.connection().execute(stmt).all()
    dates, close, adj_close = zip(*rows, strict=True) if rows else ((), (), ())
    return pd.DataFrame(
        {"close": np.array(close, dtype=float), "adj_close": np.array(adj_close, dtype=float)},
        index=pd.DatetimeIndex(np.array(dates, dtype="datetime64[ns]"), name="date"),
    )


def _dates(index: pd.DatetimeIndex, limit: int = 5) -> str:
    shown = ", ".join(f"{d:%Y-%m-%d}" for d in index[:limit])
    return shown if len(index) <= limit else f"{shown} and {len(index) - limit} more"


def clean_closes(ticker: str, frame: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """§4.4 for a tradable ticker. Returns its adjusted closes and the warnings.

    Rows with close <= 0 or adj_close <= 0 are dropped with a warning. Daily moves above 50%
    and runs of 5 or more identical adjusted closes only produce a warning; the rows stay.
    """
    warnings: list[str] = []
    bad = (frame["close"] <= 0) | (frame["adj_close"] <= 0)
    if bad.any():
        warnings.append(
            f"{ticker}: dropped {int(bad.sum())} row(s) with a non-positive close or adjusted close "
            f"({_dates(frame.index[bad])})."
        )
        frame = frame[~bad]
    adj = frame["adj_close"]
    big = (adj / adj.shift(1) - 1).abs() > BIG_MOVE
    if big.any():
        warnings.append(
            f"{ticker}: daily return above 50% on {_dates(adj.index[big])}; a bad print or an unadjusted "
            "split? The rows are kept."
        )
    run_start = adj.ne(adj.shift())
    run_size = adj.groupby(run_start.cumsum()).transform("size")
    stale = run_start & (run_size >= STALE_RUN)
    if stale.any():
        warnings.append(
            f"{ticker}: {int(stale.sum())} run(s) of {STALE_RUN} or more identical adjusted closes, starting "
            f"{_dates(adj.index[stale])}; stale data or a trading halt? The rows are kept, and vol targeting "
            "sizes up while measured volatility is low."
        )
    return adj.rename(ticker), warnings


def clean_irx(frame: pd.DataFrame) -> pd.Series:
    """^IRX is a rate in percent: drop rows outside [-1, 25] instead of keeping them as NaN (§4.4)."""
    close = frame["close"]
    return close[(close >= IRX_VALID[0]) & (close <= IRX_VALID[1])].rename(RATE_TICKER)


def load_close(session: Session, ticker: str, start: dt.date, end: dt.date) -> tuple[pd.Series, list[str]]:
    """One ticker's cleaned adjusted closes for [start, end] (end = end*), and its warnings."""
    ensure_cached(session, ticker, start, end)
    frame = read_prices(session, ticker, start, end)
    if frame.empty:
        raise DataSourceError(404, f"No data for {ticker} between {start} and {end}")
    return clean_closes(ticker, frame)


def load_rf(session: Session, index: pd.DatetimeIndex, start: dt.date, end: dt.date) -> pd.Series:
    """rf from ^IRX on the run calendar (§4.5), loaded from start - 14 days. NaN left over -> 422."""
    irx_start = start - dt.timedelta(days=IRX_BUFFER_DAYS)
    ensure_cached(session, RATE_TICKER, irx_start, end)
    irx = clean_irx(read_prices(session, RATE_TICKER, irx_start, end))
    rf = calendar.rf_from_irx(irx, index)
    missing = rf.index[rf.isna()]
    if len(missing):
        raise InsufficientDataError(
            f"No valid ^IRX rate before {missing[0]:%Y-%m-%d}; use a constant risk_free or a later start."
        )
    return rf


@dataclass(frozen=True)
class MarketData:
    """Inputs for the pure core: adjusted closes of the legs and SPY on the run calendar, rf, warnings."""

    prices: pd.DataFrame
    rf: pd.Series
    warnings: list[str]


def load_market(
    session: Session, tickers: list[str], start: dt.date, end: dt.date, risk_free: str | float
) -> MarketData:
    """Everything a backtest needs for [start, end], where end is already end* (§4.2-§4.5).

    The run calendar is the inner join of every leg and SPY (loaded once even if it is a leg).
    risk_free is "tbill" (^IRX) or a constant annual rate as a decimal.
    """
    closes: dict[str, pd.Series] = {}
    warnings: list[str] = []
    for ticker in dict.fromkeys([*tickers, MARKET]):
        closes[ticker], found = load_close(session, ticker, start, end)
        warnings += found
    prices = calendar.align_closes(closes)
    if prices.empty:
        raise InsufficientDataError(
            f"{', '.join(closes)} have no trading dates in common between {start} and {end}."
        )
    if risk_free == "tbill":
        rf = load_rf(session, prices.index, start, end)
    else:
        rf = pd.Series(float(risk_free) / 252, index=prices.index, name="rf")
    return MarketData(prices=prices, rf=rf, warnings=warnings)
