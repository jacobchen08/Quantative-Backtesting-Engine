"""ORM tables: historical_prices, ticker_coverage, backtest_runs (§4.1).

Tables are created at startup with Base.metadata.create_all(); there are no migrations.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import BigInteger, Date, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class HistoricalPrice(Base):
    """One daily bar as Yahoo served it. Prices are in the ticker's currency, unadjusted
    except adj_close, which is back-adjusted for splits and dividends."""

    __tablename__ = "historical_prices"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int | None] = mapped_column(BigInteger)  # split-adjusted volumes can exceed 2^31


class TickerCoverage(Base):
    """The one contiguous date range requested for a ticker so far (§4.2).

    It records what was requested, not the dates Yahoo returned, so a ticker that listed
    after the requested start is not refetched on every request.
    """

    __tablename__ = "ticker_coverage"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    cov_start: Mapped[dt.date] = mapped_column(Date)
    cov_end: Mapped[dt.date] = mapped_column(Date)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))  # UTC


class BacktestRun(Base):
    """Every successful /api/backtest call, with its resolved request and its metrics as JSON."""

    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))  # UTC
    engine_version: Mapped[str] = mapped_column(String(32))
    request_json: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[str] = mapped_column(Text)
