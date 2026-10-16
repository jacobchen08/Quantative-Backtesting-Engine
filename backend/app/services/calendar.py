"""today_ny() (the injectable clock) and alignment helpers (§4.2, §4.5).

Call the clock as calendar.today_ny() so tests can monkeypatch it.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")


def today_ny() -> dt.date:
    """Today's date in America/New_York. A bar is final only if it is dated before this."""
    return dt.datetime.now(NEW_YORK).date()


def last_final_day(end: dt.date) -> dt.date:
    """end* = min(end, today_ny() - 1 day): the last date a request can need (§4.2)."""
    return min(end, today_ny() - dt.timedelta(days=1))


def to_ns_index(index: object) -> pd.DatetimeIndex:
    """Normalize dates to tz-naive midnight timestamps at datetime64[ns] (§4.5).

    yfinance 1.7 returns datetime64[s], and pandas 3 refuses some operations across
    resolutions (merge_asof raises MergeError on [s] vs [ns] keys).
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # keep the exchange's calendar date
    return idx.normalize().as_unit("ns")


def align_closes(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """The run calendar: an inner join of every ticker's adjusted closes on date (§4.5).

    Prices are never forward-filled; a stale price would create a fake zero-return day.
    """
    return pd.concat(closes, axis=1, join="inner").sort_index()


def rf_from_irx(irx: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free rate on the run calendar from cleaned ^IRX closes (§4.5).

    rf_t = (the last ^IRX close dated strictly before t) / 100 / 252, a decimal per day.
    This as-of lookup forward-fills the rate, which is fine for a rate; it never back-fills,
    so days before the first observation stay NaN for the caller to reject.
    """
    left = pd.DataFrame({"date": calendar})
    right = irx.rename("irx").rename_axis("date").reset_index()
    asof = pd.merge_asof(left, right, on="date", allow_exact_matches=False)["irx"]
    return pd.Series(asof.to_numpy() / 100 / 252, index=calendar, name="rf")
