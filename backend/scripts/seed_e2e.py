r"""Write the offline database the browser tests use (§12, "Browser tests").

    .venv\Scripts\python.exe scripts\seed_e2e.py PATH\TO\e2e.db

Deletes PATH if it exists, creates every table, and caches the synthetic market from
tests/synthetic.py for 2012-01-03 to 2019-12-31:
- SYNY and SYNX, a cointegrated pair (ln SYNY = 0.3 + 1.2 ln SYNX + an AR(1));
- SYNT, a trending asset with 15% annual vol;
- SPY (the benchmark), and ^IRX held at 2.0%, from 2011-12-01 so the first day has an earlier rate.

The rows and coverage are written by the app's own data_loader.ensure_cached, with the network
seam replaced by a function that serves these frames, so the database is exactly what the API
would have cached. Run the API on it with QUANT_OFFLINE=1: any other ticker or range is then a
503 instead of a download.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))  # run as a script from backend\, so app and tests are importable

from sqlalchemy.orm import Session  # noqa: E402

from app.database import Base, make_engine  # noqa: E402
from app.models import HistoricalPrice  # noqa: E402, F401  (registers the tables on Base)
from app.services import data_loader  # noqa: E402
from tests.synthetic import IRX_PERCENT, make_prices, make_universe, yf_layout  # noqa: E402

START = dt.date(2012, 1, 3)
END = dt.date(2019, 12, 31)
IRX_START = dt.date(2011, 12, 1)  # more than the 14-day ^IRX buffer before START (§4.5)


def synthetic_frames() -> dict[str, pd.DataFrame]:
    """Every seeded ticker's history in yfinance layout."""
    index = pd.bdate_range(START, END).as_unit("ns")
    market = make_prices(n=len(index), start=str(START))  # Y, X and SPY on the same calendar
    trend = make_universe(index, drift=0.0003)["T15"]  # 15% vol, drifting up about 7.5% a year
    return {
        "SYNY": yf_layout(market["Y"], seed=1),
        "SYNX": yf_layout(market["X"], seed=2),
        "SYNT": yf_layout(trend, seed=3),
        "SPY": yf_layout(market["SPY"], seed=4),
        "^IRX": yf_layout(pd.Series(IRX_PERCENT, index=pd.bdate_range(IRX_START, END))),
    }


def seed(path: Path) -> dict[str, int]:
    """Create the database at path and cache every synthetic ticker. Returns the row counts."""
    path.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = make_engine(f"sqlite:///{path.as_posix()}")
    Base.metadata.create_all(engine)
    frames = synthetic_frames()

    def serve(ticker: str, start: dt.date, end_inclusive: dt.date) -> pd.DataFrame:
        frame = frames[ticker]
        return frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end_inclusive))]

    # The only network seam (§4.3), replaced: nothing is downloaded. Offline mode is switched off
    # for the seed itself, so it also works when QUANT_OFFLINE=1 is set for the API it feeds.
    original = data_loader.fetch_from_yahoo, data_loader.offline_mode
    data_loader.fetch_from_yahoo, data_loader.offline_mode = serve, lambda: False
    try:
        with Session(engine) as session:
            for ticker in frames:
                start = IRX_START if ticker == data_loader.RATE_TICKER else START
                data_loader.ensure_cached(session, ticker, start, END)
    finally:
        data_loader.fetch_from_yahoo, data_loader.offline_mode = original
        engine.dispose()
    return {ticker: len(frame) for ticker, frame in frames.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="the SQLite file to (re)create")
    args = parser.parse_args()
    counts = seed(args.path.resolve())
    print(f"seeded {args.path}: " + ", ".join(f"{t} {n} rows" for t, n in counts.items()))


if __name__ == "__main__":
    main()
