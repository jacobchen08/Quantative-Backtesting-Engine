r"""Record the static demo's responses (CLAUDE.md §13, "Static demo").

    .venv\Scripts\python.exe scripts\record_demo.py
    .venv\Scripts\python.exe scripts\record_demo.py --db quant.db --out ..\frontend\public\demo

For each scenario in SCENARIOS this sends the four requests the dashboard sends for it (the
backtest, the sensitivity grid, the Monte Carlo with seed 42 and the walk-forward) to the API in
process through FastAPI's TestClient, and writes each response body byte for byte, with
manifest.json mapping every request to its file.

Prices come from the local cache (--db, by default the app's quant.db). A ticker or range that
isn't cached is downloaded into it first, as the API itself would (set QUANT_OFFLINE=1 to forbid
that). The requests then run on a temporary copy of the cache with the run history emptied, so
the recording never adds runs to the app's history and the recorded run ids are 1, 2, 3, ...

The manifest's keys are the requests' JSON with sorted keys. The dashboard parses each key and
re-serializes it the same way, so the two sides agree however each prints its numbers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sqlite3
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))  # run as a script from backend\, so app is importable

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import Engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from app.database import Base, get_session, make_engine  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import BacktestRequest  # noqa: E402
from app.services import data_loader  # noqa: E402

DEFAULT_OUT = BACKEND.parent / "frontend" / "public" / "demo"
SIZE_LIMIT = 10_000_000  # §13: the recorded data stays under 10 MB

# Fixed dates (§13): whole calendar years, so a re-recording on a later day sends the same requests.
START = "2015-01-01"
END = "2025-12-31"
MC_SEED = 42

# The dashboard's defaults (frontend/src/form.ts, defaultForm). The requests below must be exactly
# what its request builder sends for each scenario; frontend/src/demo.test.ts checks that.
COMMON: dict[str, Any] = {
    "start": START,
    "end": END,
    "cost_bps": 2,
    "borrow_bps": 0,
    "execution_lag": 1,
    "risk_free": "tbill",
}
VOL_DEFAULTS: dict[str, Any] = {"vol_halflife": 42, "vol_refresh": "weekly", "max_leverage": 2}


def pairs(y: str, x: str) -> dict[str, Any]:
    return {
        "strategy": "pairs",
        "tickers": [y, x],
        "L": 60,
        "z_entry": 2.0,
        "z_exit": 0.5,
        "hedge_mode": "fixed_at_entry",
        "pairs_model": "ols",
        **COMMON,
    }


def trend(ticker: str, vol_target: float | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "strategy": "trend",
        "tickers": [ticker],
        "L": 252,
        "signal_frequency": "monthly",
        "allow_short": True,
        **COMMON,
    }
    if vol_target is not None:
        body |= {"vol_target": vol_target, **VOL_DEFAULTS}
    return body


SCENARIOS: list[dict[str, Any]] = [
    {"id": "pairs-ko-pep", "label": "Pairs: KO / PEP", "backtest": pairs("KO", "PEP")},
    {"id": "pairs-xom-cvx", "label": "Pairs: XOM / CVX", "backtest": pairs("XOM", "CVX")},
    {"id": "pairs-googl-goog", "label": "Pairs: GOOGL / GOOG", "backtest": pairs("GOOGL", "GOOG")},
    {"id": "pairs-nvda-amd", "label": "Pairs: NVDA / AMD", "backtest": pairs("NVDA", "AMD")},
    {
        "id": "mean-reversion-ko",
        "label": "Mean reversion: KO",
        "backtest": {
            "strategy": "mean_reversion",
            "tickers": ["KO"],
            "L": 60,
            "z_entry": 2.0,
            "z_exit": 0.5,
            "allow_short": True,
            **COMMON,
        },
    },
    {"id": "trend-spy", "label": "Trend: SPY", "backtest": trend("SPY", None)},
    {"id": "trend-spy-vol", "label": "Trend: SPY, vol target 10%", "backtest": trend("SPY", 0.1)},
    {
        "id": "trend-portfolio-etfs",
        "label": "Trend portfolio: SPY, QQQ, GLD, TLT",
        "backtest": {
            "strategy": "trend_portfolio",
            "tickers": ["SPY", "QQQ", "GLD", "TLT"],
            "L": 252,
            "signal_frequency": "monthly",
            "allow_short": True,
            **COMMON,
            "vol_target": 0.4,
            **VOL_DEFAULTS,
        },
    },
]


def requests_for(backtest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The four requests the dashboard sends for one backtest, by endpoint name."""
    return {
        "backtest": backtest,
        "sensitivity": backtest,  # the default grid (§9), as the dashboard sends it
        "monte_carlo": {
            "source": "strategy",
            "backtest": backtest,
            "n_paths": 10000,
            "horizon": 252,
            "block_len": 10,
            "zero_drift": False,
            "seed": MC_SEED,
        },
        "walk_forward": {**backtest, "train_days": 756, "test_days": 63, "anchored": False},
    }


PATHS = {
    "backtest": "/api/backtest",
    "sensitivity": "/api/sensitivity",
    "monte_carlo": "/api/monte-carlo",
    "walk_forward": "/api/walk-forward",
}


def request_key(body: dict[str, Any]) -> str:
    """The manifest key (§13): the request's JSON with sorted keys."""
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)


def warm_cache(db: Path) -> None:
    """Cache every scenario's prices in db, downloading what's missing as the API would."""
    engine = make_engine(f"sqlite:///{db.as_posix()}")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            for scenario in SCENARIOS:
                request = BacktestRequest.model_validate(scenario["backtest"])
                data_loader.load_market(
                    session, request.tickers, request.start, request.end, request.risk_free
                )
    finally:
        engine.dispose()


def scratch_copy(db: Path, into: Path) -> Path:
    """A copy of db with the run history emptied, so the recorded run ids start at 1."""
    copy = into / "record_demo.db"
    with sqlite3.connect(db) as source, sqlite3.connect(copy) as target:
        source.backup(target)
        target.execute("DELETE FROM backtest_runs")
    return copy


def client_on(engine: Engine) -> TestClient:
    """A TestClient whose requests use engine. The app's lifespan (create_all on quant.db) doesn't run."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override() -> Iterator[Session]:
        with factory() as session:
            yield session

    app.dependency_overrides[get_session] = override
    return TestClient(app)


def record(client: TestClient, out: Path) -> dict[str, Any]:
    """Send every scenario's requests, write the response files and return the manifest."""
    responses: dict[str, dict[str, str]] = {endpoint: {} for endpoint in PATHS}
    scenarios = []
    data_end = ""
    engine_version = ""
    for scenario in SCENARIOS:
        bodies = requests_for(scenario["backtest"])
        for endpoint, body in bodies.items():
            response = client.post(PATHS[endpoint], json=body)
            if response.status_code != 200:
                where = f"{scenario['id']}: {PATHS[endpoint]}"
                raise SystemExit(f"{where} returned HTTP {response.status_code}: {response.text}")
            name = f"{scenario['id']}-{endpoint.replace('_', '-')}.json"
            (out / name).write_bytes(response.content)
            responses[endpoint][request_key(body)] = name
            if endpoint == "backtest":
                payload = response.json()
                data_end = max(data_end, payload["series"]["dates"][-1])
                engine_version = payload["engine_version"]
        scenarios.append({"id": scenario["id"], "label": scenario["label"], **bodies})
        print(f"  {scenario['id']}", file=sys.stderr)
    return {
        "recorded_on": dt.date.today().isoformat(),
        "data_end": data_end,
        "engine_version": engine_version,
        "scenarios": scenarios,
        "responses": responses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db", type=Path, default=BACKEND / "quant.db", help="the SQLite price cache (default: quant.db)"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="output folder (default: frontend/public/demo)"
    )
    args = parser.parse_args()
    for name in ("httpx", "httpx2"):  # the TestClient's HTTP library logs every request at INFO
        logging.getLogger(name).setLevel(logging.WARNING)

    print(f"Caching prices in {args.db}...", file=sys.stderr)
    warm_cache(args.db)
    if args.out.exists():
        shutil.rmtree(args.out)  # only recorded files live here, so a dropped scenario leaves nothing behind
    args.out.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        engine = make_engine(f"sqlite:///{scratch_copy(args.db, Path(tmp)).as_posix()}")
        client = client_on(engine)  # not entered with "with", which would run the lifespan
        try:
            print("Recording...", file=sys.stderr)
            manifest = record(client, args.out)
        finally:
            client.close()
            app.dependency_overrides.clear()
            engine.dispose()  # or Windows can't delete the temporary folder
    manifest_text = json.dumps(manifest, indent=1, allow_nan=False) + "\n"
    (args.out / "manifest.json").write_text(manifest_text, encoding="utf-8")

    files = sorted(args.out.iterdir())
    total = sum(f.stat().st_size for f in files)
    largest = max(files, key=lambda f: f.stat().st_size)
    print(f"Wrote {len(files)} files to {args.out}: {total / 1e6:.2f} MB (limit {SIZE_LIMIT / 1e6:.2f} MB).")
    print(f"Largest: {largest.name}, {largest.stat().st_size / 1e6:.2f} MB.")
    print(f"Recorded on {manifest['recorded_on']}, data through {manifest['data_end']}.")
    print(f"Engine {manifest['engine_version']}.")
    if total >= SIZE_LIMIT:
        raise SystemExit("Over the 10 MB limit (§13): ask before trimming scenarios.")


if __name__ == "__main__":
    main()
