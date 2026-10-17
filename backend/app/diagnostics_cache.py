"""LRU cache of the pairs diagnostics for /api/backtest (§14).

The two Engle–Granger tests dominate a warm pairs backtest. Their inputs are the two legs'
adjusted closes and the split date, not z_entry, z_exit, costs or the hedge mode, so a request
that changes only those reuses the cached result. Under the default 70% split the split date
moves with L and execution_lag (they move t0), so those changes miss unless split_date is set.

The cache lives in the API layer; stats.py stays pure (§0, rule 8).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd

from app.services import stats
from app.services.stats import PairsDiagnostics

MAX_ENTRIES = 256

Key = tuple[str, str, pd.Timestamp, pd.Timestamp, pd.Timestamp, str]


def diagnostics_key(y_close: pd.Series, x_close: pd.Series, split_date: pd.Timestamp) -> Key:
    """(Y, X, first date, split date, last date, SHA-256 of the dates and both adjusted closes).

    The IS window is [first date, split_date) and the OOS window [split_date, last date].
    Together they cover every price the diagnostics read, so one digest covers both windows.
    The dates are hashed too, which pins how many days fall on each side of the split.
    """
    digest = hashlib.sha256(np.ascontiguousarray(y_close.index.asi8).tobytes())
    for close in (y_close, x_close):
        digest.update(np.ascontiguousarray(close.to_numpy(dtype=np.float64)).tobytes())
    index = y_close.index
    return (
        str(y_close.name),
        str(x_close.name),
        index[0],
        pd.Timestamp(split_date),
        index[-1],
        digest.hexdigest(),
    )


class DiagnosticsCache:
    """Thread-safe LRU of stats.pairs_diagnostics results, evicting the least recently used.

    get() has the same signature and result as stats.pairs_diagnostics. FastAPI runs the
    handlers in a thread pool, so the entries sit behind a lock; the computation itself runs
    outside it, and two simultaneous misses on one key both compute the same immutable result.
    """

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self._entries: OrderedDict[Key, PairsDiagnostics] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, y_close: pd.Series, x_close: pd.Series, split_date: pd.Timestamp) -> PairsDiagnostics:
        key = diagnostics_key(y_close, x_close, split_date)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return cached
            self.misses += 1
        result = stats.pairs_diagnostics(y_close, x_close, split_date)
        with self._lock:
            self._entries[key] = result
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return result

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)
