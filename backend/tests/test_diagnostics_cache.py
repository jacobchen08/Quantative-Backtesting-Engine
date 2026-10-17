"""The diagnostics cache (§14): the same result as stats.pairs_diagnostics, keyed on its inputs, LRU."""

from __future__ import annotations

import pandas as pd

from app.diagnostics_cache import MAX_ENTRIES, DiagnosticsCache
from app.services import stats


def test_a_hit_returns_what_a_fresh_computation_gives(prices: pd.DataFrame) -> None:
    cache = DiagnosticsCache()
    split = prices.index[700]
    first = cache.get(prices["Y"], prices["X"], split)
    again = cache.get(prices["Y"].copy(), prices["X"].copy(), split)  # equal data in new objects
    assert again is first
    assert (cache.hits, cache.misses, len(cache)) == (1, 1, 1)
    assert first == stats.pairs_diagnostics(prices["Y"], prices["X"], split)


def test_any_change_to_the_inputs_misses(prices: pd.DataFrame) -> None:
    cache = DiagnosticsCache()
    y, x, split = prices["Y"], prices["X"], prices.index[700]
    cache.get(y, x, split)
    bumped = y.copy()
    bumped.iloc[10] *= 1.0001  # one adjusted close, e.g. after a refetch on a new adjustment basis
    dates = prices.index.to_series()
    dates.iloc[300] += pd.Timedelta(hours=12)  # same values and endpoints, different dates
    variants = [
        (bumped, x, split),
        (y, x, prices.index[701]),  # the split date
        (y.iloc[1:], x.iloc[1:], split),  # the IS window
        (y.iloc[:-1], x.iloc[:-1], split),  # the OOS window
        (y.set_axis(pd.DatetimeIndex(dates)), x.set_axis(pd.DatetimeIndex(dates)), split),
        (y.rename("A"), x, split),  # the tickers
    ]
    for args in variants:
        cache.get(*args)
    assert (cache.hits, cache.misses) == (0, 1 + len(variants))


def test_the_least_recently_used_entry_is_evicted(prices: pd.DataFrame) -> None:
    cache = DiagnosticsCache(max_entries=2)
    y, x = prices["Y"], prices["X"]
    a, b, c = prices.index[[600, 650, 700]]
    cache.get(y, x, a)
    cache.get(y, x, b)
    cache.get(y, x, a)  # a is now the most recently used
    cache.get(y, x, c)  # evicts b
    assert len(cache) == 2
    cache.get(y, x, a)
    cache.get(y, x, c)
    assert (cache.hits, cache.misses) == (3, 3)
    cache.get(y, x, b)  # evicted, so computed again
    assert (cache.hits, cache.misses) == (3, 4)


def test_the_default_size_is_256_entries() -> None:
    assert MAX_ENTRIES == 256 and DiagnosticsCache().max_entries == 256
