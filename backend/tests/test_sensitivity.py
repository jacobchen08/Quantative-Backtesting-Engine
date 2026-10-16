"""Parameter sensitivity grid (§12, test_sensitivity.py). Tests pipeline.run_sensitivity directly."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.services.backtester import InsufficientDataError, run_backtest
from app.services.pipeline import StrategyParams, build_report, run_sensitivity, run_strategy

DEFAULT_REQUESTS = [
    StrategyParams(strategy="mean_reversion", tickers=("Y",), L=60),
    StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60),
    StrategyParams(strategy="trend", tickers=("SPY",), L=252),
]


def reference_sharpe(excess: np.ndarray, held: np.ndarray) -> float | None:
    """§8 Sharpe, written out independently of metrics.py."""
    if len(excess) < 2 or not held.any():
        return None
    sd = float(np.std(excess, ddof=1))
    if sd < 1e-12:
        return None
    return float(np.mean(excess)) / sd * math.sqrt(252)


def config(params: StrategyParams, L: int, z: float | None, split: pd.Timestamp) -> StrategyParams:
    if z is None:
        return replace(params, L=L, split_date=split)
    return replace(params, L=L, z_entry=z, split_date=split)


@pytest.mark.parametrize("params", DEFAULT_REQUESTS, ids=lambda p: p.strategy)
def test_default_grids_run_with_their_own_t0(
    prices: pd.DataFrame, rf: pd.Series, params: StrategyParams
) -> None:
    res = run_sensitivity(prices, rf, params)
    trend = params.strategy == "trend"
    assert res.n_configs_tested == (5 if trend else 20)
    assert res.z_entry_values is None if trend else res.z_entry_values == (1.5, 2.0, 2.5, 3.0)
    assert len(res.sharpe) == 5 and all(len(row) == (1 if trend else 4) for row in res.sharpe)
    assert res.selected == ((252, None) if trend else (60, 2.0))
    assert sum(v is not None for row in res.sharpe for v in row) > 0


def test_shorter_lookbacks_signal_before_the_grids_latest_t0(prices: pd.DataFrame, rf: pd.Series) -> None:
    params = StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60)
    split = run_strategy(prices, rf, params).split_date
    short = run_strategy(prices, rf, config(params, 20, 2.0, split))
    long_ = run_strategy(prices, rf, config(params, 120, 2.0, split))
    assert short.t0 < long_.t0 and (short.signal.iloc[: long_.t0] != 0).any()
    # Forcing the grid's latest t0 on the short lookback trips the §6.1 check, hence one t0 per config.
    with pytest.raises(ValueError, match="before t0"):
        run_backtest(
            R=short.returns[["Y", "X"]],
            W=short.weights,
            s=short.signal,
            rf=rf,
            t0=long_.t0,
            k=1,
            cost_bps=10.0,
            E0=1.0,
            split_date=split,
        )


@pytest.mark.parametrize("params", DEFAULT_REQUESTS, ids=lambda p: p.strategy)
def test_one_split_and_the_common_window(prices: pd.DataFrame, rf: pd.Series, params: StrategyParams) -> None:
    res = run_sensitivity(prices, rf, params)
    split = run_strategy(prices, rf, params).split_date  # the request's own resolved split
    assert res.window_end == split
    runs = [
        run_strategy(prices, rf, config(params, L, z, split))
        for L in res.L_values
        for z in res.z_entry_values or (None,)
    ]
    assert all(r.split_date == split for r in runs)
    assert res.window_start == prices.index[max(r.backtest.e for r in runs)]
    # Without the common split, the shortest lookback would have picked its own 70% split.
    assert run_strategy(prices, rf, replace(params, L=res.L_values[0])).split_date != split


@pytest.mark.parametrize("params", DEFAULT_REQUESTS, ids=lambda p: p.strategy)
def test_each_cell_is_the_sharpe_on_the_common_window(
    prices: pd.DataFrame, rf: pd.Series, params: StrategyParams
) -> None:
    res = run_sensitivity(prices, rf, params)
    start, end = prices.index.get_loc(res.window_start), prices.index.get_loc(res.window_end)
    assert end - start >= 252
    for i, L in enumerate(res.L_values):
        for j, z in enumerate(res.z_entry_values or (None,)):
            bt = run_strategy(prices, rf, config(params, L, z, res.window_end)).backtest
            excess = bt.excess.iloc[start:end].to_numpy()
            held = bt.held_id.iloc[start:end].notna().to_numpy()
            want, got = reference_sharpe(excess, held), res.sharpe[i][j]
            assert (got is None) if want is None else got == pytest.approx(want, rel=1e-12), (L, z)


def test_a_configuration_without_exposure_gives_a_null_cell(prices: pd.DataFrame, rf: pd.Series) -> None:
    params = StrategyParams(strategy="mean_reversion", tickers=("Y",), L=60)
    res = run_sensitivity(prices, rf, params, z_entry_values=(5.0,))  # calm data: |z| never exceeds 5
    assert res.sharpe == [[None]] * 5
    assert res.selected is None  # the request's own z_entry = 2.0 isn't in this grid


@pytest.mark.parametrize("offset", [0, 61, 122, 183, 244])
def test_the_trend_grid_needs_more_history_than_one_backtest(
    prices: pd.DataFrame, rf: pd.Series, offset: int
) -> None:
    window = slice(offset, offset + 756)  # about 3 years of trading days
    p, r = prices.iloc[window], rf.iloc[window]
    params = StrategyParams(strategy="trend", tickers=("SPY",), L=252)
    m = build_report(run_strategy(p, r, params)).metrics
    assert m["is"].n_days >= 252 and m["oos"].n_days >= 126
    with pytest.raises(InsufficientDataError, match=r"L = 378 leaves only \d+ in-sample days") as err:
        run_sensitivity(p, r, params)
    assert err.value.n_is < 252 and "drop L = 378 from the grid" in str(err.value)
