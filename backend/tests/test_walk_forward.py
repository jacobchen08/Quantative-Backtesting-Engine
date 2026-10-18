"""Walk-forward analysis (§12, test_walk_forward.py): folds, causal selection, stitching, fold
ownership, the fixed-grid identity, flat folds, ruin and too little data.

These call pipeline.run_walk_forward directly, apart from the last test, which goes through the
API. Small grids keep them fast; the protocol is the same for any grid.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.services import metrics
from app.services.backtester import InsufficientDataError
from app.services.pipeline import (
    FoldWindow,
    StrategyParams,
    WalkForwardResult,
    run_strategy,
    run_walk_forward,
    select_configuration,
    walk_forward_folds,
)

PAIRS = StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60, cost_bps=10.0, borrow_bps=50.0)
PAIRS_GRID = {"L_values": (20, 60, 120), "z_entry_values": (1.5, 2.5)}  # 10 folds, 3 distinct choices
MEAN_REVERSION = StrategyParams(strategy="mean_reversion", tickers=("Y",), L=20)
MR_GRID = {"L_values": (20, 40), "z_entry_values": (1.5, 2.0)}  # 11 folds, 5 changes of selection
WINDOWS = {"train_days": 252, "test_days": 63}


def reference_sharpe(excess: np.ndarray, held: np.ndarray) -> float | None:
    """§8 Sharpe, written out independently of metrics.py."""
    if len(excess) < 2 or not held.any():
        return None
    sd = float(np.std(excess, ddof=1))
    if sd < 1e-12:
        return None
    return float(np.mean(excess)) / sd * math.sqrt(252)


def same_value(got: float | None, want: float | None) -> bool:
    return got is None if want is None else got == pytest.approx(want, rel=1e-12, abs=1e-15)


def positions(res: WalkForwardResult, calendar: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """Each fold's test window [b_j, end_j) as calendar positions."""
    return [(calendar.get_loc(f.test_start), calendar.get_loc(f.test_end) + 1) for f in res.folds]


def executed_weights(
    prices: pd.DataFrame, rf: pd.Series, params: StrategyParams, cells: set[tuple[int, float | None]]
) -> dict[tuple[int, float | None], np.ndarray]:
    """Grid configurations' executed weights W.shift(k), each from its own run (§6.1)."""
    split = run_strategy(prices, rf, params).split_date
    out = {}
    for L, z in cells:
        cfg = (
            replace(params, L=L, split_date=split)
            if z is None
            else replace(params, L=L, z_entry=z, split_date=split)
        )
        out[(L, z)] = run_strategy(prices, rf, cfg).weights.shift(params.execution_lag).fillna(0.0).to_numpy()
    return out


def reference_excess(
    w: np.ndarray, prices: pd.DataFrame, legs: list[str], rf: pd.Series, params: StrategyParams
) -> tuple[np.ndarray, np.ndarray]:
    """§6.1 applied to executed weights w, written out in numpy: (excess, turnover)."""
    R = (prices[legs] / prices[legs].shift(1) - 1).to_numpy(copy=True)  # pandas 3 may return a read-only view
    R[0] = 0.0
    w_prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    gross = np.sum(w_prev * (R - rf.to_numpy()[:, None]), axis=1)
    borrow = params.borrow_bps / 10_000 / 252 * np.sum(np.clip(-w_prev, 0.0, None), axis=1)
    turnover = np.sum(np.abs(w - w_prev), axis=1)
    return gross - borrow - params.cost_bps / 10_000 * turnover, turnover


def perturbed(prices: pd.DataFrame, rf: pd.Series, cut: int, rng: np.random.Generator) -> tuple:
    """Prices and rf multiplied by lognormal noise strictly after position cut."""
    future = slice(cut + 1, None)
    noisy = prices.copy()
    noisy.iloc[future] *= np.exp(rng.normal(0.0, 0.05, noisy.iloc[future].shape))
    noisy_rf = rf.copy()
    noisy_rf.iloc[future] *= np.exp(rng.normal(0.0, 0.2, len(rf) - cut - 1))
    return noisy, noisy_rf


# ---------------------------------------------------------------- folds


def test_rolling_and_anchored_fold_boundaries() -> None:
    # n = 1000, s = 120, T = 252, S = 63: tests start at 372, 435, ..., 939; the last is cut at 1000.
    rolling = walk_forward_folds(1000, 120, 252, 63, anchored=False)
    anchored = walk_forward_folds(1000, 120, 252, 63, anchored=True)
    assert [f.test_start for f in rolling] == [372, 435, 498, 561, 624, 687, 750, 813, 876, 939]
    assert [f.test_end for f in rolling] == [435, 498, 561, 624, 687, 750, 813, 876, 939, 1000]
    assert [f.train_start for f in rolling] == [120, 183, 246, 309, 372, 435, 498, 561, 624, 687]
    assert all(f.test_start - f.train_start == 252 for f in rolling)  # a T-day rolling window
    assert [f.train_start for f in anchored] == [120] * 10  # every window starts at s
    assert [(f.test_start, f.test_end) for f in anchored] == [(f.test_start, f.test_end) for f in rolling]


@pytest.mark.parametrize("n, last", [(960, FoldWindow(687, 939, 960)), (959, FoldWindow(624, 876, 939))])
def test_a_final_test_window_under_21_days_is_dropped(n: int, last: FoldWindow) -> None:
    folds = walk_forward_folds(n, 120, 252, 63, anchored=False)
    assert folds[-1] == last  # 21 days left at n = 960 is a fold; 20 at n = 959 is dropped
    assert len(folds) == (10 if n == 960 else 9)


@pytest.mark.parametrize("anchored", [False, True])
def test_the_folds_start_at_the_grids_common_start(
    prices: pd.DataFrame, rf: pd.Series, anchored: bool
) -> None:
    res = run_walk_forward(prices, rf, PAIRS, **PAIRS_GRID, **WINDOWS, anchored=anchored)
    # The longest lookback, L = 120, has its first z on day L - 1 = 119 (§5.3), so with a one-day lag
    # the last configuration to start evaluating does so on day s = 120 (§6.2).
    s = 120
    cal = prices.index
    want = walk_forward_folds(len(cal), s, 252, 63, anchored)
    assert len(res.folds) == len(want) >= 2 and res.n_configs == 6
    for fold, fw in zip(res.folds, want, strict=True):
        assert fold.train_start == cal[fw.train_start] and fold.train_end == cal[fw.test_start - 1]
        assert fold.test_start == cal[fw.test_start] and fold.test_end == cal[fw.test_end - 1]
    assert res.e == want[0].test_start - 1  # day b0 - 1, the first fold's entry day


@pytest.mark.parametrize("n_days, n_folds", [(800, 0), (950, 1)])
def test_fewer_than_two_folds_is_an_error_with_the_counts(
    prices: pd.DataFrame, rf: pd.Series, n_days: int, n_folds: int
) -> None:
    # s = 120 (L = 120). The default T = 756 puts b0 at 876: past the end of 800 days, and at 950 days
    # one 63-day fold is followed by an 11-day window, which is dropped.
    p, r = prices.iloc[:n_days], rf.iloc[:n_days]
    with pytest.raises(InsufficientDataError) as err:
        run_walk_forward(p, r, PAIRS, **PAIRS_GRID)
    message = str(err.value)
    assert f"the {n_days - 120} trading days" in message
    assert f"make {n_folds} fold(s) of 756 training and 63 test days" in message and "at least 2" in message


# ---------------------------------------------------------------- causal selection


def test_perturbing_prices_after_a_training_window_never_changes_that_folds_selection(
    prices: pd.DataFrame, varying_rf
) -> None:
    rf = varying_rf(prices.index)
    base = run_walk_forward(prices, rf, MEAN_REVERSION, **MR_GRID, **WINDOWS)
    assert base.n_distinct_selected >= 2  # the selection does depend on the data
    rng = np.random.default_rng(7)
    for j, (b, _) in enumerate(positions(base, prices.index)):
        cut = b - 1  # the training window's last close
        noisy, noisy_rf = perturbed(prices, rf, cut, rng)
        pert = run_walk_forward(noisy, noisy_rf, MEAN_REVERSION, **MR_GRID, **WINDOWS)
        for i in range(j + 1):  # this fold and every earlier one
            assert pert.folds[i].selected == base.folds[i].selected, (j, i)
            assert pert.folds[i].train_sharpe == base.folds[i].train_sharpe, (j, i)
        after_a, after_b = base.daily.excess.to_numpy()[b:], pert.daily.excess.to_numpy()[b:]
        assert not np.array_equal(after_a, after_b)  # the perturbation did reach the later days


def test_the_selection_rules() -> None:
    """§9.2 step 3 on crafted inputs: only each run's e, ruined and stop matter."""
    a, b = (20, 1.5), (60, 2.0)  # grid order: a first
    g = np.random.default_rng(5)
    good = g.normal(0.0005, 0.01, 300)
    better = good + 0.0005  # the same spread and a higher mean: a higher Sharpe
    live = (0, False, 300)

    def select(spec: dict, excess: dict, held: dict | None = None) -> tuple:
        runs = {
            cell: SimpleNamespace(backtest=SimpleNamespace(e=e, ruined=r, stop=s))
            for cell, (e, r, s) in spec.items()
        }
        held = held or {cell: np.ones(300, dtype=bool) for cell in spec}
        return select_configuration(runs, excess, held, 100, 250)  # the training window is days 100-249

    cell, sharpe = select({a: live, b: live}, {a: good, b: better})
    assert cell == b and sharpe == pytest.approx(
        reference_sharpe(better[100:250], np.ones(150, bool)), rel=1e-12
    )
    assert select({a: live, b: live}, {a: better, b: better.copy()})[0] == a  # a tie goes to grid order
    # Ruined on the window's last day (stop = 250) is out; on the first test day (stop = 251) it is in.
    assert select({a: (0, True, 250), b: live}, {a: better, b: good})[0] == b
    assert select({a: (0, True, 251), b: live}, {a: better, b: good})[0] == a
    assert select({a: (101, False, 300), b: live}, {a: better, b: good})[0] == b  # not evaluating on day 100
    no_position = {a: np.zeros(300, dtype=bool), b: np.ones(300, dtype=bool)}
    assert (
        select({a: live, b: live}, {a: better, b: good}, held=no_position)[0] == b
    )  # no exposure, no Sharpe
    assert select({a: live, b: live}, {a: np.full(300, 0.001), b: good})[0] == b  # constant: no Sharpe
    assert select({a: (0, True, 200)}, {a: better}) == (None, None)  # nothing eligible: a flat fold
    # Only the window is read: changing the days from 250 on changes nothing.
    later = better.copy()
    later[250:] = -1.0
    assert select({a: live, b: live}, {a: good, b: later}) == select({a: live, b: live}, {a: good, b: better})


# ---------------------------------------------------------------- stitching


def test_the_stitched_weights_are_the_chosen_configurations_and_pay_the_switches(
    prices: pd.DataFrame, rf: pd.Series
) -> None:
    res = run_walk_forward(prices, rf, PAIRS, **PAIRS_GRID, **WINDOWS)
    legs = list(PAIRS.tickers)
    windows = positions(res, prices.index)
    w_star = res.weights.to_numpy()
    own = executed_weights(prices, rf, PAIRS, {f.selected for f in res.folds if f.selected is not None})
    w_of = [np.zeros_like(w_star) if f.selected is None else own[f.selected] for f in res.folds]
    assert (w_star[: windows[0][0] - 1] == 0.0).all()  # flat until the close before the first test window
    for j, (b, end) in enumerate(windows):
        last = windows[j + 1][0] - 1 if j + 1 < len(windows) else end
        span = slice(b - 1, last)  # from the close before the test window to the close before the next
        assert np.array_equal(w_star[span], w_of[j][span]), j

    excess, turnover = reference_excess(w_star, prices, legs, rf, PAIRS)
    win = slice(res.e, res.stop)
    assert np.max(np.abs(res.daily.excess.to_numpy()[win] - excess[win])) < 1e-12

    # A switch is one trade at the close b_j - 1: from the old choice's position into it to the new one.
    switches = 0
    for j in range(1, len(windows)):
        day = windows[j][0] - 1
        switch = np.abs(w_of[j][day] - w_of[j - 1][day - 1]).sum()
        assert turnover[day] == pytest.approx(switch, abs=1e-15)
        assert res.daily.cost.iloc[day] == pytest.approx(PAIRS.cost_bps / 10_000 * switch, abs=1e-18)
        switches += res.folds[j].selected != res.folds[j - 1].selected and switch > 0
    assert switches >= 1  # some switch did trade


def test_fold_ownership_partitions_the_stitched_days(prices: pd.DataFrame, rf: pd.Series) -> None:
    res = run_walk_forward(prices, rf, PAIRS, **PAIRS_GRID, **WINDOWS)
    cal, windows = prices.index, positions(res, prices.index)
    fold, dates = res.series["fold"].to_numpy(), res.series.index
    assert dates[0] == cal[res.e - 1] and np.isnan(fold[0])  # the baseline belongs to no fold
    assert dates[1] == cal[res.e] == cal[windows[0][0] - 1] and fold[1] == 0  # b0 - 1: fold 0's entry day
    x, held = res.daily.excess.to_numpy(), res.daily.gross_held.to_numpy() > 0
    summed = 0.0
    for j, (b, end) in enumerate(windows):
        lo = res.e if j == 0 else b
        assert (fold[(dates >= cal[lo]) & (dates <= cal[end - 1])] == j).all()
        assert same_value(res.folds[j].test_sharpe, reference_sharpe(x[lo:end], held[lo:end])), j
        summed += x[lo:end].sum()
        if j > 0:  # the switch into fold j trades at the close of b_j - 1, the last day of fold j - 1
            assert fold[dates.get_loc(cal[b - 1])] == j - 1
    assert abs(summed - x[res.e : res.stop].sum()) < 1e-12
    assert res.stop == windows[-1][1] and len(dates) == res.stop - res.e + 1


def test_the_outputs_cover_the_same_window(prices: pd.DataFrame, rf: pd.Series) -> None:
    # 950 days: the last fold ends on day 938, and the 11 days after it are dropped (§9.2), so the
    # walk-forward stops before the calendar does, and so must the fixed run and the benchmark.
    prices, rf = prices.iloc[:950], rf.iloc[:950]
    res = run_walk_forward(prices, rf, PAIRS, **PAIRS_GRID, **WINDOWS)
    assert res.stop == 939 and res.folds[-1].test_end == prices.index[938] == res.series.index[-1]
    own = run_strategy(prices, rf, PAIRS).backtest  # the request's own configuration, L 60 / z 2.0
    win, E0 = slice(res.e, res.stop), PAIRS.initial_capital
    spy = (prices["SPY"] / prices["SPY"].shift(1) - 1).to_numpy()[win]  # the pairs benchmark (§6.4)
    for column, total in (
        ("equity", res.daily.total.to_numpy()[win]),
        ("fixed_equity", own.total.to_numpy()[win]),
        ("benchmark_equity", spy),
    ):
        path = res.series[column].to_numpy()
        assert path[0] == E0 and np.allclose(path[1:], E0 * np.cumprod(1 + total), rtol=1e-13, atol=0), column

    m, held = res.metrics, res.daily.gross_held.to_numpy()[win] > 0  # a position held into the day
    assert m.n_days == res.stop - res.e and m.pct_days_in_market == held.mean() < 1
    assert same_value(m.sharpe, reference_sharpe(res.daily.excess.to_numpy()[win], held))
    assert (m.n_closed_trades, m.n_open_trades, m.win_rate, m.avg_holding_days) == (None, None, None, None)
    assert m.ann_turnover == pytest.approx(res.daily.turnover.iloc[win].mean() * 252, rel=1e-12)
    assert m.ann_borrow_drag > 0  # pairs hold a short leg, and the fee is 50 bps

    fixed = res.fixed_metrics
    want = reference_sharpe(own.excess.to_numpy()[win], own.held_id.iloc[win].notna().to_numpy())
    assert fixed.n_days == m.n_days and same_value(fixed.sharpe, want)
    entered = own.trades["entry_date"].between(prices.index[res.e], prices.index[res.stop - 1])
    assert fixed.n_closed_trades + fixed.n_open_trades == entered.sum() > 0
    assert res.benchmark_metrics.total_return == pytest.approx(np.prod(1 + spy) - 1, rel=1e-12)


# Daily trend on X is always long or short, so it also holds a position on the last day: the last
# fold's weights must run through the last close.
TREND = StrategyParams(strategy="trend", tickers=("X",), L=126, signal_frequency="daily")


@pytest.mark.parametrize(
    "params, grid", [(PAIRS, {"L_values": (60,), "z_entry_values": (2.0,)}), (TREND, {"L_values": (126,)})]
)
def test_a_one_configuration_grid_reproduces_that_configuration(
    prices: pd.DataFrame, rf: pd.Series, params: StrategyParams, grid: dict
) -> None:
    own = run_strategy(prices, rf, params).backtest  # the request's own configuration, the grid's only one
    # A training length that puts day b0 - 1 = s + T - 1 inside one of the configuration's trades,
    # so that the stitched run, which enters from flat, really differs from it on that day.
    T = next(T for T in range(252, 400) if own.gross_held.iloc[own.e + T - 1] > 0)
    res = run_walk_forward(prices, rf, params, **grid, train_days=T, test_days=63)
    cell = (params.L, None if params.strategy == "trend" else params.z_entry)
    assert all(f.selected == cell for f in res.folds) and res.n_distinct_selected == 1
    assert res.e == own.e + T - 1
    b0 = res.e + 1
    assert np.array_equal(res.weights.to_numpy()[res.e : res.stop], own.w.to_numpy()[res.e : res.stop])
    assert np.array_equal(res.daily.excess.to_numpy()[b0 : res.stop], own.excess.to_numpy()[b0 : res.stop])
    if params is TREND:
        assert res.stop == len(prices) and (own.w.iloc[-2:] != 0).all().all()  # held through the last close
    # Day b0 - 1: the stitched run holds nothing into it, so it pays only the cost of entering.
    entry = params.cost_bps / 10_000 * np.abs(own.w.to_numpy()[res.e]).sum()
    assert res.daily.excess.iloc[res.e] == pytest.approx(-entry, abs=1e-18)
    assert res.daily.gross_excess.iloc[res.e] == 0.0 and res.daily.borrow.iloc[res.e] == 0.0
    assert res.daily.excess.iloc[res.e] != own.excess.iloc[res.e]


# ---------------------------------------------------------------- flat folds and ruin


def test_a_fold_with_no_eligible_configuration_is_flat(prices: pd.DataFrame, rf: pd.Series) -> None:
    # 360 stale days on Y: once a window is inside them its variance is 0, so z is NaN and every
    # configuration is flat. A training window inside them has no exposure, hence no Sharpe.
    prices.iloc[400:760, prices.columns.get_loc("Y")] = prices["Y"].iloc[399]
    res = run_walk_forward(prices, rf, MEAN_REVERSION, **MR_GRID, **WINDOWS)
    windows = positions(res, prices.index)
    flat = [j for j, f in enumerate(res.folds) if f.selected is None]
    assert flat and len(flat) < len(res.folds)
    split = run_strategy(prices, rf, MEAN_REVERSION).split_date
    for j in flat:
        fold, (b, end) = res.folds[j], windows[j]
        train = slice(prices.index.get_loc(fold.train_start), b)
        for L in MR_GRID["L_values"]:  # no configuration has a defined training Sharpe
            for z in MR_GRID["z_entry_values"]:
                bt = run_strategy(
                    prices, rf, replace(MEAN_REVERSION, L=L, z_entry=z, split_date=split)
                ).backtest
                held = bt.held_id.iloc[train].notna().to_numpy()
                assert reference_sharpe(bt.excess.to_numpy()[train], held) is None, (j, L, z)
        last = windows[j + 1][0] - 1 if j + 1 < len(windows) else end
        assert (res.weights.to_numpy()[b - 1 : last] == 0.0).all()  # zero weights over its span
        assert fold.train_sharpe is None and fold.test_sharpe is None
        start, stop = f"{fold.test_start:%Y-%m-%d}", f"{fold.test_end:%Y-%m-%d}"
        assert (
            f"Fold {j} (test {start} to {stop}): no configuration was eligible, so the walk-forward was flat."
            in res.warnings
        )


def crash_market(crash_days: tuple[int, ...], n: int = 1000, seed: int = 11) -> pd.DataFrame:
    """A calm uptrend (0.1% a day, 0.5% daily vol) with a -50% day at each of crash_days."""
    g = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n).as_unit("ns")
    log_r = g.normal(0.001, 0.005, n)
    log_r[list(crash_days)] = np.log(0.5)
    return pd.DataFrame({"SPY": 100.0 * np.exp(np.cumsum(log_r))}, index=idx)


def test_ruin_ends_the_stitched_series() -> None:
    # Trend at the 2x cap. The request's own L = 21 is long into the crash on day 100 and ruined
    # there, before the walk-forward starts. The stitched series is long 2x into day 700.
    prices = crash_market((100, 700))
    rf = pd.Series(0.02 / 252, index=prices.index)
    params = StrategyParams(
        strategy="trend",
        tickers=("SPY",),
        L=21,
        signal_frequency="daily",
        vol_target=0.5,
        max_leverage=2.0,
        vol_refresh="daily",
    )
    res = run_walk_forward(prices, rf, params, L_values=(126, 252), **WINDOWS)
    assert res.ruined and res.ruin_date == prices.index[700] and res.stop == 701
    assert res.daily.total.iloc[700] == -1.0 and res.series["equity"].iloc[-1] == 0.0
    assert res.series.index[-1] == prices.index[700] and res.metrics.max_drawdown == -1.0
    assert any(w.startswith("Ruin on 20") for w in res.warnings)
    for fold in res.folds:
        if fold.test_start > res.ruin_date:  # starts after the ruin day: no stitched days
            assert fold.test_sharpe is None
            assert fold.selected is None  # and every configuration was ruined by then
    # The request's own configuration was ruined on day 100, before the window, so it has no metrics.
    assert run_strategy(prices, rf, params).backtest.ruin_date == prices.index[100] < prices.index[res.e]
    assert res.fixed_metrics is None and res.series["fixed_equity"].isna().all()


def test_a_ruined_walk_forward_cuts_the_fixed_run_at_its_ruin_day() -> None:
    # A steady climb with a 15-day slide before a -50% day on day 700. The longer lookbacks the
    # grid chooses from are still long 2x into the crash, and the walk-forward is ruined. The
    # request's own L = 21 has turned short, so it survives, with a short trade open across the crash.
    g = np.random.default_rng(11)
    idx = pd.bdate_range("2018-01-01", periods=1000).as_unit("ns")
    log_r = g.normal(0.002, 0.005, 1000)
    log_r[685:700] = g.normal(-0.01, 0.005, 15)
    log_r[700] = np.log(0.5)
    prices = pd.DataFrame({"SPY": 100.0 * np.exp(np.cumsum(log_r))}, index=idx)
    rf = pd.Series(0.02 / 252, index=idx)
    params = StrategyParams(
        strategy="trend",
        tickers=("SPY",),
        L=21,
        signal_frequency="daily",
        vol_target=0.5,
        max_leverage=2.0,
        vol_refresh="daily",
    )
    res = run_walk_forward(prices, rf, params, L_values=(126, 252), **WINDOWS)
    own = run_strategy(prices, rf, params).backtest
    assert res.ruin_date == idx[700] and not own.ruined and own.w.iloc[699, 0] < 0
    # The fixed run is measured on the walk-forward's days only, which end on day 700...
    fixed = res.fixed_metrics
    assert fixed.n_days == res.metrics.n_days == 700 - res.e + 1
    assert res.series["fixed_equity"].notna().all() and res.series.index[-1] == idx[700]
    # ...so its short trade, entered in the window and closed after day 700 in its own backtest,
    # is still open there.
    trades = own.trades
    entered = trades.loc[trades["entry_date"].between(idx[res.e], idx[700])]
    crossing = entered.loc[entered["exit_date"] > idx[700]]
    assert len(crossing) == 1 and crossing["side"].iloc[0] == "short" and not crossing["open"].iloc[0]
    assert fixed.n_open_trades == entered["open"].sum() + 1
    assert fixed.n_closed_trades == len(entered) - fixed.n_open_trades


def test_the_walk_forward_warnings() -> None:
    a, b = (20, 1.5), (60, 2.0)
    days = [(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-03-31"))] * 4

    def warned(selected: list) -> bool:
        out = metrics.walk_forward_warnings(
            selected=selected, test_windows=days[: len(selected)], ruin_date=None
        )
        return any("unstable across folds" in w for w in out)

    assert warned([a, b, a, b])  # 3 changes in 4 folds
    assert warned([a, b, a])  # 2 in 3
    assert not warned([a, b, b, a])  # 2 in 4 is not more than half
    assert not warned([a, b])  # 1 in 2
    assert warned([a, None, a, b])  # a flat fold counts as a change of selection
    flat = "(test 2020-01-02 to 2020-03-31): no configuration was eligible, so the walk-forward was flat."
    assert metrics.walk_forward_warnings(selected=[a, None], test_windows=days[:2], ruin_date=None) == [
        f"Fold 1 {flat}"
    ]
    assert metrics.walk_forward_warnings(selected=[None, a, a], test_windows=days[:3], ruin_date=None) == [
        f"Fold 0 {flat}"  # one change in three folds is stable
    ]
    ruin = metrics.walk_forward_warnings(
        selected=[a, a], test_windows=days[:2], ruin_date=pd.Timestamp("2020-02-03")
    )
    assert ruin == [metrics.ruin_warning(pd.Timestamp("2020-02-03"))] and ruin[0].startswith(
        "Ruin on 2020-02-03"
    )


def test_too_few_days_for_two_folds_is_a_422(client: TestClient, prices: pd.DataFrame) -> None:
    end = prices.index[949].date()  # one fold, then an 11-day window that is dropped
    request = {
        "strategy": "pairs",
        "tickers": ["Y", "X"],
        "start": "2018-01-01",
        "end": str(end),
        "L_values": [20, 120],
        "z_entry_values": [2.0],
    }
    response = client.post("/api/walk-forward", json=request)
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "make 1 fold(s) of 756 training and 63 test days" in detail and "at least 2 are needed" in detail
