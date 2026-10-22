"""Walk-forward analysis (§9.2): re-choose the configuration of the §9 grid at the start of each
fold, using only data before it, and stitch the chosen weights together. Pure (§0, rule 8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services import backtester, metrics
from app.services.backtester import InsufficientDataError
from app.services.metrics import Metrics
from app.services.pipeline import MARKET, StrategyParams, StrategyRun, run_strategy
from app.services.report import benchmark_returns, daily_metrics
from app.services.sensitivity import Cell, grid_axes, run_grid

TRAIN_DAYS_RANGE = (252, 2_520)
TEST_DAYS_RANGE = (21, 252)
MIN_TEST_DAYS = 21  # a shorter final test window is dropped
MIN_FOLDS = 2


@dataclass(frozen=True)
class FoldWindow:
    """Calendar positions of one fold: training [train_start, test_start), test [test_start, test_end)."""

    train_start: int
    test_start: int  # b_j, the first test day
    test_end: int  # one past the last test day


def walk_forward_folds(
    n: int, start: int, train_days: int, test_days: int, anchored: bool
) -> list[FoldWindow]:
    """The §9.2 folds on a run calendar of n days, from the grid's common start s = start.

    With T = train_days and S = test_days, fold j = 0, 1, ... tests [s + T + jS, s + T + (j+1)S),
    cut at the end of the calendar, and trains on [s + jS, s + T + jS) when rolling or on
    [s, s + T + jS) when anchored. A final test window shorter than 21 days is dropped.
    """
    folds: list[FoldWindow] = []
    b = start + train_days
    while b < n and min(b + test_days, n) - b >= MIN_TEST_DAYS:
        train_start = start if anchored else start + len(folds) * test_days
        folds.append(FoldWindow(train_start=train_start, test_start=b, test_end=min(b + test_days, n)))
        b += test_days
    return folds


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold (§9.2). Dates are inclusive; selected is None for a flat fold."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected: Cell | None
    train_sharpe: float | None  # the chosen configuration's §8 Sharpe on the training window
    test_sharpe: float | None  # §8 Sharpe of the stitched excess on the days this fold owns


@dataclass(frozen=True)
class WalkForwardResult:
    """Walk-forward re-optimization over the §9 grid (§9.2).

    folds: in order. Fold j owns its test days; fold 0 also owns day e, its entry day.
    series: one row per date from the baseline e - 1 to the last stitched day, with columns
        equity (stitched), fixed_equity (the request's own configuration, NaN after its ruin),
        benchmark_equity and fold (the fold owning the day; NaN on the baseline).
    metrics: §8 metrics of the stitched series on [e, stop), with the trade fields None.
    fixed_metrics: the request's own configuration on the same days, None if it was ruined
        before them; benchmark_metrics: buy-and-hold on the same days (§6.4).
    weights, daily: the stitched executed weights w* and their §6.1 components (ruin day
        clipped) on the whole calendar. Only positions [e, stop) form the walk-forward.
    e: the first evaluated day, b0 - 1. stop: one past the last one (the ruin day, if any).
    """

    L_values: tuple[int, ...]
    z_entry_values: tuple[float, ...] | None
    n_configs: int
    folds: list[Fold]
    n_distinct_selected: int
    series: pd.DataFrame
    metrics: Metrics
    fixed_metrics: Metrics | None
    benchmark_metrics: Metrics
    ruined: bool
    ruin_date: pd.Timestamp | None
    warnings: list[str]
    weights: pd.DataFrame
    daily: backtester.DailyReturns
    e: int
    stop: int


def run_walk_forward(
    prices: pd.DataFrame,
    rf: pd.Series,
    params: StrategyParams,
    *,
    L_values: tuple[int, ...] | None = None,
    z_entry_values: tuple[float, ...] | None = None,
    train_days: int = 756,
    test_days: int = 63,
    anchored: bool = False,
) -> WalkForwardResult:
    """Walk-forward analysis over the §9 parameter grid (§9.2).

    1. Run the grid: every configuration through run_strategy with its own t0 and the request's
       resolved split. The split only satisfies run_strategy; nothing here uses it.
    2. Folds from s, the latest first-evaluated day across the grid (walk_forward_folds).
    3. At the start of each fold, choose the eligible configuration with the highest §8 Sharpe on
       the training window, using returns through its last close only. Eligible: evaluating by
       the window's first day, not ruined on or before its last, and a defined Sharpe on it;
       ties go to grid order. With none eligible the fold is flat.
    4. Stitch: the chosen configuration's executed weights rule from the close before its test
       window, b_j - 1, through the close before the next one; the last fold's run to its end.
       §6.1 and §6.2's ruin rule then apply to the stitched weights w*, so a switch is an
       ordinary trade, charged on the last day of the fold being left.

    The weights at a close depend only on prices through that close, so the stitched series is
    causal. The request's own configuration is measured on the same days for comparison.

    Raises:
        ValueError: train_days, test_days or the grid out of range.
        InsufficientDataError: fewer than 2 folds, or a configuration has too little data (the
            message names it).
    """
    for name, value, (lo, hi) in (
        ("train_days", train_days, TRAIN_DAYS_RANGE),
        ("test_days", test_days, TEST_DAYS_RANGE),
    ):
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be in {lo}..{hi}, got {value}")
    Ls, zs = grid_axes(params, L_values, z_entry_values)
    own = run_strategy(prices, rf, params)  # the split, and the fixed-parameter comparison
    runs = run_grid(
        prices, rf, params, Ls, zs, own.split_date, label="Walk-forward grid", where="before the split"
    )

    # 2. Folds.
    calendar = prices.index
    n = len(calendar)
    s = max(run.backtest.e for run in runs.values())
    windows = walk_forward_folds(n, s, train_days, test_days, anchored)
    if len(windows) < MIN_FOLDS:
        raise InsufficientDataError(
            f"Walk-forward: the {n - s} trading days from the grid's first common evaluated day "
            f"({calendar[s]:%Y-%m-%d}) to the end make {len(windows)} fold(s) of {train_days} training "
            f"and {test_days} test days (a last test window under {MIN_TEST_DAYS} days is dropped); "
            f"at least {MIN_FOLDS} are needed. Use a longer date range, or shorter windows."
        )

    # 3. Selection, causal: fold j looks at returns through the close of b_j - 1 only.
    excess = {cell: run.backtest.excess.to_numpy() for cell, run in runs.items()}
    held = {cell: run.backtest.held.to_numpy() for cell, run in runs.items()}
    chosen = [select_configuration(runs, excess, held, fw.train_start, fw.test_start) for fw in windows]
    selected = [cell for cell, _ in chosen]

    # 4. Stitching, then §6.1 and §6.2's ruin rule on the stitched weights.
    legs = list(params.tickers)
    weights = pd.DataFrame(_stitched_weights(runs, windows, selected), index=calendar, columns=legs)
    d = backtester.daily_returns(
        w=weights, R=own.returns[legs], rf=rf, cost_bps=params.cost_bps, borrow_bps=params.borrow_bps
    )
    e = windows[0].test_start - 1  # the first fold's entry day: the first evaluated day
    r = backtester.first_ruin_day(d.total, e, windows[-1].test_end)
    stop = windows[-1].test_end if r is None else r + 1
    if r is not None:
        backtester.clip_ruin_day(d.excess, d.total, rf, r)

    # The fold table: each fold's dates, choice, and Sharpe on the training and test windows.
    stitched_held = (d.gross_held > 0).to_numpy()  # a position earns the day's P&L
    folds, owner = _fold_table(windows, chosen, d.excess.to_numpy(), stitched_held, e, stop, calendar)

    # 5. Outputs: the stitched series, the request's own configuration and the benchmark, on [e, stop).
    win = slice(e, stop)
    market_excess = own.returns[MARKET] - rf
    # Positions span configurations, so there are no trades to count.
    stitched_metrics = daily_metrics(d, stitched_held, win, market_excess, trades=None)
    fixed_stop = min(stop, own.backtest.stop)  # the request's own run ends at its ruin day, if any
    fixed_metrics = _fixed_metrics(own, e, fixed_stop, market_excess) if fixed_stop > e else None
    bench = benchmark_returns(own.returns, params)  # §6.4
    bench_metrics = metrics.benchmark_metrics(
        r=bench.iloc[win], rf=rf.iloc[win], market_excess=market_excess.iloc[win]
    )
    series = _equity_series(params.initial_capital, d, own, bench, owner, e, stop, fixed_stop)
    warnings = metrics.walk_forward_warnings(
        selected=selected,
        test_windows=[(calendar[fw.test_start], calendar[fw.test_end - 1]) for fw in windows],
        ruin_date=None if r is None else calendar[r],
    )
    return WalkForwardResult(
        L_values=Ls,
        z_entry_values=zs,
        n_configs=len(runs),
        folds=folds,
        n_distinct_selected=len({cell for cell in selected if cell is not None}),
        series=series,
        metrics=stitched_metrics,
        fixed_metrics=fixed_metrics,
        benchmark_metrics=bench_metrics,
        ruined=r is not None,
        ruin_date=None if r is None else calendar[r],
        warnings=warnings,
        weights=weights,
        daily=d,
        e=e,
        stop=stop,
    )


def _fold_table(
    windows: list[FoldWindow],
    chosen: list[tuple[Cell | None, float | None]],
    stitched_excess: np.ndarray,
    stitched_held: np.ndarray,
    e: int,
    stop: int,
    calendar: pd.DatetimeIndex,
) -> tuple[list[Fold], np.ndarray]:
    """The folds, and the fold that owns each calendar day (NaN outside [e, stop)).

    Fold j owns its test days, and fold 0 also day e, its entry day: together they partition
    [e, stop). A fold's test Sharpe is the §8 Sharpe of the stitched excess on the days it owns.
    """
    owner = np.full(len(calendar), np.nan)
    owner[e] = 0
    folds = []
    for j, (fw, (cell, train_sharpe)) in enumerate(zip(windows, chosen, strict=True)):
        owner[fw.test_start : fw.test_end] = j
        lo, hi = (e if j == 0 else fw.test_start), min(fw.test_end, stop)  # empty after a ruin
        days, exposed = stitched_excess[lo:hi], bool(stitched_held[lo:hi].any())
        test_sharpe = metrics.sharpe(days, exposed=exposed) if hi > lo else None
        folds.append(
            Fold(
                train_start=calendar[fw.train_start],
                train_end=calendar[fw.test_start - 1],
                test_start=calendar[fw.test_start],
                test_end=calendar[fw.test_end - 1],
                selected=cell,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
            )
        )
    return folds, owner


def _equity_series(
    E0: float,
    daily: backtester.DailyReturns,
    own: StrategyRun,
    bench: pd.Series,
    owner: np.ndarray,
    e: int,
    stop: int,
    fixed_stop: int,
) -> pd.DataFrame:
    """The chart series from the baseline e - 1 to stop - 1: the stitched equity, the request's own
    configuration (NaN after its ruin), the benchmark, and the fold owning each day."""
    calendar = own.prices.index
    win = slice(e, stop)
    fixed_equity = np.full(stop - e + 1, np.nan)
    if fixed_stop > e:
        fixed_equity[: fixed_stop - e + 1] = _equity_path(E0, own.backtest.total.to_numpy()[e:fixed_stop])
    return pd.DataFrame(
        {
            "equity": _equity_path(E0, daily.total.to_numpy()[win]),
            "fixed_equity": fixed_equity,
            "benchmark_equity": _equity_path(E0, bench.to_numpy()[win]),
            "fold": owner[e - 1 : stop],
        },
        index=calendar[e - 1 : stop],
    )


def _stitched_weights(
    runs: dict[Cell, StrategyRun], windows: list[FoldWindow], selected: list[Cell | None]
) -> np.ndarray:
    """§9.2 step 4: the executed weights w*. Fold j's choice rules the closes [b_j - 1, b_{j+1} - 1),
    the last fold's through the end of its test window; a fold with no choice is flat."""
    first_run = next(iter(runs.values())).backtest.w
    w_star = np.zeros(first_run.shape)
    for j, (fw, cell) in enumerate(zip(windows, selected, strict=True)):
        first = fw.test_start - 1
        last = windows[j + 1].test_start - 1 if j + 1 < len(windows) else fw.test_end
        if cell is not None:
            w_star[first:last] = runs[cell].backtest.w.to_numpy()[first:last]
    return w_star


def select_configuration(
    runs: dict[Cell, StrategyRun],
    excess: dict[Cell, np.ndarray],
    held: dict[Cell, np.ndarray],
    start: int,
    end: int,
) -> tuple[Cell | None, float | None]:
    """§9.2 step 3 on the training window, calendar positions [start, end).

    Args:
        runs: the grid's runs in grid order; only each backtest's e, ruined and stop are read.
        excess, held: each configuration's daily excess returns and held-day flags on the whole
            calendar (positions start..end - 1 are read, so only data through the close of end - 1).
        start, end: the training window.

    Returns:
        The eligible configuration with the highest §8 Sharpe on the window, and that Sharpe.
        Eligible: evaluating by the window's first day, not ruined on or before its last day, and a
        defined Sharpe there. Ties go to grid order; (None, None) when none is eligible.
    """
    best: Cell | None = None
    best_sharpe: float | None = None
    for cell, run in runs.items():  # grid order
        bt = run.backtest
        if bt.e > start or (bt.ruined and bt.stop <= end):  # not evaluating yet, or ruined by end - 1
            continue
        sharpe = metrics.sharpe(excess[cell][start:end], exposed=bool(held[cell][start:end].any()))
        if sharpe is not None and (best_sharpe is None or sharpe > best_sharpe):
            best, best_sharpe = cell, sharpe
    return best, best_sharpe


def _fixed_metrics(run: StrategyRun, start: int, stop: int, market_excess: pd.Series) -> Metrics:
    """§8 metrics of one configuration's own backtest on calendar positions [start, stop).

    The trades are those entered in the window. One still held after its last day is open there.
    """
    bt, calendar = run.backtest, run.prices.index
    first, last = calendar[start], calendar[stop - 1]
    trades = bt.trades.loc[(bt.trades["entry_date"] >= first) & (bt.trades["entry_date"] <= last)].copy()
    trades["open"] = trades["open"] | (trades["exit_date"] > last)
    return daily_metrics(bt, bt.held.to_numpy(), slice(start, stop), market_excess, trades)


def _equity_path(E0: float, total: np.ndarray) -> np.ndarray:
    """A baseline point E0, then E0 compounded by each day's total return (§6.2)."""
    out = np.empty(len(total) + 1)
    out[0] = E0
    out[1:] = E0 * np.cumprod(1.0 + total)
    return out
