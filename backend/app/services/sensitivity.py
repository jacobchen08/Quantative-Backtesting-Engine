"""The §9 sensitivity grid: in-sample Sharpe ratios of every (L, z_entry) configuration on one
common window, deflated for the number of configurations tried (§9.1). Pure (§0, rule 8)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from app.services import metrics, stats
from app.services.backtester import MIN_IS_DAYS, BacktestResult, InsufficientDataError, NoValidScoreError
from app.services.pipeline import TREND_STRATEGIES, ScoreCache, StrategyParams, StrategyRun, run_strategy

DEFAULT_L_GRID = {
    "mean_reversion": (20, 40, 60, 90, 120),
    "pairs": (20, 40, 60, 90, 120),
    "trend": (21, 63, 126, 252, 378),  # 1, 3, 6, 12 and 18 months
    "trend_portfolio": (21, 63, 126, 252, 378),
}
DEFAULT_Z_ENTRY_GRID = (1.5, 2.0, 2.5, 3.0)
MAX_GRID_VALUES = 8

Cell = tuple[int, float | None]  # a grid configuration (L, z_entry); z_entry is None for trend


@dataclass(frozen=True)
class BestCell:
    """The grid cell with the highest IS Sharpe (ties go to the first in grid order)."""

    L: int
    z_entry: float | None
    sharpe: float  # annualized, as in the grid
    dsr: float | None


@dataclass(frozen=True)
class DeflatedSharpe:
    """§9.1 deflated Sharpe ratios on the common window.

    n_trials: grid cells with a defined Sharpe. sr0_star: the expected maximum Sharpe of that
    many skill-less trials, annualized for display (None when n_trials < 2 or the trials'
    Sharpe ratios don't vary). selected_dsr: the DSR of the request's own cell, None when it is
    not in the grid. best: None only when no cell has a defined Sharpe.
    """

    n_trials: int
    sr0_star: float | None
    selected_dsr: float | None
    best: BestCell | None


@dataclass(frozen=True)
class SensitivityResult:
    """IS Sharpe ratios on the common window [window_start, window_end).

    sharpe has one row per L and one column per z_entry (a single column for trend), None
    where undefined. window_end is the split date, which the window excludes.
    """

    L_values: tuple[int, ...]
    z_entry_values: tuple[float, ...] | None
    sharpe: list[list[float | None]]
    n_configs_tested: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    selected: tuple[int, float | None] | None  # the request's own cell, if it is in the grid
    deflated: DeflatedSharpe


def run_sensitivity(
    prices: pd.DataFrame,
    rf: pd.Series,
    params: StrategyParams,
    *,
    L_values: tuple[int, ...] | None = None,
    z_entry_values: tuple[float, ...] | None = None,
) -> SensitivityResult:
    """Parameter grid scored on one common in-sample window (§9).

    1. Resolve the split once, for the request's own configuration.
    2. Run every configuration through run_strategy with its own t0 and that split.
    3. Score each one's daily excess returns on [latest first-evaluated day, split) with the
       §8 Sharpe rules, then deflate the best cell's and the request's own Sharpe ratios for
       the number of cells tried (§9.1). OOS results are never used.

    Raises:
        InsufficientDataError: a configuration leaves too short a window or has no valid
            score; the message names the culprit L.
    """
    trend = params.strategy in TREND_STRATEGIES
    Ls, zs = grid_axes(params, L_values, z_entry_values)
    split = run_strategy(prices, rf, params).split_date  # 1. exactly as /api/backtest resolves it
    # 2. Every configuration with its own t0 and the common split.
    runs = run_grid(prices, rf, params, Ls, zs, split, label="Sensitivity grid", where="in the common window")

    # 3. The common window starts at the latest first-evaluated day across the grid.
    calendar = prices.index
    start = max(r.backtest.e for r in runs.values())
    split_pos = int(calendar.searchsorted(split))
    if split_pos - start < MIN_IS_DAYS:  # implied by step 2's checks; kept as a guard
        raise InsufficientDataError(
            f"Sensitivity grid: the common window has {split_pos - start} days ({MIN_IS_DAYS} needed).",
            n_is=split_pos - start,
        )
    window = {cell: _window_excess(r.backtest, start, split_pos) for cell, r in runs.items()}
    grid = [[_cell_sharpe(window[(L, z)]) for z in zs or (None,)] for L in Ls]
    in_grid = params.L in Ls and (trend or params.z_entry in zs)
    selected = (params.L, None if trend else params.z_entry) if in_grid else None
    return SensitivityResult(
        L_values=Ls,
        z_entry_values=zs,
        sharpe=grid,
        n_configs_tested=len(runs),
        window_start=calendar[start],
        window_end=split,
        selected=selected,
        deflated=deflated_sharpe(window, selected),
    )


def grid_axes(
    params: StrategyParams, L_values: tuple[int, ...] | None, z_entry_values: tuple[float, ...] | None
) -> tuple[tuple[int, ...], tuple[float, ...] | None]:
    """The §9 grid axes: the defaults for the strategy, or the request's overrides.

    Trend and the trend portfolio have no z_entry axis, so their z_entry_values are ignored and
    None is returned for them.

    Raises:
        ValueError: an axis has no values or more than 8, or a z_entry is not above z_exit.
    """
    trend = params.strategy in TREND_STRATEGIES
    Ls = tuple(L_values) if L_values is not None else DEFAULT_L_GRID[params.strategy]
    zs = None if trend else tuple(z_entry_values) if z_entry_values is not None else DEFAULT_Z_ENTRY_GRID
    if not 1 <= len(Ls) <= MAX_GRID_VALUES or (zs is not None and not 1 <= len(zs) <= MAX_GRID_VALUES):
        raise ValueError(f"each grid axis needs 1 to {MAX_GRID_VALUES} values")
    if zs is not None and not all(z > params.z_exit for z in zs):
        raise ValueError(f"every z_entry must be greater than z_exit = {params.z_exit}")
    return Ls, zs


def run_grid(
    prices: pd.DataFrame,
    rf: pd.Series,
    params: StrategyParams,
    Ls: tuple[int, ...],
    zs: tuple[float, ...] | None,
    split: pd.Timestamp,
    *,
    label: str,
    where: str,
) -> dict[Cell, StrategyRun]:
    """Every grid configuration through run_strategy, each with its own t0 and the common split
    (§9, step 2). Keyed by (L, z_entry), in grid order: rows by L, then z_entry. The runs skip
    the trade bookkeeping, since the grids read only each backtest's daily series, and share
    one score per L.

    Raises:
        NoValidScoreError, InsufficientDataError: a configuration has too little data. The
            message starts with label and names the culprit L; `where` says which in-sample
            days ran short.
    """
    runs: dict[Cell, StrategyRun] = {}
    score_cache: ScoreCache = {}
    for L in Ls:
        for z in zs if zs is not None else (None,):
            cfg = (
                replace(params, L=L, split_date=split)
                if z is None
                else replace(params, L=L, z_entry=z, split_date=split)
            )
            try:
                runs[(L, z)] = run_strategy(prices, rf, cfg, trades=False, score_cache=score_cache)
            except NoValidScoreError as exc:
                raise NoValidScoreError(
                    f"{label}: L = {L} gives no valid score on this date range. "
                    f"Use a longer date range or drop L = {L} from the grid."
                ) from exc
            except InsufficientDataError as exc:
                if exc.n_is is not None and exc.n_is < MIN_IS_DAYS:
                    message = (
                        f"{label}: L = {L} leaves only {exc.n_is} in-sample days {where} "
                        f"({MIN_IS_DAYS} needed). Use a longer date range or drop L = {L} from the grid."
                    )
                else:
                    message = f"{label}: L = {L}: {exc}"
                raise InsufficientDataError(message, n_is=exc.n_is, n_oos=exc.n_oos) from exc
    return runs


WindowExcess = tuple[np.ndarray, bool] | None  # (daily excess, held any position); None = no days


def _window_excess(bt: BacktestResult, start: int, end: int) -> WindowExcess:
    """One configuration's daily excess returns on calendar positions [start, end), and whether
    it held a position there. None for a configuration ruined before the window."""
    stop = min(end, bt.stop)
    if stop <= start:
        return None
    sl = slice(start, stop)
    return bt.excess.iloc[sl].to_numpy(), bool(bt.held.iloc[sl].any())


def _cell_sharpe(window: WindowExcess) -> float | None:
    """§8 Sharpe of one cell on the common window (annualized); None without days or exposure."""
    return None if window is None else metrics.sharpe(window[0], exposed=window[1])


def deflated_sharpe(window: dict[Cell, WindowExcess], selected: Cell | None) -> DeflatedSharpe:
    """§9.1 deflated Sharpe ratios of a scored grid.

    Args:
        window: every grid cell's daily excess returns on the common window, in grid order
            (rows by L, then z_entry), as _window_excess returns them.
        selected: the request's own cell, or None when it isn't in the grid.

    The trials are the cells with a defined §8 Sharpe. Every Sharpe ratio in the formulas is
    daily; sr0_star is annualized only for display. The best cell is the highest Sharpe, with
    ties going to the first in grid order.
    """
    sharpe = {cell: _cell_sharpe(x) for cell, x in window.items()}
    trials = [cell for cell, value in sharpe.items() if value is not None]
    daily = [metrics.sharpe_daily(window[cell][0]) for cell in trials]
    sr0 = stats.expected_max_sharpe(daily)

    def dsr(cell: Cell) -> float | None:
        if sr0 is None or sharpe.get(cell) is None:
            return None
        return stats.probabilistic_sharpe(window[cell][0], sr0)

    best = None
    if trials:
        top = max(trials, key=lambda cell: sharpe[cell])  # max keeps the first of equal values
        best = BestCell(L=top[0], z_entry=top[1], sharpe=sharpe[top], dsr=dsr(top))
    return DeflatedSharpe(
        n_trials=len(trials),
        sr0_star=None if sr0 is None else sr0 * math.sqrt(metrics.TRADING_DAYS),
        selected_dsr=None if selected is None else dsr(selected),
        best=best,
    )
