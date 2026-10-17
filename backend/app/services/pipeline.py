"""run_strategy() (§5.0), the backtest report, run_sensitivity() (§9) and Monte Carlo inputs.

Pure (§0, rule 8). Prices and the risk-free rate come in as pandas objects on the run
calendar; results go out as dataclasses. run_strategy is the single place that runs the
§5.0 steps; everything else (the API, the sensitivity grid, the tests) goes through it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from app.services import backtester, metrics, sizing, stats, strategies
from app.services.backtester import MIN_IS_DAYS, BacktestResult, InsufficientDataError, NoValidScoreError
from app.services.metrics import CostSensitivity, Metrics, PeriodReturn
from app.services.monte_carlo import MonteCarloInputError
from app.services.stats import PairsDiagnostics

MARKET = "SPY"  # always loaded: the pairs benchmark and the beta reference (§4.5, §6.4)
STRATEGIES = ("mean_reversion", "pairs", "trend")
DEFAULT_L_GRID = {
    "mean_reversion": (20, 40, 60, 90, 120),
    "pairs": (20, 40, 60, 90, 120),
    "trend": (21, 63, 126, 252, 378),  # 1, 3, 6, 12 and 18 months
}
DEFAULT_Z_ENTRY_GRID = (1.5, 2.0, 2.5, 3.0)
MAX_GRID_VALUES = 8


@dataclass(frozen=True)
class StrategyParams:
    """A resolved strategy configuration; the API fills in every default before calling.

    Units: L and vol_halflife in trading days; z thresholds in standard deviations; cost_bps
    one-way per leg in basis points; borrow_bps an annual fee on short notional in basis points;
    vol_target annualized as a decimal; split_date is the requested split (None = the 70%
    default). allow_short=False clips the signal at 0 (§5.9). Fields that don't apply to the
    strategy are ignored.
    """

    strategy: Literal["mean_reversion", "pairs", "trend"]
    tickers: tuple[str, ...]
    L: int
    z_entry: float | None = 2.0
    z_exit: float | None = 0.5
    signal_frequency: Literal["monthly", "daily"] = "monthly"
    hedge_mode: Literal["fixed_at_entry", "rolling"] = "fixed_at_entry"
    cost_bps: float = 10.0
    execution_lag: int = 1
    borrow_bps: float = 0.0
    allow_short: bool = True
    vol_target: float | None = None
    vol_halflife: int = 42
    vol_refresh: Literal["daily", "weekly", "monthly"] = "weekly"
    max_leverage: float = 2.0
    split_date: date | None = None
    initial_capital: float = 100_000.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "tickers", tuple(self.tickers))
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        n_legs = 2 if self.strategy == "pairs" else 1
        if len(self.tickers) != n_legs or len(set(self.tickers)) != n_legs:
            raise ValueError(f"{self.strategy} needs {n_legs} distinct ticker(s), got {self.tickers}")
        if self.L < 2:
            raise ValueError(f"L must be >= 2, got {self.L}")
        if self.strategy == "trend":
            if self.signal_frequency not in ("monthly", "daily"):
                raise ValueError(f"unknown signal_frequency {self.signal_frequency!r}")
        elif self.z_entry is None or self.z_exit is None or not 0 <= self.z_exit < self.z_entry:
            raise ValueError(f"need 0 <= z_exit < z_entry, got z_exit={self.z_exit}, z_entry={self.z_entry}")
        if self.strategy == "pairs" and self.hedge_mode not in ("fixed_at_entry", "rolling"):
            raise ValueError(f"unknown hedge_mode {self.hedge_mode!r}")
        if not 0 <= self.execution_lag <= 5:
            raise ValueError(f"execution_lag must be in 0..5, got {self.execution_lag}")
        if not self.cost_bps >= 0:
            raise ValueError(f"cost_bps must be >= 0, got {self.cost_bps}")
        if not self.borrow_bps >= 0:
            raise ValueError(f"borrow_bps must be >= 0, got {self.borrow_bps}")
        if self.strategy == "pairs" and not self.allow_short:
            raise ValueError("pairs needs a short leg, so allow_short=false is not available")
        if self.vol_target is not None:
            if not self.vol_target > 0 or not self.max_leverage > 0 or self.vol_halflife < 1:
                raise ValueError("need vol_target > 0, max_leverage > 0 and vol_halflife >= 1")
            if self.vol_refresh not in ("daily", "weekly", "monthly"):
                raise ValueError(f"unknown vol_refresh {self.vol_refresh!r}")
        if not self.initial_capital > 0:
            raise ValueError(f"initial_capital must be > 0, got {self.initial_capital}")


@dataclass(frozen=True)
class StrategyRun:
    """Everything the §5.0 steps produce.

    Strategy outputs (score, signal, weights, beta, vol_forecast, scale_used) are unlagged:
    the value at date t uses data through the close of t. The backtest holds the executed,
    lagged series and the evaluation window.
    """

    params: StrategyParams
    prices: pd.DataFrame  # adjusted closes of the legs and SPY on the run calendar
    rf: pd.Series
    returns: pd.DataFrame  # simple daily returns of the same columns (first row NaN)
    score: pd.Series  # reported score: masked z (A, B) or masked, sampled m (C)
    signal: pd.Series
    unit_weights: pd.DataFrame
    weights: pd.DataFrame  # W: sized target weights (the unit weights when the overlay is off)
    beta: pd.Series | None  # pairs: rolling beta
    beta_used: pd.Series | None  # pairs: the hedge ratio held
    vol_forecast: pd.Series | None  # overlay: annualized sigma
    scale_used: pd.Series | None  # overlay: the scale applied
    t0: int
    split_date: pd.Timestamp
    backtest: BacktestResult


def run_strategy(prices: pd.DataFrame, rf: pd.Series, params: StrategyParams) -> StrategyRun:
    """Run the §5.0 steps in order: score, vol forecast and mask, signal, unit weights, sizing,
    then t0, resolve_split and the backtester.

    Args:
        prices: adjusted closes with one column per ticker, including SPY, indexed by the run
            calendar (tz-naive datetime64[ns], ascending, unique). No NaN, all > 0.
        rf: daily risk-free rate (decimal) on the same calendar (§2.3, §4.5).
        params: the resolved configuration.

    Raises:
        TypeError: rf is not a pandas Series.
        NoValidScoreError: the score is NaN on every day (§6.2).
        InsufficientDataError: fewer than 252 IS or 126 OOS evaluated days (§6.3).
    """
    cols = _check_inputs(prices, rf, params)
    p = params
    legs = list(p.tickers)
    prices = prices[cols]
    returns = prices / prices.shift(1) - 1  # §2.1 simple returns; no pct_change
    overlay = p.vol_target is not None

    # 1. Raw score.
    beta = None
    if p.strategy == "mean_reversion":
        score = strategies.mean_reversion_score(prices[legs[0]], p.L)
    elif p.strategy == "pairs":
        pair = strategies.pairs_score(prices[legs[0]], prices[legs[1]], p.L)
        score, beta = pair.z, pair.beta
    else:
        score = strategies.trend_score(prices[legs[0]], rf, p.L)

    # 2. Vol forecast, then mask the score where the forecast is undefined.
    sigma = None
    if overlay:
        if p.strategy == "pairs":
            u = sizing.pair_unit_return(returns[legs[0]], returns[legs[1]], beta)
        else:
            u = returns[legs[0]]
        sigma, score = sizing.forecast_and_mask(u, score, p.vol_halflife)

    # 3. Signal. For trend the sampled m becomes the reported score. The long-only switch
    # clips the signal last (§5.9); the score is unchanged.
    if p.strategy == "trend":
        score, signal = strategies.trend_signal(score, p.signal_frequency)
    else:
        signal = strategies.hysteresis_signal(score, p.z_entry, p.z_exit)
    if not p.allow_short:
        signal = strategies.long_only(signal)

    # 4. Unit weights, with tickers as column labels.
    beta_used = None
    if p.strategy == "pairs":
        beta_used = strategies.hedge_ratio_used(beta, signal, p.hedge_mode)
        unit_weights = strategies.pair_unit_weights(signal, beta_used, (legs[0], legs[1]))
    else:
        unit_weights = strategies.single_unit_weights(signal, legs[0])

    # 5. Sizing.
    scale_used = None
    W = unit_weights
    if overlay:
        sized = sizing.size_weights(
            unit_weights,
            signal,
            sigma,
            vol_target=p.vol_target,
            max_leverage=p.max_leverage,
            vol_refresh=p.vol_refresh,
        )
        W, scale_used = sized.weights, sized.scale_used

    # 6. Evaluation: t0, the split, then the backtester.
    valid = score.notna().to_numpy()
    if not valid.any():
        raise NoValidScoreError(
            f"The {p.strategy} score is undefined on every day: the date range is too short for "
            f"L = {p.L}{' and the volatility half-life' if overlay else ''}, or every window is degenerate."
        )
    t0 = int(np.argmax(valid))
    k = p.execution_lag
    split = backtester.resolve_split(prices.index[t0 + k :], p.split_date)
    bt = backtester.run_backtest(
        R=returns[legs],
        W=W,
        s=signal,
        rf=rf,
        t0=t0,
        k=k,
        cost_bps=p.cost_bps,
        borrow_bps=p.borrow_bps,
        E0=p.initial_capital,
        split_date=split,
    )
    return StrategyRun(
        params=p,
        prices=prices,
        rf=rf,
        returns=returns,
        score=score.rename("score"),
        signal=signal,
        unit_weights=unit_weights,
        weights=W,
        beta=beta,
        beta_used=beta_used,
        vol_forecast=sigma,
        scale_used=scale_used,
        t0=t0,
        split_date=bt.split_date,
        backtest=bt,
    )


def _check_inputs(prices: pd.DataFrame, rf: pd.Series, params: StrategyParams) -> list[str]:
    """Assert the §2.1 index contract where data enters the core; return the columns used."""
    if not isinstance(rf, pd.Series):
        raise TypeError(f"rf must be a pandas Series on the run calendar (§2.3), not {type(rf).__name__}")
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"prices must be a pandas DataFrame, not {type(prices).__name__}")
    idx = prices.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is not None or idx.unit != "ns":
        raise ValueError("prices must be indexed by a tz-naive datetime64[ns] DatetimeIndex")
    if not (idx.is_monotonic_increasing and idx.is_unique):
        raise ValueError("the price index must be sorted ascending and unique")
    cols = list(dict.fromkeys([*params.tickers, MARKET]))
    missing = [c for c in cols if c not in prices.columns]
    if missing:
        raise ValueError(f"prices has no column for {missing}")
    values = prices[cols].to_numpy(dtype=float)
    if not (np.isfinite(values).all() and (values > 0).all()):
        raise ValueError("prices must be finite and > 0 (clean and inner-join them first)")
    if not rf.index.equals(idx) or not np.isfinite(rf.to_numpy(dtype=float)).all():
        raise ValueError("rf must be finite and indexed by the same calendar as prices")
    return cols


# ---------------------------------------------------------------- report


@dataclass(frozen=True)
class SizingSummary:
    """§11.3 sizing block; pct_days_at_cap is None when the strategy never held a position."""

    vol_target: float
    max_leverage: float
    pct_days_at_cap: float | None


@dataclass(frozen=True)
class BacktestReport:
    """What /api/backtest returns, apart from the run id and the request echo (§11.3).

    series: one row per date from s0 = min(t0, e - 1) to the last evaluated day, NaN where the
        API sends null. Columns: equity, benchmark_equity, drawdown, daily_return, position,
        gross_exposure, score, beta and beta_used (pairs only), vol_forecast (overlay only),
        is_oos, rolling_sharpe and rolling_beta.
    trades: the §7.3 trade records plus beta_at_entry (NaN unless pairs).
    metrics, benchmark_metrics: keyed "is", "oos" and "full"; "oos" is None if ruin hit in IS.
    cost_sensitivity: the §8.1 cost sweep; None when the run was ruined.
    calendar_monthly, calendar_annual: §8.2 calendar returns of the evaluation window.
    """

    split_date: pd.Timestamp
    first_evaluated_date: pd.Timestamp
    series: pd.DataFrame
    sizing: SizingSummary | None
    trades: pd.DataFrame
    metrics: dict[str, Metrics | None]
    benchmark_metrics: dict[str, Metrics | None]
    diagnostics: PairsDiagnostics | None
    cost_sensitivity: CostSensitivity | None
    calendar_monthly: list[PeriodReturn]
    calendar_annual: list[PeriodReturn]
    ruined: bool
    ruin_date: pd.Timestamp | None
    warnings: list[str]


def build_report(
    run: StrategyRun,
    *,
    pairs_diagnostics: Callable[[pd.Series, pd.Series, pd.Timestamp], PairsDiagnostics] = (
        stats.pairs_diagnostics
    ),
    end: date | None = None,
) -> BacktestReport:
    """Metrics per segment, benchmark, sizing, pairs diagnostics, cost sweep, calendar returns,
    warnings and chart series.

    pairs_diagnostics computes the §5.4 diagnostics from the Y and X adjusted closes and the
    split date. The API passes its diagnostics cache (§14), which returns the same result.
    end is the request's resolved end date, which decides whether the last calendar period is
    partial (§8.2); None means the last date of the run calendar.
    """
    p, bt = run.params, run.backtest
    legs = list(p.tickers)
    pairs = p.strategy == "pairs"
    market_excess = run.returns[MARKET] - run.rf  # SPY excess return, for beta and correlation
    bench = run.returns[MARKET if pairs else legs[0]]  # §6.4

    trades = bt.trades.copy()
    if pairs:
        trades["beta_at_entry"] = run.beta_used.reindex(pd.DatetimeIndex(trades["signal_date"])).to_numpy()
    else:
        trades["beta_at_entry"] = np.nan

    seg_metrics: dict[str, Metrics | None] = {}
    bench_metrics: dict[str, Metrics | None] = {}
    for name, bounds in bt.segment_bounds().items():
        if bounds is None:
            seg_metrics[name] = bench_metrics[name] = None
            continue
        sl = slice(*bounds)
        seg_metrics[name] = metrics.segment_metrics(
            total=bt.total.iloc[sl],
            excess=bt.excess.iloc[sl],
            market_excess=market_excess.iloc[sl],
            held=bt.held_id.iloc[sl].notna(),
            gross_held=bt.gross_held.iloc[sl],
            turnover=bt.turnover.iloc[sl],
            cost=bt.cost.iloc[sl],
            borrow=bt.borrow.iloc[sl],
            trades=trades if name == "full" else trades.loc[trades["segment"] == name],
        )
        bench_metrics[name] = metrics.benchmark_metrics(
            r=bench.iloc[sl], rf=run.rf.iloc[sl], market_excess=market_excess.iloc[sl]
        )

    sizing_summary = None
    if p.vol_target is not None:
        win = slice(bt.e, bt.stop)
        pct = metrics.pct_days_at_cap(bt.gross_held.iloc[win], bt.held_id.iloc[win].notna(), p.max_leverage)
        sizing_summary = SizingSummary(
            vol_target=p.vol_target, max_leverage=p.max_leverage, pct_days_at_cap=pct
        )

    diagnostics = None
    if pairs:
        diagnostics = pairs_diagnostics(run.prices[legs[0]], run.prices[legs[1]], bt.split_date)

    cost_sensitivity = None
    if not bt.ruined:  # clipping makes excess non-linear in the cost (§8.1)
        cost_sensitivity = metrics.cost_sweep(
            gross_excess=bt.gross_excess,
            borrow=bt.borrow,
            turnover=bt.turnover,
            rf=run.rf,
            held=bt.held_id.notna(),
            segments=bt.segment_bounds(),
            e=bt.e,
            cost_bps=p.cost_bps,
        )

    win = slice(bt.e, bt.stop)
    calendar = run.prices.index
    monthly, annual = metrics.calendar_returns(
        dates=calendar[win],
        total=bt.total.iloc[win],
        benchmark=bench.iloc[win],
        calendar=calendar,
        end=calendar[-1].date() if end is None else end,
        ruined=bt.ruined,
    )

    warnings = metrics.build_warnings(
        metrics=seg_metrics,
        trades=trades,
        ruin_date=bt.ruin_date,
        pct_at_cap=None if sizing_summary is None else sizing_summary.pct_days_at_cap,
        coint_p_is=None if diagnostics is None else diagnostics.coint_p_is,
        collinear_is=diagnostics is not None and diagnostics.collinear_is,
        collinear_oos=diagnostics is not None and diagnostics.collinear_oos,
        break_even_full_bps=None if cost_sensitivity is None else cost_sensitivity.break_even_bps["full"],
        cost_bps=p.cost_bps,
    )
    return BacktestReport(
        split_date=bt.split_date,
        first_evaluated_date=run.prices.index[bt.e],
        series=_series(run, bench, market_excess),
        sizing=sizing_summary,
        trades=trades,
        metrics=seg_metrics,
        benchmark_metrics=bench_metrics,
        diagnostics=diagnostics,
        cost_sensitivity=cost_sensitivity,
        calendar_monthly=monthly,
        calendar_annual=annual,
        ruined=bt.ruined,
        ruin_date=bt.ruin_date,
        warnings=warnings,
    )


def _series(run: StrategyRun, bench: pd.Series, market_excess: pd.Series) -> pd.DataFrame:
    """Chart series from s0 = min(t0, e - 1), so every executed trade's signal date is inside."""
    bt = run.backtest
    s0, base, stop = min(run.t0, bt.e - 1), bt.e - 1, bt.stop
    idx = run.prices.index[s0:stop]
    after_base = np.arange(s0, stop) > base
    bench_equity = np.full(stop - s0, np.nan)
    bench_equity[base - s0] = run.params.initial_capital
    bench_equity[base - s0 + 1 :] = run.params.initial_capital * np.cumprod(
        1.0 + bench.to_numpy()[bt.e : stop]
    )
    cols: dict[str, object] = {
        "equity": bt.equity.reindex(idx),  # NaN before the baseline, E0 on it
        "benchmark_equity": bench_equity,
        "drawdown": bt.drawdown.reindex(idx),
        "daily_return": bt.total.iloc[s0:stop].where(after_base),  # NaN up to and including the baseline
        "position": bt.pos.iloc[s0:stop],
        "gross_exposure": bt.w.abs().sum(axis=1).iloc[s0:stop],  # executed, at the end of each day
        "score": run.score.iloc[s0:stop],
    }
    if run.beta is not None:
        cols["beta"] = run.beta.iloc[s0:stop]
        cols["beta_used"] = run.beta_used.iloc[s0:stop]
    if run.vol_forecast is not None:
        cols["vol_forecast"] = run.vol_forecast.iloc[s0:stop]
    cols["is_oos"] = idx >= bt.split_date
    # §8.2: trailing 252 evaluated days, NaN before the 252nd and outside the evaluation window.
    win = slice(bt.e, stop)
    roll_sharpe, roll_beta = metrics.rolling_sharpe_beta(
        bt.excess.iloc[win], market_excess.iloc[win], bt.held_id.iloc[win].notna()
    )
    before = np.full(bt.e - s0, np.nan)
    cols["rolling_sharpe"] = np.concatenate([before, roll_sharpe])
    cols["rolling_beta"] = np.concatenate([before, roll_beta])
    return pd.DataFrame(cols, index=idx)


# ---------------------------------------------------------------- sensitivity (§9)


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
    trend = params.strategy == "trend"
    Ls = tuple(L_values) if L_values is not None else DEFAULT_L_GRID[params.strategy]
    zs = None if trend else tuple(z_entry_values) if z_entry_values is not None else DEFAULT_Z_ENTRY_GRID
    if not 1 <= len(Ls) <= MAX_GRID_VALUES or (zs is not None and not 1 <= len(zs) <= MAX_GRID_VALUES):
        raise ValueError(f"each grid axis needs 1 to {MAX_GRID_VALUES} values")
    if zs is not None and not all(z > params.z_exit for z in zs):
        raise ValueError(f"every z_entry must be greater than z_exit = {params.z_exit}")

    split = run_strategy(prices, rf, params).split_date  # 1. exactly as /api/backtest resolves it

    runs: dict[tuple[int, float | None], StrategyRun] = {}
    for L in Ls:
        for z in zs if zs is not None else (None,):
            cfg = (
                replace(params, L=L, split_date=split)
                if z is None
                else replace(params, L=L, z_entry=z, split_date=split)
            )
            try:
                runs[(L, z)] = run_strategy(prices, rf, cfg)  # 2. its own t0, the common split
            except NoValidScoreError as exc:
                raise NoValidScoreError(
                    f"Sensitivity grid: L = {L} gives no valid score on this date range. "
                    f"Use a longer date range or drop L = {L} from the grid."
                ) from exc
            except InsufficientDataError as exc:
                if exc.n_is is not None and exc.n_is < MIN_IS_DAYS:
                    message = (
                        f"Sensitivity grid: L = {L} leaves only {exc.n_is} in-sample days in the common "
                        f"window ({MIN_IS_DAYS} needed). Use a longer date range or drop L = {L} "
                        "from the grid."
                    )
                else:
                    message = f"Sensitivity grid: L = {L}: {exc}"
                raise InsufficientDataError(message, n_is=exc.n_is, n_oos=exc.n_oos) from exc

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


Cell = tuple[int, float | None]  # (L, z_entry); z_entry is None for trend
WindowExcess = tuple[np.ndarray, bool] | None  # (daily excess, held any position); None = no days


def _window_excess(bt: BacktestResult, start: int, end: int) -> WindowExcess:
    """One configuration's daily excess returns on calendar positions [start, end), and whether
    it held a position there. None for a configuration ruined before the window."""
    stop = min(end, bt.stop)
    if stop <= start:
        return None
    sl = slice(start, stop)
    return bt.excess.iloc[sl].to_numpy(), bool(bt.held_id.iloc[sl].notna().any())


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


# ---------------------------------------------------------------- Monte Carlo inputs (§10.1)


def strategy_mc_returns(run: StrategyRun, window: Literal["full", "is"]) -> tuple[np.ndarray, bool]:
    """Daily total returns of the evaluation window ("full") or its IS part ("is").

    Returns:
        (returns, has_exposure), where has_exposure says whether the strategy held a position
        on any day of the window.

    Raises:
        MonteCarloInputError: the window contains the ruin day, where ln(1 + R) is undefined.
    """
    bt = run.backtest
    end = bt.stop if window == "full" else min(bt.split_pos, bt.stop)
    if bt.ruined and bt.stop <= end:  # the ruin day stop - 1 is inside [e, end)
        label = "evaluation" if window == "full" else "in-sample"
        raise MonteCarloInputError(
            f"The {label} window contains the ruin day {bt.ruin_date:%Y-%m-%d}, where ln(1 + R) is undefined."
        )
    sl = slice(bt.e, end)
    return bt.total.iloc[sl].to_numpy(), bool(bt.held_id.iloc[sl].notna().any())


def ticker_returns(close: pd.Series) -> np.ndarray:
    """Daily simple returns C_t / C_{t-1} - 1 of one adjusted close, without the leading NaN."""
    return (close / close.shift(1) - 1).iloc[1:].to_numpy()
