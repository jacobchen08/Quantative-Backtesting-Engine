"""Performance metrics, the cost sweep, rolling and calendar statistics, and warnings (§8).

Pure (§0, rule 8). Metric functions take the daily values of one segment (numpy arrays or
pandas Series) as decimals: 0.01 = 1%. A statistic that is undefined returns None, never NaN
or inf (§2.4). Standard deviations use ddof=1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TRADING_DAYS = 252
STD_FLOOR = 1e-12  # a std (or downside deviation) below this makes a ratio undefined
CAP_TOLERANCE = 1e-9  # §5.6: "at the cap" means gross >= max_leverage * (1 - 1e-9)
MIN_CLOSED_TRADES = 10
CAP_WARNING_SHARE = 0.25
COINT_ALPHA = 0.05
SEGMENT_LABELS = {"is": "In-sample", "oos": "Out-of-sample", "full": "Full period"}
COST_SWEEP_BPS = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)  # §8.1, plus the request's own cost
ROLLING_DAYS = 252  # §8.2: one year of evaluated days
BREAK_EVEN_MULTIPLE = 2  # §8: warn when the full-window break-even is below 2x the request's cost


@dataclass(frozen=True)
class Metrics:
    """§8 metrics for one segment of n evaluated days.

    Returns, volatility and drawdown are decimals; ann_* fields and ratios are annualized with
    252 days. Trade and cost fields are None for benchmarks.
    """

    n_days: int
    total_return: float
    cagr: float | None
    ann_vol: float | None
    sharpe: float | None
    sharpe_ci95: tuple[float, float] | None
    sortino: float | None
    max_drawdown: float
    beta_spy: float | None
    corr_spy: float | None
    pct_days_in_market: float
    n_closed_trades: int | None
    n_open_trades: int | None
    win_rate: float | None
    avg_holding_days: float | None
    ann_turnover: float | None
    ann_cost_drag: float | None
    ann_borrow_drag: float | None
    avg_gross_exposure: float


def _arr(x: object) -> np.ndarray:
    return np.asarray(x, dtype=float)


def sharpe_daily(excess: object, *, exposed: bool = True) -> float | None:
    """Daily Sharpe ratio mean(excess) / std(excess) of daily excess returns.

    None when n < 2, std < 1e-12, or the segment has no exposure.
    """
    x = _arr(excess)
    if x.size < 2 or not exposed:
        return None
    sd = x.std(ddof=1)
    if not sd >= STD_FLOOR:
        return None
    return float(x.mean() / sd)


def sharpe(excess: object, *, exposed: bool = True) -> float | None:
    """Annualized Sharpe ratio, mean(excess) / std(excess) * sqrt(252) (§8)."""
    sr = sharpe_daily(excess, exposed=exposed)
    return None if sr is None else sr * math.sqrt(TRADING_DAYS)


def sharpe_ci95(sr_daily: float, n: int) -> tuple[float, float]:
    """95% CI of the annualized Sharpe ratio (Lo 2002, iid returns).

    [SR_d -+ 1.96 * sqrt((1 + SR_d^2 / 2) / n)] * sqrt(252), where SR_d is the daily Sharpe.
    """
    half = 1.96 * math.sqrt((1 + sr_daily**2 / 2) / n)
    root = math.sqrt(TRADING_DAYS)
    return ((sr_daily - half) * root, (sr_daily + half) * root)


def sortino(excess: object, *, exposed: bool = True) -> float | None:
    """Annualized Sortino ratio, mean(excess) / DD * sqrt(252).

    DD = sqrt(mean(min(excess, 0)^2)) over all n days. None when n < 2, DD < 1e-12, or the
    segment has no exposure.
    """
    x = _arr(excess)
    if x.size < 2 or not exposed:
        return None
    dd = math.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    if not dd >= STD_FLOOR:
        return None
    return float(x.mean() / dd * math.sqrt(TRADING_DAYS))


def max_drawdown(equity: object) -> float:
    """min_t (E_t / max_{u<=t} E_u - 1) of an equity path that starts with its baseline point."""
    eq = _arr(equity)
    return float(np.min(eq / np.maximum.accumulate(eq) - 1.0))


def cagr(growth: float, n_days: int) -> float | None:
    """(E_n / E0)^(252 / n) - 1, where growth = E_n / E0. None when growth <= 0 (or on overflow)."""
    if not growth > 0:
        return None
    try:
        return float(growth) ** (TRADING_DAYS / n_days) - 1.0
    except OverflowError:
        return None


def ann_vol(total: object) -> float | None:
    """std(total) * sqrt(252) of daily total returns. None when n < 2."""
    x = _arr(total)
    if x.size < 2:
        return None
    return float(x.std(ddof=1) * math.sqrt(TRADING_DAYS))


def beta_corr(
    excess: object, market_excess: object, *, exposed: bool = True
) -> tuple[float | None, float | None]:
    """OLS slope and correlation of strategy excess returns on SPY excess returns (r_SPY - rf).

    Both None when n < 2, the segment has no exposure, or either std is < 1e-12.
    """
    x, m = _arr(excess), _arr(market_excess)
    if x.size < 2 or not exposed:
        return None, None
    sx, sm = x.std(ddof=1), m.std(ddof=1)
    if not (sx >= STD_FLOOR and sm >= STD_FLOOR):
        return None, None
    cov = float(np.sum((x - x.mean()) * (m - m.mean())) / (x.size - 1))
    return cov / sm**2, cov / (sx * sm)


def pct_days_at_cap(gross_held: object, held: object, max_leverage: float) -> float | None:
    """Share of invested days whose held gross exposure is >= max_leverage * (1 - 1e-9) (§5.6).

    gross_held is sum_i |w_{i,t-1}|, the exposure held into each day; held marks the days whose
    held_id is not null. None when no day is invested. For the trend portfolio (§5.8) the inputs
    are per leg, (days, legs): |w_{i,t-1}| and whether leg i was held into day t, with the
    per-leg cap max_leverage / N, so the share is of invested asset-days.
    """
    g, inv = _arr(gross_held), np.asarray(held, dtype=bool)
    if not inv.any():
        return None
    return float(np.mean(g[inv] >= max_leverage * (1 - CAP_TOLERANCE)))


def _return_fields(total: np.ndarray, excess: np.ndarray, market_excess: object, exposed: bool) -> dict:
    """The fields shared by strategy and benchmark metrics."""
    n = total.size
    if n == 0:
        raise ValueError("a segment needs at least one day")
    growth = np.cumprod(1.0 + total)
    sr_d = sharpe_daily(excess, exposed=exposed)
    beta, corr = beta_corr(excess, market_excess, exposed=exposed)
    return {
        "n_days": n,
        "total_return": float(growth[-1] - 1.0),
        "cagr": cagr(float(growth[-1]), n),
        "ann_vol": ann_vol(total),
        "sharpe": None if sr_d is None else sr_d * math.sqrt(TRADING_DAYS),
        "sharpe_ci95": None if sr_d is None else sharpe_ci95(sr_d, n),
        "sortino": sortino(excess, exposed=exposed),
        "max_drawdown": max_drawdown(
            np.concatenate([[1.0], growth])
        ),  # within the segment, from its baseline
        "beta_spy": beta,
        "corr_spy": corr,
    }


def segment_metrics(
    *,
    total: object,
    excess: object,
    market_excess: object,
    held: object,
    gross_held: object,
    turnover: object,
    cost: object,
    borrow: object,
    trades: pd.DataFrame | None,
) -> Metrics:
    """Strategy metrics for one segment (§8).

    Args:
        total, excess: the segment's daily total and excess returns.
        market_excess: SPY's daily excess return r_SPY - rf on the same days.
        held: True on days whose held_id is not null (a position earns that day's P&L).
        gross_held: sum_i |w_{i,t-1}| on each day.
        turnover, cost, borrow: daily turnover, trading cost and borrow fee (fractions of equity).
        trades: the trades entered in this segment, with columns open, pnl and holding_days.
            None when trades aren't defined, as in the walk-forward's stitched series, whose
            positions span configurations (§9.2): the four trade fields are then None.

    The segment's equity is rebased to its own baseline, so total_return, CAGR and the
    drawdown are measured within the segment.
    """
    tot, exc = _arr(total), _arr(excess)
    pct_in_market = float(np.mean(np.asarray(held, dtype=bool)))
    trade_fields: dict = dict.fromkeys(("n_closed_trades", "n_open_trades", "win_rate", "avg_holding_days"))
    if trades is not None:
        closed = trades.loc[~trades["open"].to_numpy(dtype=bool)]
        n_closed = len(closed)
        trade_fields = {
            "n_closed_trades": n_closed,
            "n_open_trades": len(trades) - n_closed,
            "win_rate": float((closed["pnl"] > 0).mean()) if n_closed else None,
            "avg_holding_days": float(closed["holding_days"].mean()) if n_closed else None,
        }
    return Metrics(
        **_return_fields(tot, exc, market_excess, exposed=pct_in_market > 0),
        pct_days_in_market=pct_in_market,
        **trade_fields,
        ann_turnover=float(np.mean(_arr(turnover)) * TRADING_DAYS),
        ann_cost_drag=float(np.mean(_arr(cost)) * TRADING_DAYS),
        ann_borrow_drag=float(np.mean(_arr(borrow)) * TRADING_DAYS),
        avg_gross_exposure=float(np.mean(_arr(gross_held))),
    )


def benchmark_metrics(*, r: object, rf: object, market_excess: object) -> Metrics:
    """Buy-and-hold benchmark metrics for one segment (§6.4): w = 1, no costs, no cash.

    Args:
        r: the benchmark's daily simple returns (its total return).
        rf: the daily risk-free rate on the same days.
        market_excess: SPY's daily excess return on the same days.
    """
    tot = _arr(r)
    exc = tot - _arr(rf)
    return Metrics(
        **_return_fields(tot, exc, market_excess, exposed=True),
        pct_days_in_market=1.0,
        n_closed_trades=None,
        n_open_trades=None,
        win_rate=None,
        avg_holding_days=None,
        ann_turnover=None,
        ann_cost_drag=None,
        ann_borrow_drag=None,
        avg_gross_exposure=1.0,
    )


# ---------------------------------------------------------------- cost sweep (§8.1)


@dataclass(frozen=True)
class CostSensitivity:
    """Sharpe ratio against one-way cost, from one run with positions held fixed (§8.1).

    cost_bps: the sweep grid in basis points, ascending, including the request's own cost.
    sharpe: per segment ("is", "oos", "full"), one annualized Sharpe per cost; None where the
        §8 Sharpe is undefined, or where the segment contains (or starts after) that cost's
        ruin day.
    break_even_bps: per segment, the one-way cost at which the mean excess return is 0, in
        basis points; None when the segment has no turnover or loses money before costs.
    """

    cost_bps: tuple[float, ...]
    sharpe: dict[str, list[float | None]]
    break_even_bps: dict[str, float | None]


def cost_sweep(
    *,
    gross_excess: object,
    borrow: object,
    turnover: object,
    rf: object,
    held: object,
    segments: dict[str, tuple[int, int]],
    e: int,
    cost_bps: float,
) -> CostSensitivity:
    """Sharpe at each cost of the §8.1 grid, and the break-even cost, per segment.

    Positions don't depend on the cost, so the daily excess return at a one-way cost of c is
    exactly gross_excess - borrow - c * turnover, and one run gives the whole sweep. At the
    request's own cost this is the run's own excess return, computed the same way (§6.1).

    Args:
        gross_excess, borrow, turnover, rf: daily values on the whole run calendar (§6.1).
        held: True on days whose held_id is not null.
        segments: calendar positions [start, end) of "is", "oos" and "full" (§6.3). Only a run
            that was not ruined has a cost sweep, so every segment exists.
        e: the first evaluated day. A cost's ruin day r_c is the first day from e on whose
            unclipped total return at that cost is <= -1.
        cost_bps: the request's own one-way cost in basis points.
    """
    ge, bw, to = _arr(gross_excess), _arr(borrow), _arr(turnover)
    rate, inv = _arr(rf), np.asarray(held, dtype=bool)
    grid = tuple(sorted({*COST_SWEEP_BPS, float(cost_bps)}))
    sharpe_by_cost: dict[str, list[float | None]] = {name: [] for name in segments}
    for bps in grid:
        excess = ge - bw - (bps / 10_000) * to  # the backtester's own expression
        hits = np.flatnonzero(rate[e:] + excess[e:] <= -1.0)
        ruin = e + int(hits[0]) if hits.size else None
        for name, (a, b) in segments.items():
            if ruin is not None and ruin < b:  # the segment contains r_c or starts after it
                sharpe_by_cost[name].append(None)
            else:
                sharpe_by_cost[name].append(sharpe(excess[a:b], exposed=bool(inv[a:b].any())))
    break_even: dict[str, float | None] = {}
    for name, (a, b) in segments.items():
        edge, mean_turnover = float(np.mean(ge[a:b] - bw[a:b])), float(np.mean(to[a:b]))
        ok = mean_turnover > 0 and edge > 0  # else: no trades, or losing before costs
        break_even[name] = edge / mean_turnover * 10_000 if ok else None
    return CostSensitivity(cost_bps=grid, sharpe=sharpe_by_cost, break_even_bps=break_even)


# ---------------------------------------------------------------- rolling and calendar statistics (§8.2)


def rolling_sharpe_beta(
    excess: object, market_excess: object, held: object, window: int = ROLLING_DAYS
) -> tuple[np.ndarray, np.ndarray]:
    """Trailing-window Sharpe ratio and beta to SPY over the evaluated days (§8.2).

    Args:
        excess: the strategy's daily excess returns on the evaluated days, in order.
        market_excess: SPY's daily excess return r_SPY - rf on the same days.
        held: True on days whose held_id is not null.
        window: the trailing window in evaluated days (min_periods = window).

    Returns:
        (sharpe, beta), arrays as long as the inputs. The value on evaluated day i uses days
        i - window + 1 .. i: the annualized §8 Sharpe and the OLS slope on SPY's excess return.
        NaN before the window-th day and where §8's null rules apply: no held day in the
        window, or a standard deviation below 1e-12.
    """
    x, m, inv = _arr(excess), _arr(market_excess), np.asarray(held, dtype=bool)
    n = x.size
    sharpe_out, beta_out = np.full(n, np.nan), np.full(n, np.nan)
    if n < window:
        return sharpe_out, beta_out
    X, M = sliding_window_view(x, window), sliding_window_view(m, window)  # one row per window, no copy
    held_days = np.convolve(inv.astype(np.int64), np.ones(window, dtype=np.int64), mode="valid")
    mean_x, sx, sm = X.mean(axis=1), X.std(axis=1, ddof=1), M.std(axis=1, ddof=1)
    ok = (held_days > 0) & (sx >= STD_FLOOR)
    sharpe_out[window - 1 :] = np.where(ok, mean_x / np.where(ok, sx, 1.0), np.nan) * math.sqrt(TRADING_DAYS)
    cov = np.sum((X - mean_x[:, None]) * (M - M.mean(axis=1, keepdims=True)), axis=1) / (window - 1)
    ok_beta = ok & (sm >= STD_FLOOR)
    beta_out[window - 1 :] = np.where(ok_beta, cov / np.where(ok_beta, sm, 1.0) ** 2, np.nan)
    return sharpe_out, beta_out


@dataclass(frozen=True)
class PeriodReturn:
    """Compounded total returns over a calendar month (month 1-12) or year (month None)."""

    year: int
    month: int | None
    strategy: float
    benchmark: float
    partial: bool


def calendar_returns(
    *,
    dates: pd.DatetimeIndex,
    total: object,
    benchmark: object,
    calendar: pd.DatetimeIndex,
    end: date,
    ruined: bool,
) -> tuple[list[PeriodReturn], list[PeriodReturn]]:
    """Monthly and annual total returns of the evaluation window (§8.2).

    Args:
        dates: the evaluated days, from the first evaluated day to the last one (the ruin
            day if the run was ruined).
        total, benchmark: the strategy's and the benchmark's daily total returns on those days.
        calendar: the whole run calendar, including the warm-up and any days after ruin.
        end: the resolved end date of the request.
        ruined: whether the run ended by ruin.

    Returns:
        (monthly, annual), in date order. Each return is prod(1 + r) - 1 over the period's
        evaluated days. A period is partial when the window doesn't include every run-calendar
        day of it. The window's last period is also partial when the run ended by ruin, or when
        `end` falls before the period's last calendar day.
    """
    growth = pd.DataFrame({"strategy": 1.0 + _arr(total), "benchmark": 1.0 + _arr(benchmark)}, index=dates)
    out: list[list[PeriodReturn]] = []
    for freq in ("M", "Y"):
        periods = dates.to_period(freq)
        compounded = growth.groupby(periods).prod() - 1.0
        in_window = growth.groupby(periods).size()
        on_calendar = pd.Series(0, index=calendar.to_period(freq)).groupby(level=0).size()
        complete = in_window.to_numpy() == on_calendar.reindex(compounded.index).to_numpy()
        last = compounded.index[-1]
        complete[-1] &= not (ruined or end < last.end_time.date())
        out.append(
            [
                PeriodReturn(
                    year=period.year,
                    month=period.month if freq == "M" else None,
                    strategy=float(s_ret),
                    benchmark=float(b_ret),
                    partial=not bool(ok),
                )
                for period, s_ret, b_ret, ok in zip(
                    compounded.index,
                    compounded["strategy"].to_numpy(),
                    compounded["benchmark"].to_numpy(),
                    complete,
                    strict=True,
                )
            ]
        )
    return out[0], out[1]


def build_warnings(
    *,
    metrics: dict[str, Metrics | None],
    trades: pd.DataFrame,
    ruin_date: pd.Timestamp | None,
    pct_at_cap: float | None,
    coint_p_is: float | None = None,
    collinear_is: bool = False,
    collinear_oos: bool = False,
    break_even_full_bps: float | None = None,
    cost_bps: float = 0.0,
    per_leg_cap: bool = False,
) -> list[str]:
    """The §8 warnings that the pure core can detect.

    The data-cleaning warnings (§4.4) come from the loader and are added by the API.
    trades may carry a beta_at_entry column (pairs); NaN there means not applicable.
    per_leg_cap: pct_at_cap is a share of invested asset-days against max_leverage / N (§5.8).
    """
    out: list[str] = []
    if coint_p_is is not None and coint_p_is >= COINT_ALPHA:
        out.append(
            f"In-sample Engle–Granger p = {coint_p_is:.3f} (≥ {COINT_ALPHA}): "
            "no evidence the pair is cointegrated in-sample."
        )
    for flag, label in ((collinear_is, "in-sample"), (collinear_oos, "out-of-sample")):
        if flag:
            out.append(
                f"The {label} Engle–Granger test raised a collinearity warning (the legs are almost "
                "perfectly collinear), so its p-value is null."
            )
    for seg, m in metrics.items():
        if m is not None and m.n_closed_trades is not None and m.n_closed_trades < MIN_CLOSED_TRADES:
            out.append(
                f"{SEGMENT_LABELS[seg]}: {m.n_closed_trades} closed trades (fewer than {MIN_CLOSED_TRADES}), "
                "so trade statistics are unreliable."
            )
    if "beta_at_entry" in trades:
        n_bad = int((trades["beta_at_entry"] <= 0).sum())
        if n_bad:
            out.append(
                f"beta_used was ≤ 0 at {n_bad} {'entry' if n_bad == 1 else 'entries'}: "
                "the hedge adds exposure instead of offsetting it."
            )
    if ruin_date is not None:
        out.append(ruin_warning(ruin_date))
    if pct_at_cap is not None and pct_at_cap > CAP_WARNING_SHARE:
        where = (
            "Legs were at their leverage cap (max_leverage / N) on "
            if per_leg_cap
            else "The position was at the leverage cap on "
        )
        days = "invested asset-days" if per_leg_cap else "invested days"
        out.append(
            f"{where}{pct_at_cap:.0%} of {days} (more than {CAP_WARNING_SHARE:.0%}): "
            "the volatility target is not reachable within max_leverage."
        )
    if break_even_full_bps is not None and break_even_full_bps < BREAK_EVEN_MULTIPLE * cost_bps:
        out.append(
            f"The full-period break-even cost is {break_even_full_bps:.1f} bps, below twice the request's "
            f"{cost_bps:g} bps: the strategy's edge is less than twice its trading cost."
        )
    return out


def ruin_warning(ruin_date: pd.Timestamp) -> str:
    """The §8 ruin warning, shared by the backtest and the walk-forward."""
    return (
        f"Ruin on {ruin_date:%Y-%m-%d}: that day's loss reached 100% of equity, so the evaluation "
        "window ends there (limited liability, no margin calls)."
    )


def walk_forward_warnings(
    *,
    selected: list[object | None],
    test_windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    ruin_date: pd.Timestamp | None,
) -> list[str]:
    """The §8 walk-forward warnings (§9.2).

    Args:
        selected: each fold's chosen configuration, in fold order; None for a flat fold.
        test_windows: each fold's first and last test day.
        ruin_date: the stitched series' ruin day, if any.

    A fold with no eligible configuration is flat and gets a warning. The selection is
    unstable when it changed from the previous fold in more than half the folds.
    """
    out: list[str] = []
    for j, (cell, (first, last)) in enumerate(zip(selected, test_windows, strict=True)):
        if cell is None:
            out.append(
                f"Fold {j} (test {first:%Y-%m-%d} to {last:%Y-%m-%d}): no configuration was eligible, "
                "so the walk-forward was flat."
            )
    n_folds = len(selected)
    changes = sum(selected[j] != selected[j - 1] for j in range(1, n_folds))
    if changes > n_folds / 2:
        out.append(
            f"The selected configuration changed in {changes} of {n_folds} folds: the selected parameters "
            "were unstable across folds."
        )
    if ruin_date is not None:
        out.append(ruin_warning(ruin_date))
    return out
