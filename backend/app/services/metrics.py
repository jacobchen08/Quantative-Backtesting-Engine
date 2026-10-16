"""Performance metrics and warnings (§8). Pure (§0, rule 8).

Metric functions take the daily values of one segment (numpy arrays or pandas Series) as
decimals: 0.01 = 1%. A statistic that is undefined returns None, never NaN or inf (§2.4).
Standard deviations use ddof=1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
STD_FLOOR = 1e-12  # a std (or downside deviation) below this makes a ratio undefined
CAP_TOLERANCE = 1e-9  # §5.6: "at the cap" means gross >= max_leverage * (1 - 1e-9)
MIN_CLOSED_TRADES = 10
CAP_WARNING_SHARE = 0.25
COINT_ALPHA = 0.05
SEGMENT_LABELS = {"is": "In-sample", "oos": "Out-of-sample", "full": "Full period"}


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
    held_id is not null. None when no day is invested.
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
    trades: pd.DataFrame,
) -> Metrics:
    """Strategy metrics for one segment (§8).

    Args:
        total, excess: the segment's daily total and excess returns.
        market_excess: SPY's daily excess return r_SPY - rf on the same days.
        held: True on days whose held_id is not null (a position earns that day's P&L).
        gross_held: sum_i |w_{i,t-1}| on each day.
        turnover, cost: daily turnover and cost (fractions of equity).
        trades: the trades entered in this segment, with columns open, pnl and holding_days.

    The segment's equity is rebased to its own baseline, so total_return, CAGR and the
    drawdown are measured within the segment.
    """
    tot, exc = _arr(total), _arr(excess)
    pct_in_market = float(np.mean(np.asarray(held, dtype=bool)))
    closed = trades.loc[~trades["open"].to_numpy(dtype=bool)]
    n_closed = len(closed)
    return Metrics(
        **_return_fields(tot, exc, market_excess, exposed=pct_in_market > 0),
        pct_days_in_market=pct_in_market,
        n_closed_trades=n_closed,
        n_open_trades=len(trades) - n_closed,
        win_rate=float((closed["pnl"] > 0).mean()) if n_closed else None,
        avg_holding_days=float(closed["holding_days"].mean()) if n_closed else None,
        ann_turnover=float(np.mean(_arr(turnover)) * TRADING_DAYS),
        ann_cost_drag=float(np.mean(_arr(cost)) * TRADING_DAYS),
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
        avg_gross_exposure=1.0,
    )


def build_warnings(
    *,
    metrics: dict[str, Metrics | None],
    trades: pd.DataFrame,
    ruin_date: pd.Timestamp | None,
    pct_at_cap: float | None,
    coint_p_is: float | None = None,
    collinear_is: bool = False,
    collinear_oos: bool = False,
) -> list[str]:
    """The §8 warnings that the pure core can detect.

    The data-cleaning warnings (§4.4) come from the loader and are added by the API.
    trades may carry a beta_at_entry column (pairs); NaN there means not applicable.
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
        out.append(
            f"Ruin on {ruin_date:%Y-%m-%d}: that day's loss reached 100% of equity, so the evaluation "
            "window ends there (limited liability, no margin calls)."
        )
    if pct_at_cap is not None and pct_at_cap > CAP_WARNING_SHARE:
        out.append(
            f"The position was at the leverage cap on {pct_at_cap:.0%} of invested days (more than "
            f"{CAP_WARNING_SHARE:.0%}): the volatility target is not reachable within max_leverage."
        )
    return out
