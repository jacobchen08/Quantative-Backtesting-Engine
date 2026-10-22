"""The backtest report (§8, §11.3): metrics per segment, the benchmark, pairs diagnostics, the
cost sweep, calendar returns, warnings and the chart series of one run. Pure (§0, rule 8)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from app.services import backtester, metrics, stats
from app.services.backtester import BacktestResult
from app.services.metrics import CostSensitivity, Metrics, PeriodReturn
from app.services.pipeline import MARKET, StrategyParams, StrategyRun
from app.services.stats import PairsDiagnostics


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
        gross_exposure, score (not for the trend portfolio), beta and beta_used (pairs only),
        vol_forecast (overlay only, not for the trend portfolio), is_oos, rolling_sharpe and
        rolling_beta.
    legs: trend portfolio only, one frame per ticker on the series index, with the leg's score
        and vol_forecast (unlagged) and its executed position and weight. None otherwise.
    trades: the §7.3 trade records plus ticker (§7.4) and beta_at_entry (NaN unless pairs).
    metrics, benchmark_metrics: keyed "is", "oos" and "full"; "oos" is None if ruin hit in IS.
    cost_sensitivity: the §8.1 cost sweep; None when the run was ruined.
    calendar_monthly, calendar_annual: §8.2 calendar returns of the evaluation window.
    """

    split_date: pd.Timestamp
    first_evaluated_date: pd.Timestamp
    series: pd.DataFrame
    legs: dict[str, pd.DataFrame] | None
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
    market_excess = run.returns[MARKET] - run.rf  # SPY excess return, for beta and correlation
    bench = benchmark_returns(run.returns, p)
    trades = _trades_with_beta(run)
    seg_metrics, bench_metrics = _segment_metrics(run, trades, bench, market_excess)
    sizing_summary = _sizing_summary(run)

    diagnostics = None
    if p.strategy == "pairs":
        diagnostics = pairs_diagnostics(run.prices[legs[0]], run.prices[legs[1]], bt.split_date)

    cost_sensitivity = None
    if not bt.ruined:  # clipping makes excess non-linear in the cost (§8.1)
        cost_sensitivity = metrics.cost_sweep(
            gross_excess=bt.gross_excess,
            borrow=bt.borrow,
            turnover=bt.turnover,
            rf=run.rf,
            held=bt.held,
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
        per_leg_cap=p.strategy == "trend_portfolio",
    )
    series = _series(run, bench, market_excess)
    return BacktestReport(
        split_date=bt.split_date,
        first_evaluated_date=run.prices.index[bt.e],
        series=series,
        legs=_leg_series(run, series.index),
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


def _trades_with_beta(run: StrategyRun) -> pd.DataFrame:
    """The backtest's trade records plus beta_at_entry: the hedge ratio held, read on each trade's
    signal date for pairs (§7.3), and NaN for the other strategies."""
    trades = run.backtest.trades.copy()
    if run.params.strategy == "pairs":
        trades["beta_at_entry"] = run.beta_used.reindex(pd.DatetimeIndex(trades["signal_date"])).to_numpy()
    else:
        trades["beta_at_entry"] = np.nan
    return trades


def _segment_metrics(
    run: StrategyRun, trades: pd.DataFrame, bench: pd.Series, market_excess: pd.Series
) -> tuple[dict[str, Metrics | None], dict[str, Metrics | None]]:
    """§8 metrics of the strategy and of its benchmark on each segment ("is", "oos", "full").

    A segment's trades are those entered in it. A segment that doesn't exist (OOS after a ruin
    in IS) is None for both.
    """
    bt = run.backtest
    strategy: dict[str, Metrics | None] = {}
    benchmark: dict[str, Metrics | None] = {}
    for name, bounds in bt.segment_bounds().items():
        if bounds is None:
            strategy[name] = benchmark[name] = None
            continue
        sl = slice(*bounds)
        entered = trades if name == "full" else trades.loc[trades["segment"] == name]
        strategy[name] = daily_metrics(bt, bt.held.to_numpy(), sl, market_excess, entered)
        benchmark[name] = metrics.benchmark_metrics(
            r=bench.iloc[sl], rf=run.rf.iloc[sl], market_excess=market_excess.iloc[sl]
        )
    return strategy, benchmark


def _sizing_summary(run: StrategyRun) -> SizingSummary | None:
    """The §11.3 sizing block when the overlay is on (always, for the trend portfolio), else None.

    pct_days_at_cap counts invested days at the leverage cap; for the trend portfolio, invested
    asset-days at the per-leg cap max_leverage / N (§5.8).
    """
    p, bt = run.params, run.backtest
    if p.vol_target is None:
        return None
    win = slice(bt.e, bt.stop)
    if p.strategy == "trend_portfolio":
        pct = metrics.pct_days_at_cap(
            bt.w.shift(1).fillna(0.0).abs().iloc[win],
            bt.held_id.iloc[win].notna(),
            p.max_leverage / len(p.tickers),
        )
    else:
        pct = metrics.pct_days_at_cap(bt.gross_held.iloc[win], bt.held.iloc[win], p.max_leverage)
    return SizingSummary(vol_target=p.vol_target, max_leverage=p.max_leverage, pct_days_at_cap=pct)


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
    }
    if run.score is not None:  # the trend portfolio's scores are per leg (_leg_series)
        cols["score"] = run.score.iloc[s0:stop]
    if run.beta is not None:
        cols["beta"] = run.beta.iloc[s0:stop]
        cols["beta_used"] = run.beta_used.iloc[s0:stop]
    if run.vol_forecast is not None:
        cols["vol_forecast"] = run.vol_forecast.iloc[s0:stop]
    cols["is_oos"] = idx >= bt.split_date
    # §8.2: trailing 252 evaluated days, NaN before the 252nd and outside the evaluation window.
    win = slice(bt.e, stop)
    roll_sharpe, roll_beta = metrics.rolling_sharpe_beta(
        bt.excess.iloc[win], market_excess.iloc[win], bt.held.iloc[win]
    )
    before = np.full(bt.e - s0, np.nan)
    cols["rolling_sharpe"] = np.concatenate([before, roll_sharpe])
    cols["rolling_beta"] = np.concatenate([before, roll_beta])
    return pd.DataFrame(cols, index=idx)


def _leg_series(run: StrategyRun, index: pd.DatetimeIndex) -> dict[str, pd.DataFrame] | None:
    """Trend portfolio: each leg's score and vol forecast (unlagged), and its executed position
    and weight, on the chart series' dates. None for the other strategies."""
    if run.leg_scores is None:
        return None
    bt = run.backtest
    return {
        leg: pd.DataFrame(
            {
                "score": run.leg_scores[leg],
                "position": bt.leg_pos[leg],
                "weight": bt.w[leg],
                "vol_forecast": run.leg_vol_forecasts[leg],
            }
        ).reindex(index)
        for leg in run.params.tickers
    }


def benchmark_returns(returns: pd.DataFrame, params: StrategyParams) -> pd.Series:
    """The benchmark's daily total return (§6.4): buy-and-hold of the traded ticker (A, C), of SPY
    (pairs), or an equal-weight, daily-rebalanced, long-only basket of the legs (trend portfolio).
    No costs, no cash, never vol-targeted."""
    if params.strategy == "pairs":
        return returns[MARKET]
    if params.strategy == "trend_portfolio":
        return returns[list(params.tickers)].mean(axis=1).rename("basket")
    return returns[params.tickers[0]]


def daily_metrics(
    daily: BacktestResult | backtester.DailyReturns,
    held: np.ndarray,
    sl: slice,
    market_excess: pd.Series,
    trades: pd.DataFrame | None,
) -> Metrics:
    """§8 metrics of a backtest's daily series (or the walk-forward's stitched one) on the
    calendar positions sl. held marks the days a position earns its P&L, on the whole calendar."""
    return metrics.segment_metrics(
        total=daily.total.iloc[sl],
        excess=daily.excess.iloc[sl],
        market_excess=market_excess.iloc[sl],
        held=held[sl],
        gross_held=daily.gross_held.iloc[sl],
        turnover=daily.turnover.iloc[sl],
        cost=daily.cost.iloc[sl],
        borrow=daily.borrow.iloc[sl],
        trades=trades,
    )
