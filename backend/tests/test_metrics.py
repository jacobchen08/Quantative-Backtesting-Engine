"""Performance metrics, null rules and warnings (§12, test_metrics.py)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.services import metrics

ROOT = math.sqrt(252)
NO_TRADES = pd.DataFrame({"open": pd.Series([], dtype=bool), "pnl": [], "holding_days": []})


def test_sharpe_uses_the_daily_risk_free_rate() -> None:
    r = np.array([0.012, -0.004, 0.007, 0.001, -0.009, 0.015, 0.003])
    rf = np.array([0.0002, 0.0002, 0.0003, 0.0003, 0.0001, 0.0002, 0.0004])
    m = metrics.benchmark_metrics(r=r, rf=rf, market_excess=r - rf)
    excess = r - rf
    assert m.sharpe == pytest.approx(excess.mean() / excess.std(ddof=1) * ROOT, rel=1e-12)
    assert m.sharpe != pytest.approx(r.mean() / r.std(ddof=1) * ROOT, rel=1e-3)  # excess, not total


def test_sortino_uses_the_downside_deviation_over_all_days() -> None:
    x = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    dd_all_days = math.sqrt(np.mean(np.minimum(x, 0.0) ** 2))  # sqrt((1e-4 + 4e-4) / 5) = 0.01
    assert metrics.sortino(x) == pytest.approx(x.mean() / dd_all_days * ROOT, rel=1e-12)
    dd_down_days_only = math.sqrt(np.mean(x[x < 0] ** 2))
    assert metrics.sortino(x) != pytest.approx(x.mean() / dd_down_days_only * ROOT, rel=1e-3)


def test_max_drawdown() -> None:
    assert metrics.max_drawdown([100, 120, 90, 130, 65]) == -0.5
    assert metrics.max_drawdown([100, 101, 102]) == 0.0


def test_cagr_on_a_known_path() -> None:
    assert metrics.cagr(1.21, 504) == pytest.approx(0.1, rel=1e-12)  # +21% over two years
    daily = np.full(504, 1.21 ** (1 / 504) - 1)
    m = metrics.benchmark_metrics(r=daily, rf=np.zeros(504), market_excess=daily)
    assert m.total_return == pytest.approx(0.21, rel=1e-12) and m.cagr == pytest.approx(0.1, rel=1e-10)
    assert metrics.cagr(0.0, 10) is None and metrics.cagr(-0.2, 10) is None


def test_sharpe_confidence_interval() -> None:
    lo, hi = metrics.sharpe_ci95(0.1, 252)
    half = 1.96 * math.sqrt((1 + 0.1**2 / 2) / 252)
    assert lo == pytest.approx((0.1 - half) * ROOT, rel=1e-12)
    assert hi == pytest.approx((0.1 + half) * ROOT, rel=1e-12)
    x = np.random.default_rng(0).normal(0.001, 0.01, 252)
    m = metrics.benchmark_metrics(r=x, rf=np.zeros(252), market_excess=x)
    assert m.sharpe_ci95 == pytest.approx(metrics.sharpe_ci95(x.mean() / x.std(ddof=1), 252), rel=1e-12)
    assert m.sharpe_ci95[0] < m.sharpe < m.sharpe_ci95[1]


def test_a_constant_series_has_a_null_sharpe_not_1e12() -> None:
    arr = np.full(252, 1 / 3)
    assert 0 < arr.std(ddof=1) < 1e-15  # numpy leaves rounding noise...
    assert abs(arr.mean() / arr.std(ddof=1)) > 1e12  # ...so a naive Sharpe explodes
    assert metrics.sharpe(arr) is None
    series = pd.Series(0.3, index=range(252))
    assert series.std() == 0.0
    assert metrics.sharpe(series) is None
    assert metrics.sharpe(np.array([0.01])) is None  # n < 2
    assert metrics.sharpe(np.array([0.01, -0.02]), exposed=False) is None


def test_no_downside_days_give_a_null_sortino() -> None:
    assert metrics.sortino(np.array([0.01, 0.0, 0.02, 0.005])) is None
    assert metrics.sortino(np.array([0.01, -0.01])) is not None


def test_beta_and_correlation_to_spy() -> None:
    g = np.random.default_rng(1)
    mkt = g.normal(0, 0.01, 300)
    strat = 0.6 * mkt + g.normal(0, 0.005, 300)
    beta, corr = metrics.beta_corr(strat, mkt)
    assert beta == pytest.approx(np.polyfit(mkt, strat, 1)[0], rel=1e-10)
    assert corr == pytest.approx(np.corrcoef(strat, mkt)[0, 1], rel=1e-12)
    assert metrics.beta_corr(strat, np.full(300, 0.3)) == (None, None)  # a flat market
    assert metrics.beta_corr(strat, mkt, exposed=False) == (None, None)
    assert metrics.beta_corr(strat[:1], mkt[:1]) == (None, None)


def test_segment_metrics_on_a_hand_built_segment() -> None:
    total = np.array([0.01, -0.02, 0.0, 0.015])
    trades = pd.DataFrame(
        {"open": [False, False, True], "pnl": [0.004, -0.002, 0.01], "holding_days": [3, 1, 2]}
    )
    turnover = np.array([1.0, 0.0, 1.0, 2.0])
    m = metrics.segment_metrics(
        total=total,
        excess=total - 0.0001,
        market_excess=np.array([0.004, -0.01, 0.002, 0.006]),
        held=np.array([True, True, False, True]),
        gross_held=np.array([1.0, 1.0, 0.0, 2.0]),
        turnover=turnover,
        cost=0.001 * turnover,
        borrow=np.array([0.0002, 0.0001, 0.0, 0.0001]),
        trades=trades,
    )
    assert m.n_days == 4
    assert m.total_return == pytest.approx(1.01 * 0.98 * 1.0 * 1.015 - 1, rel=1e-12)
    assert m.max_drawdown == pytest.approx(-0.02, rel=1e-12)
    assert m.pct_days_in_market == 0.75
    assert (m.n_closed_trades, m.n_open_trades, m.win_rate, m.avg_holding_days) == (2, 1, 0.5, 2.0)
    assert m.ann_turnover == pytest.approx(252.0) and m.ann_cost_drag == pytest.approx(0.252)
    assert m.ann_borrow_drag == pytest.approx(0.0001 * 252, rel=1e-12)  # mean(borrow) * 252
    assert m.avg_gross_exposure == 1.0


def test_no_trades_give_a_null_win_rate() -> None:
    m = metrics.segment_metrics(
        total=np.full(5, 0.0001),
        excess=np.zeros(5),
        market_excess=np.array([0.01, -0.01, 0.02, 0.0, 0.01]),
        held=np.zeros(5, dtype=bool),
        gross_held=np.zeros(5),
        turnover=np.zeros(5),
        cost=np.zeros(5),
        borrow=np.zeros(5),
        trades=NO_TRADES,
    )
    assert (m.n_closed_trades, m.n_open_trades, m.win_rate, m.avg_holding_days) == (0, 0, None, None)
    assert m.sharpe is None and m.sortino is None and m.beta_spy is None and m.pct_days_in_market == 0.0


def test_benchmark_metrics_have_null_trade_and_cost_fields() -> None:
    r = np.random.default_rng(2).normal(0.0005, 0.01, 100)
    m = metrics.benchmark_metrics(r=r, rf=np.full(100, 0.0001), market_excess=r - 0.0001)
    assert (m.n_closed_trades, m.n_open_trades, m.win_rate, m.avg_holding_days) == (None, None, None, None)
    assert (m.ann_turnover, m.ann_cost_drag, m.ann_borrow_drag) == (None, None, None)
    assert (m.pct_days_in_market, m.avg_gross_exposure) == (1.0, 1.0)
    assert m.beta_spy == pytest.approx(1.0) and m.corr_spy == pytest.approx(1.0)


def test_warnings() -> None:
    base = metrics.benchmark_metrics(
        r=np.array([0.01, -0.01]), rf=np.zeros(2), market_excess=np.array([0.01, -0.01])
    )
    many, few = replace(base, n_closed_trades=12), replace(base, n_closed_trades=3)
    trades = pd.DataFrame({"open": [False], "pnl": [0.01], "holding_days": [2], "beta_at_entry": [-0.2]})
    out = metrics.build_warnings(
        metrics={"is": many, "oos": few, "full": many},
        trades=trades,
        ruin_date=pd.Timestamp("2021-03-04"),
        pct_at_cap=0.4,
        coint_p_is=0.31,
        collinear_oos=True,
    )
    assert len(out) == 6
    text = "\n".join(out)
    for phrase in (
        "no evidence the pair is cointegrated in-sample",
        "out-of-sample Engle–Granger test raised a collinearity warning",
        "Out-of-sample: 3 closed trades",
        "trade statistics are unreliable",
        "the hedge adds exposure instead of offsetting it",
        "Ruin on 2021-03-04",
        "the volatility target is not reachable within max_leverage",
    ):
        assert phrase in text, phrase
    quiet = metrics.build_warnings(
        metrics={"is": many, "oos": None, "full": many},
        trades=trades.assign(beta_at_entry=0.9),
        ruin_date=None,
        pct_at_cap=0.25,  # "more than 25%" is strict
        coint_p_is=0.01,
    )
    assert quiet == []


def test_the_break_even_warning() -> None:
    base = metrics.benchmark_metrics(
        r=np.array([0.01, -0.01]), rf=np.zeros(2), market_excess=np.array([0.01, -0.01])
    )
    many = replace(base, n_closed_trades=12)
    common = {"metrics": {"is": many, "oos": None, "full": many}, "trades": NO_TRADES, "ruin_date": None}
    low = metrics.build_warnings(**common, pct_at_cap=None, break_even_full_bps=15.0, cost_bps=10.0)
    assert len(low) == 1 and "15.0 bps" in low[0]
    assert "the strategy's edge is less than twice its trading cost" in low[0]
    for be in (20.0, None):  # "below" is strict; a null break-even is skipped
        assert metrics.build_warnings(**common, pct_at_cap=None, break_even_full_bps=be, cost_bps=10.0) == []
