"""Backtest accounting, trades, evaluation window, ruin and leverage (§12, test_backtester.py)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.services import metrics
from app.services.backtester import BacktestResult, InsufficientDataError, resolve_split, run_backtest
from app.services.monte_carlo import MonteCarloInputError
from app.services.pipeline import StrategyParams, StrategyRun, build_report, run_strategy, strategy_mc_returns

OVERLAYS = [None, "daily", "weekly", "monthly"]

# ---------------------------------------------------------------- 1. look-ahead perturbation

BASE_CONFIGS = [
    ("mean-reversion", {"strategy": "mean_reversion", "tickers": ("Y",), "L": 60}),
    ("pairs-fixed", {"strategy": "pairs", "tickers": ("Y", "X"), "L": 60, "hedge_mode": "fixed_at_entry"}),
    ("pairs-rolling", {"strategy": "pairs", "tickers": ("Y", "X"), "L": 60, "hedge_mode": "rolling"}),
    ("trend-monthly", {"strategy": "trend", "tickers": ("X",), "L": 126, "signal_frequency": "monthly"}),
    ("trend-daily", {"strategy": "trend", "tickers": ("X",), "L": 126, "signal_frequency": "daily"}),
]
PERTURBATION_CASES = [
    pytest.param(
        {
            **cfg,
            "execution_lag": i % 3,
            "vol_target": None if vol is None else 0.10,
            "vol_refresh": vol or "weekly",
        },
        id=f"{name}-{vol or 'no-overlay'}-k{i % 3}",
    )
    for i, (name, cfg, vol) in enumerate((n, c, v) for n, c in BASE_CONFIGS for v in OVERLAYS)
]
CUTS = (300, 480, 655, 870)


def _outputs(run: StrategyRun) -> dict[str, pd.Series | pd.DataFrame]:
    out = {
        "score": run.score,
        "signal": run.signal,
        "weights": run.weights,
        "pos": run.backtest.pos,
        "excess": run.backtest.excess,
        "total": run.backtest.total,
    }
    for name in ("vol_forecast", "beta", "beta_used"):
        if getattr(run, name) is not None:
            out[name] = getattr(run, name)
    return out


@pytest.mark.parametrize("config", PERTURBATION_CASES)
def test_no_lookahead_under_future_perturbation(prices: pd.DataFrame, varying_rf, config: dict) -> None:
    params = StrategyParams(**config)
    rf = varying_rf(prices.index)
    base = _outputs(run_strategy(prices, rf, params))
    rng = np.random.default_rng(99)
    for cut in CUTS:
        future = slice(cut + 1, None)
        noisy = prices.copy()
        noisy.iloc[future] *= np.exp(rng.normal(0.0, 0.05, noisy.iloc[future].shape))
        noisy_rf = rf.copy()
        noisy_rf.iloc[future] *= np.exp(rng.normal(0.0, 0.2, len(rf) - cut - 1))
        pert = _outputs(run_strategy(noisy, noisy_rf, params))
        assert pert.keys() == base.keys()
        for name, a in base.items():
            same = np.array_equal(
                a.iloc[: cut + 1].to_numpy(), pert[name].iloc[: cut + 1].to_numpy(), equal_nan=True
            )
            assert same, f"{name} changed on or before the cut at {cut}"
        for name in ("score", "total"):  # the perturbation did reach the later dates
            after_a, after_b = base[name].iloc[cut + 1 :].to_numpy(), pert[name].iloc[cut + 1 :].to_numpy()
            assert not np.array_equal(after_a, after_b, equal_nan=True)


# ---------------------------------------------------------------- 2. hand-computed toy cases

DATES = pd.bdate_range("2024-01-01", periods=6).as_unit("ns")
TOY_R = pd.DataFrame({"A": [0.0, 0.01, 0.02, -0.03, 0.04, 0.05]}, index=DATES)


def toy(signal: list[float], k: int, *, split: int = 4) -> BacktestResult:
    """§12 toy setup: c = 10 bps, rf = 0, t0 = 1, W = s."""
    s = pd.Series(signal, index=DATES, dtype=float)
    return run_backtest(
        R=TOY_R,
        W=s.to_frame("A"),
        s=s,
        rf=pd.Series(0.0, index=DATES),
        t0=1,
        k=k,
        cost_bps=10.0,
        E0=1.0,
        split_date=DATES[split],
    )


def trade_rows(bt: BacktestResult) -> list[tuple]:
    """(id, entry day, exit day, holding days, open, pnl) with days as positions 0..5."""
    day = {d: i for i, d in enumerate(DATES)}
    return [
        (
            int(t.id),
            day[t.entry_date],
            None if pd.isna(t.exit_date) else day[t.exit_date],
            int(t.holding_days),
            bool(t.open),
            round(float(t.pnl), 12),
        )
        for t in bt.trades.itertuples()
    ]


def test_toy_single_trade_exits_on_day_3() -> None:
    bt = toy([0, 1, 1, 0, 0, 0], k=0)
    assert trade_rows(bt) == [(1, 1, 3, 2, False, -0.012)]  # the naive answer, day 2, is wrong
    t = bt.trades.iloc[0]
    assert (t.signal_date, t.exit_signal_date) == (DATES[1], DATES[3])


def test_toy_direct_flip_splits_the_flip_cost() -> None:
    bt = toy([0, 1, 1, -1, -1, 0], k=0)
    assert bt.cost.iloc[3] == pytest.approx(0.002)  # 2c on the flip day
    # -0.012 and -0.092 hold only if the flip day's 2c is split c / c between the two trades.
    assert trade_rows(bt) == [(1, 1, 3, 2, False, -0.012), (2, 3, 5, 2, False, -0.092)]
    assert bt.trades["side"].tolist() == ["long", "short"]


def test_toy_with_one_day_lag() -> None:
    bt = toy([0, 1, 1, 0, 0, 0], k=1)
    assert np.allclose(bt.excess.to_numpy(), [0, 0, -0.001, -0.03, 0.039, 0], rtol=0, atol=1e-15)
    assert trade_rows(bt) == [(1, 2, 4, 2, False, 0.008)]
    t = bt.trades.iloc[0]
    assert (t.signal_date, t.exit_signal_date) == (DATES[1], DATES[3])  # k = 1 trading day earlier


def test_toy_entry_on_the_final_day_is_open() -> None:
    bt = toy([0, 0, 0, 0, 0, 1], k=0)
    assert trade_rows(bt) == [(1, 5, None, 0, True, -0.001)]
    assert pd.isna(bt.trades.iloc[0].exit_signal_date)


def test_toy_flip_on_the_final_day() -> None:
    bt = toy([0, 1, 1, 1, 1, -1], k=0)
    assert trade_rows(bt) == [(1, 1, 5, 4, False, 0.078), (2, 5, None, 0, True, -0.001)]


# ---------------------------------------------------------------- 3-6. costs, cash, reconciliation, leverage


def test_costs(prices: pd.DataFrame, rf: pd.Series) -> None:
    run = run_strategy(prices, rf, StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60, cost_bps=10.0))
    bt, c = run.backtest, 10.0 / 10_000
    assert (bt.excess <= bt.gross_excess).all()
    no_trade = bt.turnover == 0
    assert no_trade.any() and (bt.excess[no_trade] == bt.gross_excess[no_trade]).all()
    assert bt.cost.sum() == pytest.approx(c * bt.turnover.sum(), rel=1e-12)
    entry = bt.trade_id.first_valid_index()  # the first entry, from flat
    w_y, w_x = bt.w.loc[entry, "Y"], bt.w.loc[entry, "X"]
    assert w_y != 0 and w_x != 0
    assert bt.cost.loc[entry] == pytest.approx(c * (abs(w_y) + abs(w_x)), rel=1e-12)  # both legs are charged
    assert bt.turnover.loc[entry] == pytest.approx(1.0, rel=1e-12)


def test_a_strategy_that_never_trades_earns_exactly_rf(prices: pd.DataFrame, varying_rf) -> None:
    rf = varying_rf(prices.index)
    # With L = 20 a z-score can't exceed 19 / sqrt(20) = 4.25 (Samuelson), so z_entry = 5 never trades.
    run = run_strategy(
        prices, rf, StrategyParams(strategy="mean_reversion", tickers=("Y",), L=20, z_entry=5.0)
    )
    bt, report = run.backtest, build_report(run)
    assert bt.trades.empty and (bt.pos == 0).all()
    assert (bt.excess == 0.0).all() and (bt.total == rf).all()
    for m in report.metrics.values():
        assert (
            m.sharpe is None and m.sharpe_ci95 is None and m.sortino is None and m.pct_days_in_market == 0.0
        )


def test_trade_pnl_reconciles_with_excess_returns(synthetic_prices, varying_rf) -> None:
    rng = np.random.default_rng(2024)
    kinds = [("pairs", "fixed_at_entry"), ("pairs", "rolling"), ("mean_reversion", None), ("trend", None)]
    combos = [(s, h, v, k) for s, h in kinds for v in OVERLAYS for k in (0, 1, 2)]  # 48 combinations
    worst, runs, resized = 0.0, 0, 0
    for rep in range(3):
        for strategy, hedge, vol, k in combos:
            prices = synthetic_prices(n=int(rng.integers(700, 1100)), seed=int(rng.integers(10**9)))
            rf = varying_rf(prices.index, seed=int(rng.integers(10**9)))
            z_entry = float(rng.uniform(1.0, 2.5))
            params = StrategyParams(
                strategy=strategy,
                tickers=("Y", "X") if strategy == "pairs" else ("X",),
                L=int(rng.choice([21, 40, 63])) if strategy == "trend" else int(rng.choice([20, 40, 60, 90])),
                z_entry=z_entry,
                z_exit=float(rng.uniform(0.0, 0.9 * z_entry)),
                signal_frequency="daily" if rep % 2 else "monthly",
                hedge_mode=hedge or "fixed_at_entry",
                execution_lag=k,
                cost_bps=float(rng.uniform(0.0, 30.0)),
                vol_target=None if vol is None else float(rng.uniform(0.05, 0.25)),
                vol_refresh=vol or "weekly",
                max_leverage=float(rng.uniform(1.0, 3.0)),
            )
            bt = run_strategy(prices, rf, params).backtest
            worst = max(worst, abs(bt.pnl.sum() - bt.excess.iloc[bt.e : bt.stop].sum()))
            in_trade = bt.held_id.eq(bt.trade_id) & (bt.turnover > 0)  # resized inside a trade
            resized += int(in_trade.sum())
            runs += 1
    assert runs >= 100 and resized > 0
    assert worst < 1e-9


def test_open_trades_are_excluded_from_the_win_rate() -> None:
    bt = toy([0, 1, 1, 1, 1, -1], k=0)  # trade 1 wins and closes; trade 2 is open and losing
    sl = slice(bt.e, bt.stop)
    m = metrics.segment_metrics(
        total=bt.total.iloc[sl],
        excess=bt.excess.iloc[sl],
        market_excess=TOY_R["A"].iloc[sl],
        held=bt.held_id.iloc[sl].notna(),
        gross_held=bt.gross_held.iloc[sl],
        turnover=bt.turnover.iloc[sl],
        cost=bt.cost.iloc[sl],
        trades=bt.trades,
    )
    assert (m.n_closed_trades, m.n_open_trades, m.win_rate, m.avg_holding_days) == (1, 1, 1.0, 4.0)


def test_leverage_is_financed_at_rf() -> None:
    idx = pd.bdate_range("2020-01-01", periods=10).as_unit("ns")
    R = pd.DataFrame({"A": [0.0, 0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, 0.0, 0.01]}, index=idx)
    rf = pd.Series(0.0002, index=idx)
    s = pd.Series([0.0] + [1.0] * 9, index=idx)
    bt = run_backtest(
        R=R, W=(2 * s).to_frame("A"), s=s, rf=rf, t0=1, k=0, cost_bps=0.0, E0=1.0, split_date=idx[6]
    )
    held = bt.held_id.notna()
    assert held.sum() == 8
    expected = rf + 2 * (R["A"] - rf)  # the cash weight is -1 and also earns (pays) rf
    assert np.array_equal(bt.total[held].to_numpy(), expected[held].to_numpy())


# ---------------------------------------------------------------- 7. evaluation window and series


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_first_evaluated_day_and_baseline(prices: pd.DataFrame, rf: pd.Series, k: int) -> None:
    run = run_strategy(
        prices, rf, StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60, execution_lag=k)
    )
    bt, report = run.backtest, build_report(run)
    assert bt.e == run.t0 + k and report.first_evaluated_date == prices.index[run.t0 + k]
    baseline = prices.index[bt.e - 1]
    assert bt.equity.index[0] == baseline and bt.equity.iloc[0] == 100_000.0
    row = report.series.loc[baseline]
    assert (row.equity, row.benchmark_equity, row.drawdown) == (100_000.0, 100_000.0, 0.0)
    assert np.isnan(row.daily_return)
    assert report.series.index[0] == prices.index[min(run.t0, bt.e - 1)]
    assert report.series.index[-1] == prices.index[-1]
    assert bt.gross_held.iloc[bt.e] == 0.0  # nothing is held into the first evaluated day


def test_a_loss_on_the_first_evaluated_day_is_a_drawdown() -> None:
    bt = toy([0, 1, 1, 0, 0, 0], k=0)
    assert bt.total.iloc[1] == pytest.approx(-0.001)  # the entry cost
    assert bt.drawdown.iloc[0] == 0.0 and bt.drawdown.iloc[1] < 0  # the running max includes the baseline


def test_a_trade_straddling_the_split_belongs_to_is() -> None:
    bt = toy([0, 1, 1, 1, 1, 0], k=0, split=3)
    t = bt.trades.iloc[0]
    assert t.entry_date < DATES[3] <= t.exit_date
    assert t.segment == "is"


def test_k2_signal_on_t0_is_inside_the_series(prices: pd.DataFrame, rf: pd.Series) -> None:
    params = StrategyParams(
        strategy="trend", tickers=("SPY",), L=252, signal_frequency="daily", execution_lag=2
    )
    run = run_strategy(prices, rf, params)
    report = build_report(run)
    first = report.trades.iloc[0]
    assert first.signal_date == prices.index[run.t0]
    assert first.signal_date in report.series.index and report.series.index[0] == prices.index[run.t0]
    before_baseline = report.series.index < prices.index[run.backtest.e - 1]
    assert before_baseline.sum() == 1
    for col in ("equity", "benchmark_equity", "drawdown", "daily_return"):
        assert report.series.loc[before_baseline, col].isna().all()
    assert report.series["daily_return"].iloc[:2].isna().all()  # null up to and including the baseline


@pytest.mark.parametrize("strategy, tickers", [("pairs", ("Y", "X")), ("trend", ("SPY",))])
def test_compounding_daily_returns_reproduces_equity(
    prices: pd.DataFrame, rf: pd.Series, strategy: str, tickers: tuple[str, ...]
) -> None:
    report = build_report(run_strategy(prices, rf, StrategyParams(strategy=strategy, tickers=tickers, L=126)))
    series = report.series
    after = series["daily_return"].notna()
    rebuilt = 100_000.0 * np.cumprod(1.0 + series.loc[after, "daily_return"].to_numpy())
    np.testing.assert_allclose(rebuilt, series.loc[after, "equity"].to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("strategy, tickers", [("mean_reversion", ("Y",)), ("pairs", ("Y", "X"))])
def test_l120_on_1000_days_leaves_616_is_and_264_oos(
    prices: pd.DataFrame, rf: pd.Series, strategy: str, tickers: tuple[str, ...]
) -> None:
    report = build_report(run_strategy(prices, rf, StrategyParams(strategy=strategy, tickers=tickers, L=120)))
    m = report.metrics
    assert (m["full"].n_days, m["is"].n_days, m["oos"].n_days) == (880, 616, 264)


def test_trend_at_its_maximum_lookback_still_fits(prices: pd.DataFrame, rf: pd.Series) -> None:
    report = build_report(run_strategy(prices, rf, StrategyParams(strategy="trend", tickers=("SPY",), L=504)))
    m = report.metrics
    assert m["is"].n_days >= 252 and m["oos"].n_days >= 126
    assert m["full"].n_days == m["is"].n_days + m["oos"].n_days


# ---------------------------------------------------------------- 8. input validation and resolve_split


def test_backtester_input_validation() -> None:
    s = pd.Series([0, 1, 1, 0, 0, 0], index=DATES, dtype=float)
    zero = pd.Series(0.0, index=DATES)
    good = {
        "R": TOY_R,
        "W": s.to_frame("A"),
        "s": s,
        "rf": zero,
        "t0": 1,
        "k": 0,
        "cost_bps": 10.0,
        "E0": 1.0,
        "split_date": DATES[4],
    }
    R_nan = TOY_R.copy()
    R_nan.iloc[3, 0] = np.nan
    run_backtest(**good)
    bad = {
        "mismatched column labels": {"W": s.to_frame("B")},
        "an rf index that differs from R's": {"rf": pd.Series(0.0, index=DATES + pd.Timedelta(days=1))},
        "a non-zero signal before t0": {"t0": 2},
        "zero IS days": {"split_date": DATES[1]},
        "zero OOS days": {"split_date": DATES[-1] + pd.Timedelta(days=1)},
        # Additions to the §6.1 list, which the trade bookkeeping relies on:
        "a weight on a flat day": {"W": pd.DataFrame({"A": [0, 1, 1, 0, 0.5, 0]}, index=DATES, dtype=float)},
        "a signal outside {-1, 0, 1}": {"s": s * 0.5, "W": (s * 0.5).to_frame("A")},
        "a NaN return after the first row": {"R": R_nan},
    }
    for name, change in bad.items():
        with pytest.raises(ValueError):
            run_backtest(**{**good, **change})
            pytest.fail(f"accepted {name}")
    with pytest.raises(TypeError):
        run_backtest(**{**good, "rf": zero.to_numpy()})


@pytest.mark.parametrize("n", [650, 880, 1000])
def test_resolve_split_default_is_the_70pct_index(n: int) -> None:
    ev = pd.bdate_range("2015-01-01", periods=n).as_unit("ns")
    # floor(0.7 n) exactly: for n = 650, the float 0.7 * 650 is 454.99999999999994.
    assert resolve_split(ev, None) == ev[7 * n // 10]


def test_resolve_split_rolls_a_weekend_forward() -> None:
    ev = pd.bdate_range("2020-01-01", periods=500).as_unit("ns")
    assert resolve_split(ev, date(2020, 12, 26)) == pd.Timestamp("2020-12-28")  # Saturday -> Monday


@pytest.mark.parametrize("n_is, n_oos, ok", [(251, 200, False), (300, 125, False), (252, 126, True)])
def test_resolve_split_enforces_252_is_and_126_oos(n_is: int, n_oos: int, ok: bool) -> None:
    ev = pd.bdate_range("2015-01-01", periods=n_is + n_oos).as_unit("ns")
    if ok:
        assert resolve_split(ev, ev[n_is]) == ev[n_is]
    else:
        with pytest.raises(InsufficientDataError, match=f"{n_is} in-sample and {n_oos} out-of-sample"):
            resolve_split(ev, ev[n_is])


# ---------------------------------------------------------------- 9. ruin


@pytest.mark.parametrize("crash", [-0.5, -0.6])
def test_a_2x_levered_long_hit_by_a_big_down_day_is_ruined(crash: float) -> None:
    idx = pd.bdate_range("2020-01-01", periods=10).as_unit("ns")
    R = pd.DataFrame({"A": [0.0, 0.01, 0.0, crash, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    s = pd.Series([0.0] + [1.0] * 9, index=idx)
    bt = run_backtest(
        R=R,
        W=(2 * s).to_frame("A"),
        s=s,
        rf=pd.Series(0.0, index=idx),
        t0=1,
        k=0,
        cost_bps=0.0,
        E0=100.0,
        split_date=idx[6],
    )
    assert bt.ruined and bt.ruin_date == idx[3] and bt.stop == 4  # the window ends at r
    assert bt.total.iloc[3] == -1.0 and bt.equity.iloc[-1] == 0.0 and bt.drawdown.iloc[-1] == -1.0
    assert bt.trades["open"].tolist() == [True] and bt.trades["holding_days"].tolist() == [2]
    assert bt.segment_bounds()["oos"] is None
    # At -60% the unclipped day is -120%; the clipped excess is credited to the trade held into r.
    assert abs(bt.pnl.sum() - bt.excess.iloc[bt.e : bt.stop].sum()) < 1e-12


def crash_market(crash_at: int, n: int = 1000, seed: int = 11) -> pd.DataFrame:
    """A calm uptrend (0.1% a day, 0.5% daily vol) with one -50% day at position crash_at."""
    g = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n).as_unit("ns")
    log_r = g.normal(0.001, 0.005, n)
    log_r[crash_at] = np.log(0.5)
    return pd.DataFrame({"SPY": 100.0 * np.exp(np.cumsum(log_r))}, index=idx)


# Trend + a 50% vol target on a ~8%-vol asset: the scale sits at the 2x cap.
LEVERED_TREND = StrategyParams(
    strategy="trend",
    tickers=("SPY",),
    L=252,
    signal_frequency="daily",
    vol_target=0.5,
    max_leverage=2.0,
    vol_refresh="daily",
)


def test_ruin_in_sample_ends_the_window_and_nulls_oos() -> None:
    r = 500
    prices = crash_market(r)
    rf = pd.Series(0.02 / 252, index=prices.index)
    run = run_strategy(prices, rf, LEVERED_TREND)
    bt, report = run.backtest, build_report(run)
    assert bt.w.iloc[r - 1, 0] == 2.0  # 2x long held into the crash
    assert report.ruined and report.ruin_date == prices.index[r] and bt.stop == r + 1
    assert bt.total.iloc[r] == -1.0 and bt.equity.iloc[-1] == 0.0 and bt.drawdown.iloc[-1] == -1.0
    assert report.series.index[-1] == prices.index[r] and report.series["equity"].iloc[-1] == 0.0
    last = report.trades.iloc[-1]
    assert last.open and pd.isna(last.exit_date)
    assert report.metrics["oos"] is None and report.benchmark_metrics["oos"] is None
    assert report.metrics["is"].n_days == r - bt.e + 1 and report.metrics["is"].max_drawdown == -1.0
    assert report.metrics["is"].cagr is None
    assert any(w.startswith("Ruin on") for w in report.warnings)
    assert abs(bt.pnl.sum() - bt.excess.iloc[bt.e : bt.stop].sum()) < 1e-9
    for window in ("full", "is"):
        with pytest.raises(MonteCarloInputError, match="ruin day"):
            strategy_mc_returns(run, window)


def test_ruin_on_the_first_oos_day_leaves_a_one_day_segment() -> None:
    # Ruin can't happen on the first evaluated day (nothing is held into it), so the one-day
    # segment comes from ruin on the split date, which leaves one OOS day (§12).
    r = 700
    prices = crash_market(r)
    rf = pd.Series(0.02 / 252, index=prices.index)
    run = run_strategy(prices, rf, replace(LEVERED_TREND, split_date=prices.index[r].date()))
    report = build_report(run)
    oos = report.metrics["oos"]
    assert report.ruin_date == prices.index[r] == report.split_date and oos.n_days == 1
    for field in ("ann_vol", "sharpe", "sharpe_ci95", "sortino", "beta_spy", "corr_spy", "cagr"):
        assert getattr(oos, field) is None, field
    assert oos.total_return == -1.0 and oos.max_drawdown == -1.0
    bench = report.benchmark_metrics["oos"]
    assert bench.n_days == 1 and bench.ann_vol is None and bench.sharpe is None
    R, exposed = strategy_mc_returns(run, "is")  # the IS window doesn't contain the ruin day
    assert exposed and len(R) == r - run.backtest.e
    with pytest.raises(MonteCarloInputError):
        strategy_mc_returns(run, "full")


# ---------------------------------------------------------------- 10. pct_days_at_cap


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize(
    "strategy, tickers, target", [("trend", ("SPY",), 0.20), ("pairs", ("Y", "X"), 0.10)]
)
def test_pct_days_at_cap(
    prices: pd.DataFrame, rf: pd.Series, k: int, strategy: str, tickers: tuple[str, ...], target: float
) -> None:
    cap = 1.25
    params = StrategyParams(
        strategy=strategy,
        tickers=tickers,
        L=126 if strategy == "trend" else 60,
        signal_frequency="daily",
        vol_target=target,
        max_leverage=cap,
        execution_lag=k,
    )
    run = run_strategy(prices, rf, params)
    bt, report = run.backtest, build_report(run)
    win = slice(bt.e, bt.stop)
    held = bt.held_id.iloc[win].notna()
    by_gross = bt.gross_held.iloc[win][held] >= cap * (1 - 1e-9)
    by_scale = run.scale_used.shift(k + 1).iloc[win][held] >= cap
    assert by_gross.equals(by_scale)
    pct = report.sizing.pct_days_at_cap
    assert pct == by_gross.mean() and 0 < pct < 1
    assert any("not reachable within max_leverage" in w for w in report.warnings) == (pct > 0.25)


def test_pct_days_at_cap_is_null_when_never_invested_or_overlay_off(
    prices: pd.DataFrame, rf: pd.Series
) -> None:
    never = StrategyParams(strategy="mean_reversion", tickers=("Y",), L=20, z_entry=5.0, vol_target=0.10)
    report = build_report(run_strategy(prices, rf, never))
    assert report.sizing is not None and report.sizing.pct_days_at_cap is None
    assert build_report(run_strategy(prices, rf, replace(never, vol_target=None))).sizing is None
