"""Backtest accounting, trades, evaluation window, ruin and leverage (§12, test_backtester.py)."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.services import metrics
from app.services.backtester import BacktestResult, InsufficientDataError, resolve_split, run_backtest
from app.services.monte_carlo import MonteCarloInputError
from app.services.pipeline import (
    StrategyParams,
    StrategyRun,
    WalkForwardResult,
    build_report,
    run_strategy,
    run_walk_forward,
    strategy_mc_returns,
)

OVERLAYS = [None, "daily", "weekly", "monthly"]

# ---------------------------------------------------------------- 1. look-ahead perturbation

BASE_CONFIGS = [
    ("mean-reversion", {"strategy": "mean_reversion", "tickers": ("Y",), "L": 60}),
    ("pairs-fixed", {"strategy": "pairs", "tickers": ("Y", "X"), "L": 60, "hedge_mode": "fixed_at_entry"}),
    ("pairs-rolling", {"strategy": "pairs", "tickers": ("Y", "X"), "L": 60, "hedge_mode": "rolling"}),
    ("trend-monthly", {"strategy": "trend", "tickers": ("X",), "L": 126, "signal_frequency": "monthly"}),
    ("trend-daily", {"strategy": "trend", "tickers": ("X",), "L": 126, "signal_frequency": "daily"}),
]
PERTURBATION_CASES = (
    [
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
    + [
        pytest.param({**dict(BASE_CONFIGS)[base], **extra}, id=name)
        for name, base, extra in [
            ("mean-reversion-long-only-k1", "mean-reversion", {"allow_short": False, "execution_lag": 1}),
            (
                "trend-daily-long-only-weekly-k2",
                "trend-daily",
                {"allow_short": False, "execution_lag": 2, "vol_target": 0.10, "vol_refresh": "weekly"},
            ),
            ("pairs-fixed-borrow-300bps-k1", "pairs-fixed", {"borrow_bps": 300.0, "execution_lag": 1}),
            (
                "mean-reversion-borrow-500bps-daily-k0",
                "mean-reversion",
                {"borrow_bps": 500.0, "execution_lag": 0, "vol_target": 0.10, "vol_refresh": "daily"},
            ),
            # Phase 7: the Kalman model and the trend portfolio.
            ("pairs-kalman-fixed-k1", "pairs-fixed", {"pairs_model": "kalman", "execution_lag": 1}),
            (
                "pairs-kalman-rolling-weekly-k2",
                "pairs-rolling",
                {"pairs_model": "kalman", "kalman_delta": 1e-5, "execution_lag": 2, "vol_target": 0.10},
            ),
        ]
    ]
    + [
        pytest.param(
            {
                "strategy": "trend_portfolio",
                "tickers": ("T08", "T15", "T20", "T30"),
                "L": 126,
                "vol_target": 0.40,
                **extra,
            },
            id=name,
        )
        for name, extra in [
            ("trend-portfolio-monthly-weekly-k1", {"signal_frequency": "monthly", "vol_refresh": "weekly"}),
            (
                "trend-portfolio-daily-daily-k0",
                {"signal_frequency": "daily", "vol_refresh": "daily", "execution_lag": 0},
            ),
            (
                "trend-portfolio-monthly-long-only-borrow-k2",
                {"vol_refresh": "monthly", "execution_lag": 2, "allow_short": False, "borrow_bps": 300.0},
            ),
        ]
    ]
)
CUTS = (300, 480, 655, 870)


def _outputs(run: StrategyRun) -> dict[str, pd.Series | pd.DataFrame]:
    out = {
        "signal": run.signal,
        "weights": run.weights,
        "pos": run.backtest.pos,
        "excess": run.backtest.excess,
        "total": run.backtest.total,
    }
    for name in (
        "score",
        "vol_forecast",
        "beta",
        "beta_used",
        "leg_scores",
        "leg_signals",
        "leg_vol_forecasts",
    ):
        if getattr(run, name) is not None:
            out[name] = getattr(run, name)
    if run.backtest.leg_pos is not None:
        out["leg_pos"] = run.backtest.leg_pos
    return out


def _perturb_after(
    prices: pd.DataFrame, rf: pd.Series, cut: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.Series]:
    """Prices and rf multiplied by random lognormal noise on every date strictly after position cut."""
    future = slice(cut + 1, None)
    noisy = prices.copy()
    noisy.iloc[future] *= np.exp(rng.normal(0.0, 0.05, noisy.iloc[future].shape))
    noisy_rf = rf.copy()
    noisy_rf.iloc[future] *= np.exp(rng.normal(0.0, 0.2, len(rf) - cut - 1))
    return noisy, noisy_rf


@pytest.mark.parametrize("config", PERTURBATION_CASES)
def test_no_lookahead_under_future_perturbation(
    portfolio_prices: pd.DataFrame, varying_rf, config: dict
) -> None:
    prices = portfolio_prices
    params = StrategyParams(**config)
    rf = varying_rf(prices.index)
    base = _outputs(run_strategy(prices, rf, params))
    rng = np.random.default_rng(99)
    for cut in CUTS:
        noisy, noisy_rf = _perturb_after(prices, rf, cut, rng)
        pert = _outputs(run_strategy(noisy, noisy_rf, params))
        assert pert.keys() == base.keys()
        for name, a in base.items():
            same = np.array_equal(
                a.iloc[: cut + 1].to_numpy(), pert[name].iloc[: cut + 1].to_numpy(), equal_nan=True
            )
            assert same, f"{name} changed on or before the cut at {cut}"
        # The perturbation did reach the later dates.
        for name in ("score" if "score" in base else "leg_scores", "total"):
            after_a, after_b = base[name].iloc[cut + 1 :].to_numpy(), pert[name].iloc[cut + 1 :].to_numpy()
            assert not np.array_equal(after_a, after_b, equal_nan=True)


# The walk-forward's stitched series (§9.2), where "excess" is the stitched excess. Small grids:
# the stitching and the switch timing are the same for any grid.
WALK_FORWARD_CASES = [
    pytest.param(
        {"strategy": "pairs", "tickers": ("Y", "X"), "L": 60, "borrow_bps": 200.0},
        {"L_values": (20, 60, 120), "z_entry_values": (1.5, 2.5)},
        id="pairs-borrow-k1",
    ),
    pytest.param(
        {
            "strategy": "trend",
            "tickers": ("X",),
            "L": 63,
            "signal_frequency": "daily",
            "execution_lag": 2,
            "vol_target": 0.10,
            "vol_refresh": "daily",
        },
        {"L_values": (21, 63, 126)},
        id="trend-daily-overlay-k2",
    ),
]


@pytest.mark.parametrize("config, grid", WALK_FORWARD_CASES)
def test_no_lookahead_in_the_walk_forward_stitched_series(
    prices: pd.DataFrame, varying_rf, config: dict, grid: dict
) -> None:
    params = StrategyParams(**config)
    rf = varying_rf(prices.index)
    windows = {**grid, "train_days": 252, "test_days": 63}
    base = run_walk_forward(prices, rf, params, **windows)
    assert base.n_distinct_selected >= 2  # some fold switches configuration
    starts = [prices.index.get_loc(fold.test_start) for fold in base.folds]
    # The close before a switch, a switch close itself (b_j - 1), inside a fold, and in the last fold.
    cuts = (starts[1] - 2, starts[2] - 1, starts[3] + 30, starts[-1] + 10)

    def stitched(res: WalkForwardResult) -> dict[str, np.ndarray]:
        return {
            "weights": res.weights.to_numpy(),
            "excess": res.daily.excess.to_numpy(),
            "total": res.daily.total.to_numpy(),
        }

    rng = np.random.default_rng(99)
    for cut in cuts:
        noisy, noisy_rf = _perturb_after(prices, rf, cut, rng)
        pert = run_walk_forward(noisy, noisy_rf, params, **windows)
        a, b = stitched(base), stitched(pert)
        for name in a:
            assert np.array_equal(a[name][: cut + 1], b[name][: cut + 1]), (
                f"{name} changed on or before {cut}"
            )
        after = slice(cut + 1, base.stop)  # the perturbation did reach the later stitched days
        assert not np.array_equal(a["excess"][after], b["excess"][after])


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
        borrow_bps=0.0,
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
    worst, runs, resized, borrowed = 0.0, 0, 0, 0
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
                borrow_bps=float(rng.choice([0.0, rng.uniform(0.0, 2000.0)])),
                allow_short=strategy == "pairs" or bool(rng.random() < 0.75),
                vol_target=None if vol is None else float(rng.uniform(0.05, 0.25)),
                vol_refresh=vol or "weekly",
                max_leverage=float(rng.uniform(1.0, 3.0)),
            )
            bt = run_strategy(prices, rf, params).backtest
            worst = max(worst, abs(bt.pnl.sum() - bt.excess.iloc[bt.e : bt.stop].sum()))
            in_trade = bt.held_id.eq(bt.trade_id) & (bt.turnover > 0)  # resized inside a trade
            resized += int(in_trade.sum())
            borrowed += int((bt.borrow > 0).any())
            runs += 1
    assert runs >= 100 and resized > 0 and borrowed >= 30
    assert worst < 1e-9


def test_per_leg_trades_reconcile_and_never_overlap(synthetic_prices, universe, varying_rf) -> None:
    """§12 item 12 (Phase 7): the trend portfolio's leg trades over at least 50 random runs."""
    rng = np.random.default_rng(77)
    worst, runs, flips, resized = 0.0, 0, 0, 0
    pool = ["T08", "T15", "T20", "T30", "X", "SPY"]
    for _ in range(54):
        base = synthetic_prices(n=int(rng.integers(700, 1000)), seed=int(rng.integers(10**9)))
        prices = base.join(universe(base.index, seed=int(rng.integers(10**9))))
        rf = varying_rf(prices.index, seed=int(rng.integers(10**9)))
        legs = tuple(rng.choice(pool, size=int(rng.integers(2, 6)), replace=False))
        params = StrategyParams(
            strategy="trend_portfolio",
            tickers=legs,
            L=int(rng.choice([21, 42, 63, 126])),
            signal_frequency=str(rng.choice(["daily", "monthly"])),
            execution_lag=int(rng.integers(0, 3)),
            cost_bps=float(rng.uniform(0.0, 30.0)),
            borrow_bps=float(rng.choice([0.0, rng.uniform(0.0, 2000.0)])),
            allow_short=bool(rng.random() < 0.75),
            vol_target=float(rng.uniform(0.05, 0.6)),
            vol_halflife=int(rng.choice([5, 21, 42])),
            vol_refresh=str(rng.choice(["daily", "weekly", "monthly"])),
            max_leverage=float(rng.uniform(1.0, 4.0)),
        )
        bt = run_strategy(prices, rf, params).backtest
        worst = max(worst, abs(bt.pnl.sum() - bt.excess.iloc[bt.e : bt.stop].sum()))
        trades = bt.trades
        assert set(trades["ticker"]) <= set(legs) and trades["ticker"].notna().all()
        assert trades["id"].tolist() == list(range(1, len(trades) + 1))
        order = trades.sort_values(["entry_date", "ticker"], kind="stable")["id"].tolist()
        assert order == trades["id"].tolist()  # numbered by entry date, then ticker
        for leg, leg_trades in trades.groupby("ticker"):
            leg_trades = leg_trades.sort_values("entry_date")
            exits = leg_trades["exit_date"].iloc[:-1].to_numpy()
            entries = leg_trades["entry_date"].iloc[1:].to_numpy()
            assert not leg_trades["open"].iloc[:-1].any()  # only a leg's last trade can be open
            assert (entries >= exits).all(), leg  # no two trades of a leg are held at once
            flips += int((entries == exits).sum())
            # The leg's held_id days are exactly its trades': the counts and the last held days.
            held = bt.held_id[leg].iloc[: bt.stop].dropna()
            days = held.index.to_series().groupby(held.to_numpy())
            ids = leg_trades["id"].to_numpy()
            assert np.array_equal(
                days.size().reindex(ids, fill_value=0).to_numpy(), leg_trades["holding_days"]
            )
            first = days.first().reindex(ids)
            assert (
                first.dropna().to_numpy() > leg_trades["entry_date"].to_numpy()[first.notna().to_numpy()]
            ).all()
            closed = ~leg_trades["open"].to_numpy()
            assert np.array_equal(
                days.last().reindex(ids).to_numpy()[closed], leg_trades["exit_date"].to_numpy()[closed]
            )
        resized += int((bt.turnover[bt.held] > 0).sum())
        runs += 1
    assert runs >= 50 and flips > 0 and resized > 0
    assert worst < 1e-9


@pytest.mark.parametrize(
    "config",
    [
        {"strategy": "pairs", "tickers": ("Y", "X"), "L": 60, "vol_target": 0.10, "execution_lag": 2},
        {"strategy": "mean_reversion", "tickers": ("Y",), "L": 40, "allow_short": False, "borrow_bps": 200.0},
        {"strategy": "trend_portfolio", "tickers": ("T08", "T15", "T20"), "L": 126, "vol_target": 0.40},
    ],
    ids=["pairs", "mean-reversion", "trend-portfolio"],
)
def test_skipping_the_trade_bookkeeping_changes_nothing_else(
    portfolio_prices: pd.DataFrame, varying_rf, config: dict
) -> None:
    """The parameter grids run without the trade bookkeeping (trades=False): the trade fields are
    None and every other field of the backtest is exactly as in a full run."""
    rf = varying_rf(portfolio_prices.index)
    params = StrategyParams(**config)
    full = run_strategy(portfolio_prices, rf, params).backtest
    lean = run_strategy(portfolio_prices, rf, params, trades=False).backtest
    trade_fields = ("trade_id", "held_id", "pnl", "trades")
    assert all(getattr(lean, name) is None for name in trade_fields)
    assert len(full.trades) > 0
    for field in fields(BacktestResult):
        if field.name in trade_fields:
            continue
        a, b = getattr(full, field.name), getattr(lean, field.name)
        if isinstance(a, pd.Series):
            pd.testing.assert_series_equal(a, b, check_exact=True)
        elif isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        else:
            assert a == b, field.name


@pytest.mark.parametrize("k, borrow, allow_short", [(0, 0.0, True), (1, 400.0, True), (2, 0.0, False)])
def test_each_legs_trades_are_those_of_that_leg_backtested_alone(
    portfolio_prices: pd.DataFrame, varying_rf, k: int, borrow: float, allow_short: bool
) -> None:
    """§7.4: identification, attribution and bookkeeping run leg by leg, so each leg's trades are
    exactly what §7 gives for that leg alone, holding its weights (with the portfolio's ids)."""
    rf = varying_rf(portfolio_prices.index)
    params = StrategyParams(
        strategy="trend_portfolio",
        tickers=("T08", "T15", "T20", "T30"),
        L=21,
        signal_frequency="daily",
        vol_target=0.30,
        vol_refresh="weekly",
        execution_lag=k,
        cost_bps=12.0,
        borrow_bps=borrow,
        allow_short=allow_short,
    )
    run = run_strategy(portfolio_prices, rf, params)
    bt = run.backtest
    assert bt.held.equals((bt.w.shift(1).fillna(0.0).abs().sum(axis=1) > 0).rename("held"))  # any leg held
    if not allow_short:  # long-only legs go flat on their own, so "any leg held" differs from "every leg"
        assert (bt.held & bt.held_id.isna().any(axis=1)).any()
    columns = ["side", "segment", "open", "signal_date", "entry_date", "exit_signal_date", "exit_date"]
    flips = 0
    for leg in params.tickers:
        alone = run_backtest(
            R=run.returns[[leg]],
            W=run.weights[[leg]],
            s=run.leg_signals[leg],
            rf=rf,
            t0=run.t0,
            k=k,
            cost_bps=12.0,
            borrow_bps=borrow,
            E0=1.0,
            split_date=bt.split_date,
        ).trades
        mine = bt.trades.loc[bt.trades["ticker"] == leg].reset_index(drop=True)
        assert len(mine) == len(alone) > 5
        assert mine[[*columns, "holding_days"]].equals(alone[[*columns, "holding_days"]]), leg
        np.testing.assert_allclose(mine["pnl"].to_numpy(), alone["pnl"].to_numpy(), rtol=0, atol=1e-15)
        flips += int(
            (alone["entry_date"].iloc[1:].to_numpy() == alone["exit_date"].iloc[:-1].to_numpy()).sum()
        )
    assert (
        flips > 0 or not allow_short
    )  # a flip's cost is split between its trades (§7.1); none when long-only


def test_a_ruin_day_credit_is_split_by_gross_weight() -> None:
    """Two trend-portfolio legs held into a crash that ruins the run: the clipped amount is
    credited to each leg's open trade in proportion to its gross weight (approved, §6.2)."""
    n, crash = 800, 600
    g = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=n).as_unit("ns")
    calm = g.normal(0.001, 0.005, n)  # sigma about 8% a year: 0.6 / sigma is capped at 4
    rough = g.normal(0.002, 0.02, n)  # about 32% a year: below the cap
    calm[crash] = rough[crash] = np.log(0.5)
    prices = pd.DataFrame(
        {"A": 100 * np.exp(np.cumsum(calm)), "B": 100 * np.exp(np.cumsum(rough))}, index=idx
    )
    prices["SPY"] = prices["A"]
    rf = pd.Series(0.02 / 252, index=idx)
    params = StrategyParams(
        strategy="trend_portfolio",
        tickers=("A", "B"),
        L=252,
        signal_frequency="daily",
        vol_target=0.6,
        max_leverage=4.0,
        vol_refresh="daily",
        cost_bps=5.0,
        borrow_bps=0.0,
    )
    run = run_strategy(prices, rf, params)
    bt = run.backtest
    assert bt.ruined and bt.ruin_date == idx[crash] and bt.total.iloc[crash] == -1.0
    held_into = bt.w.iloc[crash - 1]
    assert held_into["A"] == 2.0 and 0.5 < held_into["B"] < 1.5  # A at its cap of 4 / 2, B below
    # Each leg's unclipped daily contribution; leg i's trades together earn these plus its credit.
    w_prev, R = bt.w.shift(1).fillna(0.0), run.returns[["A", "B"]].fillna(0.0)
    contrib = w_prev.mul(R.sub(rf, axis=0)) - 5e-4 * (bt.w - w_prev).abs()
    window = slice(bt.e, bt.stop)
    unclipped = contrib.iloc[crash].sum()
    credit = bt.excess.iloc[crash] - unclipped
    assert credit > 0.4
    for leg in ("A", "B"):
        leg_pnl = bt.trades.loc[bt.trades["ticker"] == leg, "pnl"].sum()
        credit_leg = leg_pnl - contrib[leg].iloc[window].sum()
        assert credit_leg == pytest.approx(credit * held_into[leg] / held_into.sum(), rel=1e-12)
    assert abs(bt.pnl.sum() - bt.excess.iloc[window].sum()) < 1e-12
    assert bt.trades.groupby("ticker")["open"].last().all()  # both positions are open at r


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
        borrow=bt.borrow.iloc[sl],
        trades=bt.trades,
    )
    assert (m.n_closed_trades, m.n_open_trades, m.win_rate, m.avg_holding_days) == (1, 1, 1.0, 4.0)


def test_leverage_is_financed_at_rf() -> None:
    idx = pd.bdate_range("2020-01-01", periods=10).as_unit("ns")
    R = pd.DataFrame({"A": [0.0, 0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, 0.0, 0.01]}, index=idx)
    rf = pd.Series(0.0002, index=idx)
    s = pd.Series([0.0] + [1.0] * 9, index=idx)
    bt = run_backtest(
        R=R,
        W=(2 * s).to_frame("A"),
        s=s,
        rf=rf,
        t0=1,
        k=0,
        cost_bps=0.0,
        borrow_bps=0.0,
        E0=1.0,
        split_date=idx[6],
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
        "borrow_bps": 0.0,
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
        borrow_bps=0.0,
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


# ---------------------------------------------------------------- 11. borrow fee (Phase 5)


def test_a_constant_short_pays_exactly_the_borrow_fee() -> None:
    idx = pd.bdate_range("2020-01-01", periods=10).as_unit("ns")
    R = pd.DataFrame({"A": [0.0, 0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, 0.0, 0.01]}, index=idx)
    rf = pd.Series(0.0002, index=idx)
    s = pd.Series([0.0] + [-1.0] * 9, index=idx)
    b = 250.0
    bt = run_backtest(
        R=R, W=s.to_frame("A"), s=s, rf=rf, t0=1, k=0, cost_bps=0.0, borrow_bps=b, E0=1.0, split_date=idx[6]
    )
    held = bt.held_id.notna()
    assert held.sum() == 8
    expected = -(R["A"] - rf) - b / 10_000 / 252  # short proceeds earn rf; the fee is charged on top
    assert np.array_equal(bt.excess[held].to_numpy(), expected[held].to_numpy())
    assert (bt.borrow[~held] == 0.0).all()
    assert abs(bt.pnl.sum() - bt.excess.iloc[bt.e : bt.stop].sum()) < 1e-15


def test_long_only_runs_and_flat_days_pay_no_borrow_fee(prices: pd.DataFrame, rf: pd.Series) -> None:
    long_only = StrategyParams(
        strategy="mean_reversion", tickers=("Y",), L=40, allow_short=False, borrow_bps=500.0
    )
    run = run_strategy(prices, rf, long_only)
    assert (run.backtest.borrow == 0.0).all() and run.backtest.held_id.notna().any()
    assert all(m.ann_borrow_drag == 0.0 for m in build_report(run).metrics.values())
    pairs = run_strategy(
        prices, rf, StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60, borrow_bps=500.0)
    )
    bt = pairs.backtest
    flat = bt.gross_held == 0
    assert flat.any() and (bt.borrow[flat] == 0.0).all() and (bt.borrow[~flat] > 0).any()
    shorts = (-bt.w.shift(1).fillna(0.0)).clip(lower=0).sum(axis=1)  # the short leg held into each day
    np.testing.assert_allclose(bt.borrow.to_numpy(), 500.0 / 10_000 / 252 * shorts.to_numpy(), rtol=1e-15)


def test_a_zero_borrow_fee_changes_nothing(prices: pd.DataFrame, varying_rf) -> None:
    rf = varying_rf(prices.index)
    for params in (
        StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60, hedge_mode="rolling", vol_target=0.1),
        StrategyParams(strategy="mean_reversion", tickers=("Y",), L=40, execution_lag=2),
    ):
        bt = run_strategy(prices, rf, params).backtest
        assert params.borrow_bps == 0.0 and (bt.borrow == 0.0).all()
        # Bit for bit the formula before Phase 5: excess = gross_excess - cost.
        assert np.array_equal(bt.excess.to_numpy(), (bt.gross_excess - bt.cost).to_numpy())


# ---------------------------------------------------------------- 13. cost sweep and break-even (Phase 5)


def segment_sharpe(bt: BacktestResult, bounds: tuple[int, int], excess: pd.Series) -> float | None:
    a, b = bounds
    return metrics.sharpe(excess.iloc[a:b], exposed=bool(bt.held_id.iloc[a:b].notna().any()))


SWEEP_PARAMS = StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60, cost_bps=7.5, borrow_bps=100.0)


def test_the_sweep_at_the_requests_own_cost_is_the_runs_sharpe(prices: pd.DataFrame, varying_rf) -> None:
    report = build_report(run_strategy(prices, varying_rf(prices.index), SWEEP_PARAMS))
    sweep = report.cost_sensitivity
    assert sweep.cost_bps == (0.0, 2.0, 5.0, 7.5, 10.0, 20.0, 50.0, 100.0)  # the grid plus 7.5, sorted
    own = sweep.cost_bps.index(7.5)
    for name in ("is", "oos", "full"):
        assert len(sweep.sharpe[name]) == 8
        assert sweep.sharpe[name][own] == report.metrics[name].sharpe  # exactly: the same excess series


def test_rerunning_at_each_cost_reproduces_the_sweep(prices: pd.DataFrame, varying_rf) -> None:
    rf = varying_rf(prices.index)
    sweep = build_report(run_strategy(prices, rf, SWEEP_PARAMS)).cost_sensitivity
    for i, bps in enumerate(sweep.cost_bps):
        bt = run_strategy(prices, rf, replace(SWEEP_PARAMS, cost_bps=bps)).backtest
        for name, bounds in bt.segment_bounds().items():
            want = segment_sharpe(bt, bounds, bt.excess)
            assert sweep.sharpe[name][i] == pytest.approx(want, rel=1e-12, abs=1e-12), (bps, name)


def test_at_the_break_even_cost_the_mean_excess_is_zero(prices: pd.DataFrame, rf: pd.Series) -> None:
    run = run_strategy(prices, rf, SWEEP_PARAMS)
    bt, sweep = run.backtest, build_report(run).cost_sensitivity
    for name, (a, b) in bt.segment_bounds().items():
        be = sweep.break_even_bps[name]
        assert be is not None and be > 0
        excess = bt.gross_excess - bt.borrow - (be / 10_000) * bt.turnover
        assert abs(excess.iloc[a:b].mean()) < 1e-15
        sr = segment_sharpe(bt, (a, b), excess)
        assert sr is None or abs(sr) < 1e-9  # 0 up to rounding


def test_break_even_is_null_without_trades_and_the_sweep_is_null_after_ruin(
    prices: pd.DataFrame, rf: pd.Series
) -> None:
    never = StrategyParams(strategy="mean_reversion", tickers=("Y",), L=20, z_entry=5.0)
    sweep = build_report(run_strategy(prices, rf, never)).cost_sensitivity
    assert sweep.break_even_bps == {"is": None, "oos": None, "full": None}
    assert all(v is None for values in sweep.sharpe.values() for v in values)  # no exposure
    crash = crash_market(500)
    ruined = build_report(run_strategy(crash, pd.Series(0.02 / 252, index=crash.index), LEVERED_TREND))
    assert ruined.ruined and ruined.cost_sensitivity is None


def test_break_even_is_null_when_losing_before_costs() -> None:
    n = 10
    sweep = metrics.cost_sweep(
        gross_excess=np.full(n, -0.001),
        borrow=np.zeros(n),
        turnover=np.full(n, 0.5),
        rf=np.zeros(n),
        held=np.ones(n, dtype=bool),
        segments={"is": (1, 6), "oos": (6, n), "full": (1, n)},
        e=1,
        cost_bps=10.0,
    )
    assert sweep.break_even_bps == {"is": None, "oos": None, "full": None}


@pytest.mark.parametrize("ruin_day, null_segments", [(7, {"oos", "full"}), (3, {"is", "oos", "full"})])
def test_a_cost_that_ruins_the_run_nulls_the_segments_from_its_ruin_day(
    ruin_day: int, null_segments: set[str]
) -> None:
    n = 10
    gross = 0.001 + np.random.default_rng(0).normal(0.0, 0.0005, n)  # every segment's std is > 0
    gross[ruin_day] = -0.995  # survives up to 20 bps (total -0.999), ruined from 50 bps (total -1.005)
    turnover = np.full(n, 0.2)
    turnover[ruin_day] = 2.0
    sweep = metrics.cost_sweep(
        gross_excess=gross,
        borrow=np.zeros(n),
        turnover=turnover,
        rf=np.zeros(n),
        held=np.ones(n, dtype=bool),
        segments={"is": (1, 5), "oos": (5, n), "full": (1, n)},
        e=1,
        cost_bps=10.0,
    )
    for i, bps in enumerate(sweep.cost_bps):
        ruined = -0.995 - bps / 10_000 * 2.0 <= -1.0
        assert ruined == (bps >= 50.0)
        for name, values in sweep.sharpe.items():
            assert (values[i] is None) == (ruined and name in null_segments), (bps, name)
    for name, (a, b) in {
        "is": (1, 5),
        "oos": (5, n),
        "full": (1, n),
    }.items():  # one per segment, not per cost
        assert (sweep.break_even_bps[name] is None) == (a <= ruin_day < b)  # -99.5% before costs: losing


# ---------------------------------------------------------------- 14. rolling and calendar stats (Phase 5)


@pytest.mark.parametrize("strategy, tickers", [("pairs", ("Y", "X")), ("trend", ("SPY",))])
def test_rolling_sharpe_and_beta_are_the_metrics_of_each_window(
    prices: pd.DataFrame, varying_rf, strategy: str, tickers: tuple[str, ...]
) -> None:
    rf = varying_rf(prices.index)
    params = StrategyParams(strategy=strategy, tickers=tickers, L=60, signal_frequency="daily")
    run = run_strategy(prices, rf, params)
    bt, series = run.backtest, build_report(run).series
    market_excess = run.returns["SPY"] - rf
    pos = prices.index.get_indexer(series.index)
    before = pos < bt.e + 251  # null before the 252nd evaluated day
    assert before.any() and series.loc[before, ["rolling_sharpe", "rolling_beta"]].isna().all().all()
    checked = 0
    for i in np.flatnonzero(~before)[::7]:  # every 7th window
        sl = slice(pos[i] - 251, pos[i] + 1)
        exposed = bool(bt.held_id.iloc[sl].notna().any())
        want_sr = metrics.sharpe(bt.excess.iloc[sl], exposed=exposed)
        want_beta = metrics.beta_corr(bt.excess.iloc[sl], market_excess.iloc[sl], exposed=exposed)[0]
        for got, want in (
            (series["rolling_sharpe"].iloc[i], want_sr),
            (series["rolling_beta"].iloc[i], want_beta),
        ):
            assert np.isnan(got) if want is None else got == pytest.approx(want, rel=1e-12)
        checked += want_sr is not None
    assert checked > 50


def test_rolling_values_are_null_on_windows_without_a_position() -> None:
    n = 700
    g = np.random.default_rng(1)
    x, held = np.zeros(n), np.zeros(n, dtype=bool)
    x[100:150], held[100:150] = g.normal(0.0, 0.01, 50), True  # a first trade
    x[500] = -0.001  # an entry day: it pays the cost, but no position is held into it
    x[501:], held[501:] = g.normal(0.0, 0.01, n - 501), True  # the second trade
    sr, beta = metrics.rolling_sharpe_beta(x, g.normal(0.0, 0.01, n), held)
    for values in (sr, beta):
        assert np.isnan(values[:251]).all()  # before the 252nd day
        assert np.isfinite(values[251:401]).all()  # windows that include days 100-149
        assert np.isnan(values[401:500]).all()  # no held day, and every excess return is 0
        assert np.isnan(values[500])  # no held day, though the entry cost makes the std > 0
        assert np.isfinite(values[501:]).all()


def test_calendar_returns_compound_to_the_annual_returns(prices: pd.DataFrame, rf: pd.Series) -> None:
    run = run_strategy(prices, rf, StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60))
    last_day = prices.index[-1].date()  # 2021-10-29, a Friday: the last trading day of October
    report = build_report(run, end=date(2021, 10, 31))
    monthly, annual = report.calendar_monthly, report.calendar_annual
    assert [a.year for a in annual] == sorted({m.year for m in monthly})
    for year in annual:
        months = [m for m in monthly if m.year == year.year]
        for field in ("strategy", "benchmark"):
            compounded = np.prod([1.0 + getattr(m, field) for m in months]) - 1.0
            assert compounded == pytest.approx(getattr(year, field), rel=1e-12, abs=1e-15)
    assert prices.index[run.backtest.e].day > 1  # the window starts mid-month...
    assert monthly[0].partial and annual[0].partial  # ...so its first month and year are partial
    assert not any(m.partial for m in monthly[1:])  # October 2021 is complete: `end` is its last day
    assert annual[-1].partial  # 2021 continues after `end`
    early = build_report(run, end=last_day)  # Oct 29 is before Oct 31, the month's last calendar day
    assert early.calendar_monthly[-1].partial


def test_the_last_period_is_partial_after_ruin_or_before_its_last_calendar_day() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-03-31").as_unit("ns")
    total = np.full(len(dates), 0.001)
    kwargs = {"dates": dates, "total": total, "benchmark": total, "calendar": dates}
    monthly, annual = metrics.calendar_returns(**kwargs, end=date(2020, 3, 31), ruined=False)
    assert [m.partial for m in monthly] == [False, False, False] and annual[0].partial  # 2020 goes on
    assert monthly[0].strategy == pytest.approx(1.001**23 - 1, rel=1e-12)  # 23 trading days in Jan 2020
    monthly, _ = metrics.calendar_returns(**kwargs, end=date(2020, 3, 31), ruined=True)
    assert [m.partial for m in monthly] == [False, False, True]  # ruin ends the window's last period
    wider = pd.bdate_range("2019-12-02", "2020-04-30").as_unit("ns")  # days outside the window's months
    monthly, _ = metrics.calendar_returns(
        **{**kwargs, "calendar": wider}, end=date(2020, 4, 30), ruined=False
    )
    assert [m.partial for m in monthly] == [False, False, False]
    cut = dates[dates <= "2020-03-20"]
    monthly, _ = metrics.calendar_returns(
        dates=cut,
        total=total[: len(cut)],
        benchmark=total[: len(cut)],
        calendar=dates,
        end=date(2020, 3, 31),
        ruined=False,
    )
    assert monthly[-1].partial  # March has run-calendar days after the window
