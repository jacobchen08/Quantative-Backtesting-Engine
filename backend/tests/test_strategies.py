"""Strategies, the entry/exit rule and the vol-targeting overlay (§12, test_strategies.py)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.services import sizing
from app.services.backtester import NoValidScoreError, run_backtest
from app.services.pipeline import StrategyParams, run_strategy
from app.services.strategies import (
    first_of_period,
    hysteresis_signal,
    mean_reversion_score,
    pairs_score,
    trend_score,
    trend_signal,
)

# ---------------------------------------------------------------- 1-2. entry/exit rule


def reference_signal(z: np.ndarray, z_entry: float, z_exit: float) -> np.ndarray:
    """Ground truth for the §5.1 crossing rule: a state machine run one bar at a time."""
    state, out = 0.0, np.empty(len(z))
    for i, v in enumerate(z):
        if np.isnan(v):
            state = 0.0
        elif v > z_entry:
            state = -1.0
        elif v < -z_entry:
            state = 1.0
        elif state == -1.0 and v < z_exit:
            state = 0.0
        elif state == 1.0 and v > -z_exit:
            state = 0.0
        out[i] = state
    return out


def test_entry_exit_rule_matches_reference_state_machine() -> None:
    rng = np.random.default_rng(0)
    flips = 0
    for case in range(1200):
        z_entry = float(rng.uniform(0.1, 5.0)) if case % 2 else float(rng.choice([0.5, 1.0, 2.0, 2.5, 5.0]))
        z_exit = float(rng.choice([0.0, rng.uniform(0.0, z_entry), 0.5 * z_entry]))
        n = int(rng.integers(1, 160))
        z = rng.normal(0.0, 1.3 * z_entry, n)
        special = rng.random(n) < 0.3  # NaNs and values exactly at the thresholds
        z[special] = rng.choice([z_entry, -z_entry, z_exit, -z_exit, 0.0, np.nan], int(special.sum()))
        got = hysteresis_signal(pd.Series(z), z_entry, z_exit).to_numpy()
        want = reference_signal(z, z_entry, z_exit)
        assert np.array_equal(got, want), f"case {case}: z_entry={z_entry}, z_exit={z_exit}, z={z.tolist()}"
        flips += int(np.sum(np.abs(np.diff(want)) == 2))
    assert flips > 0  # direct long/short flips were exercised


def test_jump_through_the_band() -> None:
    z = pd.Series([0, 2.5, 1.5, 0.7, -0.7, -1.2, -1.6, -1.0, -0.3], dtype=float)
    assert hysteresis_signal(z, 2.0, 0.5).tolist() == [0, -1, -1, -1, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("z_entry, z_exit", [(2.0, 2.0), (2.0, -0.1), (1.0, 1.5)])
def test_entry_exit_rule_rejects_invalid_thresholds(z_entry: float, z_exit: float) -> None:
    with pytest.raises(ValueError):
        hysteresis_signal(pd.Series([0.0, 1.0]), z_entry, z_exit)


# ---------------------------------------------------------------- 3-4. pairs z-score and degenerate inputs


@pytest.mark.parametrize("L", [20, 60])
def test_pairs_zscore_and_beta_match_polyfit(prices: pd.DataFrame, L: int) -> None:
    out = pairs_score(prices["Y"], prices["X"], L)
    y = np.log(prices["Y"] / prices["Y"].iloc[0]).to_numpy()
    x = np.log(prices["X"] / prices["X"].iloc[0]).to_numpy()
    rng = np.random.default_rng(L)
    for t in rng.choice(np.arange(L - 1, len(y)), size=25, replace=False):
        xw, yw = x[t - L + 1 : t + 1], y[t - L + 1 : t + 1]
        slope, intercept = np.polyfit(xw, yw, 1)
        resid = yw - (intercept + slope * xw)
        assert abs(out.beta.iloc[t] - slope) < 1e-8
        assert abs(out.z.iloc[t] - resid[-1] / np.std(resid, ddof=1)) < 1e-8
    assert out.z.iloc[: L - 1].isna().all() and out.z.iloc[L - 1 :].notna().all()


def test_exactly_collinear_pair_has_no_score(prices: pd.DataFrame, rf: pd.Series) -> None:
    Y = prices["X"] ** 2  # ln Y = 2 ln X exactly
    out = pairs_score(Y, prices["X"], 60)
    assert out.z.isna().all()
    assert (hysteresis_signal(out.z, 2.0, 0.5) == 0).all()
    with pytest.raises(NoValidScoreError):
        run_strategy(prices.assign(Y=Y), rf, StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60))


def test_flat_x_window_gives_nan_and_never_inf(prices: pd.DataFrame, rf: pd.Series) -> None:
    X = prices["X"].copy()
    X.iloc[300:420] = X.iloc[300]  # 120 days with no price change
    L = 60
    out = pairs_score(prices["Y"], X, L)
    inside = slice(300 + L - 1, 420)  # windows lying entirely inside the flat stretch
    assert out.beta.iloc[inside].isna().all() and out.z.iloc[inside].isna().all()
    assert out.z.iloc[: 300 + L - 1].notna().sum() > 0 and out.z.iloc[420:].notna().all()
    for series in (out.z, out.beta):
        assert not np.isinf(series.to_numpy()).any()
    run = run_strategy(prices.assign(X=X), rf, StrategyParams(strategy="pairs", tickers=("Y", "X"), L=L))
    assert np.isfinite(run.weights.to_numpy()).all()
    assert (run.signal.iloc[inside] == 0).all()


def test_mean_reversion_on_a_flat_price_has_no_score(prices: pd.DataFrame, rf: pd.Series) -> None:
    flat = pd.Series(37.5, index=prices.index)
    z = mean_reversion_score(flat, 60)
    assert z.isna().all()
    assert (hysteresis_signal(z, 2.0, 0.5) == 0).all()
    with pytest.raises(NoValidScoreError):
        run_strategy(
            prices.assign(Y=flat), rf, StrategyParams(strategy="mean_reversion", tickers=("Y",), L=60)
        )


# ---------------------------------------------------------------- 5. hedge ratio and flat weights


def test_fixed_hedge_ratio_changes_only_on_opening_days(prices: pd.DataFrame, rf: pd.Series) -> None:
    run = run_strategy(prices, rf, StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60))
    s, bu, beta = run.signal, run.beta_used, run.beta
    opens = s.ne(s.shift()) & s.ne(0)  # entries from flat and direct flips
    changed = bu.ne(bu.shift()) & bu.notna()
    assert opens.sum() >= 10
    assert not (changed & ~opens).any()
    assert np.array_equal(bu[opens].to_numpy(), beta[opens].to_numpy())
    held = s != 0
    assert (bu[held] != beta[held]).any()  # the rolling beta drifts while the frozen one holds


def test_rolling_hedge_ratio_is_the_rolling_beta(prices: pd.DataFrame, rf: pd.Series) -> None:
    params = StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60, hedge_mode="rolling")
    run = run_strategy(prices, rf, params)
    assert np.array_equal(run.beta_used.to_numpy(), run.beta.to_numpy(), equal_nan=True)


@pytest.mark.parametrize("hedge_mode", ["fixed_at_entry", "rolling"])
def test_pair_weights_are_exactly_zero_when_flat(
    prices: pd.DataFrame, rf: pd.Series, hedge_mode: str
) -> None:
    L = 60
    run = run_strategy(
        prices, rf, StrategyParams(strategy="pairs", tickers=("Y", "X"), L=L, hedge_mode=hedge_mode)
    )
    W, s = run.weights, run.signal
    assert list(W.columns) == ["Y", "X"]
    assert (W[s == 0] == 0.0).all().all()
    assert (W.iloc[: L - 1] == 0.0).all().all()  # warm-up, where beta is NaN
    gross = W.abs().sum(axis=1)[s != 0]
    assert np.allclose(gross, 1.0, rtol=0, atol=1e-15)  # unit weights have gross exposure 1
    long_days = s == 1
    assert (W.loc[long_days, "Y"] > 0).all()  # a long spread is long Y


# ---------------------------------------------------------------- 6. trend score


def test_trend_score_matches_hand_computation(prices: pd.DataFrame, varying_rf) -> None:
    close, L = prices["SPY"], 63
    rf = varying_rf(prices.index)
    m = trend_score(close, rf, L)
    for t in (L, L + 1, 200, 517, len(close) - 1):
        hand = np.log(close.iloc[t] / close.iloc[t - L]) - rf.iloc[t - L + 1 : t + 1].sum()
        assert m.iloc[t] == pytest.approx(hand, rel=0, abs=1e-14)
    assert m.iloc[:L].isna().all() and m.iloc[L:].notna().all()
    for freq in ("daily", "monthly"):
        assert (trend_signal(m, freq)[1].iloc[:L] == 0).all()


def test_zero_trailing_excess_return_is_flat() -> None:
    L = 21
    idx = pd.bdate_range("2020-01-01", periods=10 * L).as_unit("ns")
    base = np.exp(np.random.default_rng(4).normal(4.0, 0.1, L))
    close = pd.Series(np.tile(base, 10), index=idx)  # C_t == C_{t-L} exactly
    m = trend_score(close, pd.Series(0.0, index=idx), L)
    assert (m.iloc[L:] == 0.0).all()
    for freq in ("daily", "monthly"):
        assert (trend_signal(m, freq)[1] == 0).all()


def test_trend_runs_with_constant_and_varying_rf(prices: pd.DataFrame, rf: pd.Series, varying_rf) -> None:
    params = StrategyParams(strategy="trend", tickers=("X",), L=126, signal_frequency="daily")
    runs = [run_strategy(prices, r, params) for r in (rf, varying_rf(prices.index))]
    for run in runs:
        assert set(run.signal.unique()) <= {-1.0, 0.0, 1.0}
    assert not np.array_equal(runs[0].score.to_numpy(), runs[1].score.to_numpy(), equal_nan=True)


def test_rf_as_a_numpy_array_raises_type_error(prices: pd.DataFrame, rf: pd.Series) -> None:
    with pytest.raises(TypeError):
        trend_score(prices["SPY"], rf.to_numpy(), 63)
    with pytest.raises(TypeError):
        run_strategy(prices, rf.to_numpy(), StrategyParams(strategy="trend", tickers=("SPY",), L=63))


# ---------------------------------------------------------------- 7. monthly sampling


def test_monthly_signal_changes_only_on_first_trading_days_of_months(
    prices: pd.DataFrame, rf: pd.Series
) -> None:
    L = 63
    m = trend_score(prices["X"], rf, L)
    score, signal = trend_signal(m, "monthly")
    starts = first_of_period(prices.index, "M")
    changes = signal.ne(signal.shift()) & (np.arange(len(signal)) > 0)
    assert changes.sum() >= 3
    assert not (changes & ~starts).any()
    first_position = prices.index[signal.ne(0).to_numpy().argmax()]
    first_start_after_lookback = prices.index[(starts.to_numpy() & (np.arange(len(starts)) >= L)).argmax()]
    assert first_position == first_start_after_lookback
    assert score.first_valid_index() == first_start_after_lookback  # t0 is a month start


def test_first_of_period_marks_one_day_per_month_and_week() -> None:
    idx = pd.bdate_range("2019-01-01", "2021-12-31").as_unit("ns")
    idx = idx.drop(pd.DatetimeIndex(["2019-07-01", "2020-06-01", "2021-11-01"]))  # missing month starts
    for freq, key in (("M", idx.to_period("M")), ("W", idx.to_period("W"))):
        marks = first_of_period(idx, freq)
        per_period = marks.groupby(key).sum()
        assert (per_period == 1).all()
        assert marks.groupby(key).idxmax().tolist() == pd.Series(idx, index=idx).groupby(key).min().tolist()
    assert first_of_period(idx, "M")[pd.Timestamp("2020-06-02")]  # the next trading day takes over


# ---------------------------------------------------------------- 8. vol targeting


def test_vol_forecast_matches_explicit_weighted_average() -> None:
    u = pd.Series(np.random.default_rng(3).normal(0.0, 0.01, 400))
    H = 42
    sigma = sizing.vol_forecast(u, H)
    assert sigma.iloc[: H - 1].isna().all() and sigma.iloc[H - 1 :].notna().all()
    a = 1 - 2 ** (-1 / H)
    u2 = u.to_numpy() ** 2
    for t in range(H - 1, len(u)):
        w = (1 - a) ** np.arange(t + 1)  # weight on u_{t-i}
        ref = np.sqrt(252 * np.sum(w * u2[t::-1]) / w.sum())
        assert abs(sigma.iloc[t] / ref - 1) < 1e-12


def test_vol_targeting_tracks_the_target_in_every_regime() -> None:
    target, H, cap, n, block = 0.15, 42, 4.0, 3000, 500
    idx = pd.bdate_range("2010-01-01", periods=n).as_unit("ns")
    true_vol = np.where((np.arange(n) // block) % 2 == 0, 0.10, 0.40) / np.sqrt(252)
    zero = pd.Series(0.0, index=idx)
    for seed in range(200):
        r = pd.Series(np.random.default_rng(seed).normal(0.0, true_vol), index=idx)
        sigma, score = sizing.forecast_and_mask(r, pd.Series(1.0, index=idx), H)
        signal = score.fillna(0.0)  # constant long once the forecast exists
        sized = sizing.size_weights(
            signal.to_frame("A"), signal, sigma, vol_target=target, max_leverage=cap, vol_refresh="weekly"
        )
        t0 = int(np.argmax(score.notna().to_numpy()))
        bt = run_backtest(
            R=r.to_frame("A"),
            W=sized.weights,
            s=signal,
            rf=zero,
            t0=t0,
            k=1,
            cost_bps=0.0,
            borrow_bps=0.0,
            E0=1.0,
            split_date=idx[n // 2],
        )
        realized, unscaled = [], []
        for b in range(n // block):
            seg = slice(b * block + 3 * H, (b + 1) * block)  # skip 3 half-lives after each switch
            realized.append(bt.total.iloc[seg].std() * np.sqrt(252) / target)
            unscaled.append(r.iloc[seg].std() * np.sqrt(252))
        assert all(0.8 <= x <= 1.2 for x in realized), (seed, realized)
        assert min(unscaled[1::2]) / max(unscaled[0::2]) > 3  # the regimes really differ about 4x


def test_all_zero_returns_give_the_leverage_cap() -> None:
    idx = pd.bdate_range("2020-01-01", periods=300).as_unit("ns")
    close = pd.Series(50.0, index=idx)
    u = close / close.shift(1) - 1  # all zero after the first row
    sigma = sizing.vol_forecast(u, 21)
    signal = pd.Series(1.0, index=idx).where(sigma.notna(), 0.0)
    sized = sizing.size_weights(
        signal.to_frame("A"), signal, sigma, vol_target=0.10, max_leverage=2.5, vol_refresh="daily"
    )
    valid = sigma.notna()
    assert valid.sum() > 200 and (sigma[valid] == 0).all()
    assert (sized.scale[valid] == 2.5).all()
    assert not np.isinf(sized.scale.to_numpy()).any()
    assert (sized.weights["A"][valid] == 2.5).all() and (sized.weights["A"][~valid] == 0.0).all()


def test_weekly_refresh_resizes_only_on_week_starts_or_signal_changes(
    prices: pd.DataFrame, rf: pd.Series
) -> None:
    params = StrategyParams(
        strategy="trend",
        tickers=("X",),
        L=63,
        signal_frequency="daily",
        vol_target=0.10,
        vol_refresh="weekly",
    )
    run = run_strategy(prices, rf, params)
    su, s = run.scale_used, run.signal
    changed = su.ne(su.shift()) & su.notna() & su.shift().notna()
    allowed = first_of_period(su.index, "W") | s.ne(s.shift())
    assert changed.sum() > 50 and (s.ne(s.shift()) & s.ne(0)).sum() >= 3
    assert not (changed & ~allowed).any()


@pytest.mark.parametrize(
    "strategy, tickers, L, freq",
    [
        ("mean_reversion", ("Y",), 20, "monthly"),
        ("pairs", ("Y", "X"), 20, "monthly"),
        ("trend", ("SPY",), 21, "daily"),
        ("trend", ("SPY",), 21, "monthly"),
    ],
)
def test_long_vol_halflife_delays_t0_to_the_first_forecast(
    prices: pd.DataFrame, rf: pd.Series, strategy: str, tickers: tuple[str, ...], L: int, freq: str
) -> None:
    H = 126
    base = StrategyParams(strategy=strategy, tickers=tickers, L=L, signal_frequency=freq)
    off = run_strategy(prices, rf, base)
    on = run_strategy(prices, rf, replace(base, vol_target=0.10, vol_halflife=H))
    first_forecast = int(np.argmax(on.vol_forecast.notna().to_numpy()))
    assert off.t0 < first_forecast
    if strategy == "trend" and freq == "monthly":
        starts = np.flatnonzero(first_of_period(prices.index, "M").to_numpy())
        assert on.t0 == starts[starts >= first_forecast][0]
    else:
        assert on.t0 == first_forecast
    assert on.score.iloc[: on.t0].isna().all() and (on.signal.iloc[: on.t0] == 0).all()
    assert on.backtest.e == on.t0 + on.params.execution_lag


@pytest.mark.parametrize(
    "strategy, tickers", [("pairs", ("Y", "X")), ("trend", ("X",)), ("mean_reversion", ("Y",))]
)
def test_sized_weights_are_exactly_zero_when_flat(
    prices: pd.DataFrame, rf: pd.Series, strategy: str, tickers: tuple[str, ...]
) -> None:
    params = StrategyParams(
        strategy=strategy,
        tickers=tickers,
        L=63,
        signal_frequency="daily",
        vol_target=0.10,
        vol_refresh="daily",
    )
    run = run_strategy(prices, rf, params)
    flat = run.signal == 0
    assert flat.any() and (~flat).any()
    assert (run.weights[flat] == 0.0).all().all()
    assert not run.weights.isna().any().any()


# ---------------------------------------------------------------- 11. long-only switch (Phase 5)

LONG_ONLY_CASES = [
    {"strategy": "mean_reversion", "tickers": ("Y",), "L": 40},
    {"strategy": "mean_reversion", "tickers": ("Y",), "L": 40, "vol_target": 0.10, "vol_refresh": "weekly"},
    {"strategy": "trend", "tickers": ("X",), "L": 63, "signal_frequency": "daily"},
    {"strategy": "trend", "tickers": ("X",), "L": 63, "vol_target": 0.10, "vol_refresh": "daily"},
]


@pytest.mark.parametrize("config", LONG_ONLY_CASES, ids=lambda c: f"{c['strategy']}-vol{c.get('vol_target')}")
def test_long_only_never_holds_a_negative_position(prices: pd.DataFrame, rf: pd.Series, config: dict) -> None:
    both = run_strategy(prices, rf, StrategyParams(**config))
    long_only = run_strategy(prices, rf, StrategyParams(**config, allow_short=False))
    assert (both.signal < 0).any()  # the switch has something to remove
    assert (long_only.signal >= 0).all() and (long_only.weights >= 0).all().all()
    assert (long_only.backtest.pos >= 0).all() and (long_only.backtest.w >= 0).all().all()
    assert (long_only.backtest.trades["side"] == "long").all()
    # The clip comes after the score: the reported score and t0 are unchanged.
    assert np.array_equal(both.score.to_numpy(), long_only.score.to_numpy(), equal_nan=True)
    assert both.t0 == long_only.t0 and both.split_date == long_only.split_date


@pytest.mark.parametrize("vol_target", [None, 0.10])
def test_long_only_keeps_exactly_the_long_trades(
    prices: pd.DataFrame, rf: pd.Series, vol_target: float | None
) -> None:
    # A short lookback and z_exit = 0 make direct flips common: about 120 trades, 25 of them flips.
    params = StrategyParams(
        strategy="mean_reversion", tickers=("Y",), L=10, z_entry=1.0, z_exit=0.0, vol_target=vol_target
    )
    both = run_strategy(prices, rf, params).backtest.trades
    long_only = run_strategy(prices, rf, replace(params, allow_short=False)).backtest.trades
    longs = both.loc[both["side"] == "long"].reset_index(drop=True)
    flips = (both["entry_date"].iloc[1:].to_numpy() == both["exit_date"].iloc[:-1].to_numpy()).sum()
    assert len(longs) == len(long_only) >= 30 and flips >= 10  # direct flips were exercised
    columns = ["side", "segment", "open", "signal_date", "entry_date", "exit_signal_date", "exit_date"]
    assert longs[[*columns, "holding_days"]].equals(long_only[[*columns, "holding_days"]])
    if vol_target is None:  # unit weights: a flip's 2c cost splits into exactly c + c
        assert np.array_equal(longs["pnl"].to_numpy(), long_only["pnl"].to_numpy())
    else:  # sized weights: the flip's cost split is exact only up to rounding
        np.testing.assert_allclose(longs["pnl"].to_numpy(), long_only["pnl"].to_numpy(), rtol=0, atol=1e-15)


def test_pairs_cannot_be_long_only() -> None:
    with pytest.raises(ValueError, match="pairs needs a short leg"):
        StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60, allow_short=False)
