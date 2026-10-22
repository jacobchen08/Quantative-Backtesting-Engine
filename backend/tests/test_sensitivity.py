"""Parameter sensitivity grid and deflated Sharpe ratio (§12, test_sensitivity.py).

Tests sensitivity.run_sensitivity and sensitivity.deflated_sharpe directly.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from app.services import stats
from app.services.backtester import InsufficientDataError, run_backtest
from app.services.pipeline import StrategyParams, run_strategy
from app.services.report import build_report
from app.services.sensitivity import deflated_sharpe, run_grid, run_sensitivity

DEFAULT_REQUESTS = [
    StrategyParams(strategy="mean_reversion", tickers=("Y",), L=60),
    StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60),
    StrategyParams(strategy="trend", tickers=("SPY",), L=252),
]


def reference_sharpe(excess: np.ndarray, held: np.ndarray) -> float | None:
    """§8 Sharpe, written out independently of metrics.py."""
    if len(excess) < 2 or not held.any():
        return None
    sd = float(np.std(excess, ddof=1))
    if sd < 1e-12:
        return None
    return float(np.mean(excess)) / sd * math.sqrt(252)


def reference_sr0_star(daily_sharpes: list[float]) -> float:
    """§9.1 SR0*, written out from the formula."""
    n, g = len(daily_sharpes), 0.5772156649
    v = float(np.var(daily_sharpes, ddof=1))
    return math.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * math.e)))


def reference_psr(x: np.ndarray, sr_star: float) -> float:
    """§9.1 PSR, with the bias-corrected skewness and kurtosis written out from their moments."""
    n = len(x)
    d = x - x.mean()
    m2, m3, m4 = np.mean(d**2), np.mean(d**3), np.mean(d**4)
    g3 = m3 / m2**1.5 * math.sqrt(n * (n - 1)) / (n - 2)
    g4 = ((n + 1) * (m4 / m2**2 - 3) + 6) * (n - 1) / ((n - 2) * (n - 3)) + 3
    sr = x.mean() / x.std(ddof=1)
    return float(norm.cdf((sr - sr_star) * math.sqrt(n - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)))


def config(params: StrategyParams, L: int, z: float | None, split: pd.Timestamp) -> StrategyParams:
    if z is None:
        return replace(params, L=L, split_date=split)
    return replace(params, L=L, z_entry=z, split_date=split)


@pytest.mark.parametrize("params", DEFAULT_REQUESTS, ids=lambda p: p.strategy)
def test_default_grids_run_with_their_own_t0(
    prices: pd.DataFrame, rf: pd.Series, params: StrategyParams
) -> None:
    res = run_sensitivity(prices, rf, params)
    trend = params.strategy == "trend"
    assert res.n_configs_tested == (5 if trend else 20)
    assert res.z_entry_values is None if trend else res.z_entry_values == (1.5, 2.0, 2.5, 3.0)
    assert len(res.sharpe) == 5 and all(len(row) == (1 if trend else 4) for row in res.sharpe)
    assert res.selected == ((252, None) if trend else (60, 2.0))
    assert sum(v is not None for row in res.sharpe for v in row) > 0


def test_shorter_lookbacks_signal_before_the_grids_latest_t0(prices: pd.DataFrame, rf: pd.Series) -> None:
    params = StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60)
    split = run_strategy(prices, rf, params).split_date
    short = run_strategy(prices, rf, config(params, 20, 2.0, split))
    long_ = run_strategy(prices, rf, config(params, 120, 2.0, split))
    assert short.t0 < long_.t0 and (short.signal.iloc[: long_.t0] != 0).any()
    # Forcing the grid's latest t0 on the short lookback trips the §6.1 check, hence one t0 per config.
    with pytest.raises(ValueError, match="before t0"):
        run_backtest(
            R=short.returns[["Y", "X"]],
            W=short.weights,
            s=short.signal,
            rf=rf,
            t0=long_.t0,
            k=1,
            cost_bps=10.0,
            borrow_bps=0.0,
            E0=1.0,
            split_date=split,
        )


@pytest.mark.parametrize("params", DEFAULT_REQUESTS, ids=lambda p: p.strategy)
def test_one_split_and_the_common_window(prices: pd.DataFrame, rf: pd.Series, params: StrategyParams) -> None:
    res = run_sensitivity(prices, rf, params)
    split = run_strategy(prices, rf, params).split_date  # the request's own resolved split
    assert res.window_end == split
    runs = [
        run_strategy(prices, rf, config(params, L, z, split))
        for L in res.L_values
        for z in res.z_entry_values or (None,)
    ]
    assert all(r.split_date == split for r in runs)
    assert res.window_start == prices.index[max(r.backtest.e for r in runs)]
    # Without the common split, the shortest lookback would have picked its own 70% split.
    assert run_strategy(prices, rf, replace(params, L=res.L_values[0])).split_date != split


@pytest.mark.parametrize("params", DEFAULT_REQUESTS, ids=lambda p: p.strategy)
def test_each_cell_is_the_sharpe_on_the_common_window(
    prices: pd.DataFrame, rf: pd.Series, params: StrategyParams
) -> None:
    res = run_sensitivity(prices, rf, params)
    start, end = prices.index.get_loc(res.window_start), prices.index.get_loc(res.window_end)
    assert end - start >= 252
    trials = {}
    for i, L in enumerate(res.L_values):
        for j, z in enumerate(res.z_entry_values or (None,)):
            bt = run_strategy(prices, rf, config(params, L, z, res.window_end)).backtest
            excess = bt.excess.iloc[start:end].to_numpy()
            held = bt.held_id.iloc[start:end].notna().to_numpy()
            want, got = reference_sharpe(excess, held), res.sharpe[i][j]
            assert (got is None) if want is None else got == pytest.approx(want, rel=1e-12), (L, z)
            if want is not None:
                trials[(L, z)] = (want, excess)
    # §9.1 on the same window, from the formulas: the best cell (ties to the first) and this run's.
    dsr = res.deflated
    sr0 = reference_sr0_star([x.mean() / x.std(ddof=1) for _, x in trials.values()])
    best = max(trials, key=lambda cell: trials[cell][0])
    assert dsr.n_trials == len(trials) >= 2
    assert dsr.sr0_star == pytest.approx(sr0 * math.sqrt(252), rel=1e-12)
    assert (dsr.best.L, dsr.best.z_entry) == best and dsr.best.sharpe == pytest.approx(
        trials[best][0], rel=1e-12
    )
    assert dsr.best.dsr == pytest.approx(reference_psr(trials[best][1], sr0), rel=1e-12, abs=1e-15)
    own = trials[res.selected][1]
    assert dsr.selected_dsr == pytest.approx(reference_psr(own, sr0), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize(
    "params",
    [
        StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60, vol_target=0.1),
        StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60, pairs_model="kalman"),
        StrategyParams(strategy="mean_reversion", tickers=("Y",), L=60, vol_target=0.15, vol_refresh="daily"),
    ],
    ids=["pairs-overlay", "kalman", "mean-reversion-overlay"],
)
def test_sharing_one_score_per_L_changes_no_cell(
    prices: pd.DataFrame, rf: pd.Series, params: StrategyParams
) -> None:
    """run_grid computes steps 1-2 once per L; every cell must equal its own, uncached run exactly."""
    split = run_strategy(prices, rf, params).split_date
    Ls, zs = (40, 60), (1.5, 2.0, 2.5)
    runs = run_grid(prices, rf, params, Ls, zs, split, label="test", where="here")
    for (L, z), run in runs.items():
        alone = run_strategy(prices, rf, config(params, L, z, split), trades=False)
        pd.testing.assert_series_equal(run.score, alone.score, check_exact=True)
        pd.testing.assert_frame_equal(run.weights, alone.weights, check_exact=True)
        pd.testing.assert_series_equal(run.backtest.excess, alone.backtest.excess, check_exact=True)


def test_a_configuration_without_exposure_gives_a_null_cell(prices: pd.DataFrame, rf: pd.Series) -> None:
    params = StrategyParams(strategy="mean_reversion", tickers=("Y",), L=60)
    res = run_sensitivity(prices, rf, params, z_entry_values=(5.0,))  # calm data: |z| never exceeds 5
    assert res.sharpe == [[None]] * 5
    assert res.selected is None  # the request's own z_entry = 2.0 isn't in this grid


@pytest.mark.parametrize("offset", [0, 61, 122, 183, 244])
def test_the_trend_grid_needs_more_history_than_one_backtest(
    prices: pd.DataFrame, rf: pd.Series, offset: int
) -> None:
    window = slice(offset, offset + 756)  # about 3 years of trading days
    p, r = prices.iloc[window], rf.iloc[window]
    params = StrategyParams(strategy="trend", tickers=("SPY",), L=252)
    m = build_report(run_strategy(p, r, params)).metrics
    assert m["is"].n_days >= 252 and m["oos"].n_days >= 126
    with pytest.raises(InsufficientDataError, match=r"L = 378 leaves only \d+ in-sample days") as err:
        run_sensitivity(p, r, params)
    assert err.value.n_is < 252 and "drop L = 378 from the grid" in str(err.value)


# ---------------------------------------------------------------- deflated Sharpe ratio (§9.1)


def cells(*excess: np.ndarray) -> dict:
    """A one-row grid: cell (L, None) for L = 1, 2, ... with these daily excess returns."""
    return {(L, None): (x, True) for L, x in enumerate(excess, start=1)}


def test_one_trial_or_equal_sharpes_give_a_null_dsr() -> None:
    x = np.random.default_rng(3).normal(0.0005, 0.01, 300)
    one = deflated_sharpe(cells(x), selected=(1, None))
    assert (one.n_trials, one.sr0_star, one.selected_dsr) == (1, None, None)
    assert one.best is not None and one.best.dsr is None and one.best.sharpe > 0  # still named
    same = deflated_sharpe(cells(x, x.copy(), x.copy()), selected=(2, None))
    assert (same.n_trials, same.sr0_star, same.selected_dsr, same.best.dsr) == (3, None, None, None)
    assert same.best.L == 1  # a tie goes to the first cell in grid order
    none = deflated_sharpe({(1, None): None, (2, None): (np.zeros(300), False)}, selected=None)
    assert (none.n_trials, none.sr0_star, none.best) == (0, None, None)


def test_more_trials_never_lower_sr0_star() -> None:
    previous = 0.0
    for n in range(2, 201):
        sharpes = np.linspace(-1.0, 1.0, n)
        sharpes *= 0.05 / sharpes.std(ddof=1)  # the same variance for every N
        sr0 = stats.expected_max_sharpe(sharpes)
        assert sr0 > previous, n
        previous = sr0


def noise_grid(n_seeds: int, n_days: int = 1000) -> tuple[dict, int, int, pd.DataFrame]:
    """The default mean-reversion grid (5 L x 4 z_entry, z_exit 0.5, lag 1) on pure-noise
    prices, vectorized over seeds: column j is seed j. Simple returns are iid N(0, 1.5%^2), rf = 0
    and costs are 0, so no configuration has any skill.

    This is the §5.1-§6.1 arithmetic of run_strategy, written for a 2-D array; the calibration
    test checks it against run_sensitivity on seed 0. Returns {(L, z): daily excess (days x
    seeds)}, the common window [start, end) and the closes.
    """
    rng = np.random.default_rng(20261017)
    idx = pd.bdate_range("2015-01-01", periods=n_days).as_unit("ns")
    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.015, (n_days, n_seeds)), axis=0), index=idx
    )
    r = (close / close.shift(1) - 1).fillna(0.0).to_numpy()
    lc = np.log(close / close.iloc[0])
    excess = {}
    for L in (20, 40, 60, 90, 120):
        roll = lc.rolling(L, min_periods=L)
        m, v = roll.mean(), roll.var()
        z = ((lc - m) / np.sqrt(v.where(v > 1e-12))).to_numpy()
        for z_entry in (1.5, 2.0, 2.5, 3.0):
            short = np.where(np.isnan(z), 0.0, np.where(z > z_entry, 1.0, np.where(z < 0.5, 0.0, np.nan)))
            long_ = np.where(np.isnan(z), 0.0, np.where(z < -z_entry, 1.0, np.where(z > -0.5, 0.0, np.nan)))
            signal = pd.DataFrame(long_).ffill().fillna(0.0) - pd.DataFrame(short).ffill().fillna(0.0)
            held = signal.shift(2).fillna(0.0).to_numpy()  # lag 1, then held into the next day
            excess[(L, z_entry)] = held * r
    e_own = 60  # the request's own L = 60: t0 = 59, e = t0 + 1
    return excess, 120, e_own + 7 * (n_days - e_own) // 10, close


def test_the_dsr_rarely_passes_on_pure_noise() -> None:
    excess, start, end, close = noise_grid(200)
    # The vectorized grid is the engine's own arithmetic: check it against run_sensitivity on seed 0.
    prices = pd.DataFrame({"Y": close[0], "SPY": close[1]})
    params = StrategyParams(strategy="mean_reversion", tickers=("Y",), L=60, cost_bps=0.0)
    res = run_sensitivity(prices, pd.Series(0.0, index=prices.index), params)
    assert res.window_start == prices.index[start] and res.window_end == prices.index[end]

    def window(j: int) -> dict:  # held a position <=> a non-zero excess return, on continuous noise
        return {cell: (x[start:end, j], bool(np.any(x[start:end, j] != 0))) for cell, x in excess.items()}

    assert deflated_sharpe(window(0), selected=(60, 2.0)) == res.deflated  # bit for bit
    # 200 seeds: how often does the best of 20 skill-less configurations look significant?
    passed = 0
    for j in range(200):
        best = deflated_sharpe(window(j), selected=(60, 2.0)).best
        passed += best is not None and best.dsr is not None and best.dsr > 0.95
    assert passed / 200 <= 0.10  # these 200 seeds: 2 pass (1.0%); 1.5-3.0% across the prototype's grids
