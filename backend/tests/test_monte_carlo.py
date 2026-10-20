"""Monte Carlo: GBM, block and stationary bootstraps, and closed-form VaR/CVaR (§12, test_monte_carlo.py)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from arch.bootstrap import optimal_block_length
from scipy.signal import lfilter
from scipy.stats import norm

from app.services import monte_carlo as mc
from app.services.monte_carlo import MonteCarloInputError
from app.services.pipeline import StrategyParams, run_strategy, strategy_mc_returns

BAND_KEYS = ("p5", "p25", "p50", "p75", "p95")
MODELS = ("gbm", "bootstrap", "stationary")


def history(n: int = 1000, mu: float = 0.0003, sigma: float = 0.012, seed: int = 0) -> np.ndarray:
    """Daily simple returns whose log returns are iid normal."""
    return np.expm1(np.random.default_rng(seed).normal(mu, sigma, n))


def test_shapes_and_seeds() -> None:
    R, T, N = history(), 63, 2000
    a = mc.simulate(R, horizon=T, n_paths=N, block_len=10, seed=7)
    b = mc.simulate(R, horizon=T, n_paths=N, block_len=10, seed=7)
    c = mc.simulate(R, horizon=T, n_paths=N, block_len=10, seed=8)
    assert (a.seed, a.n_hist, a.hist_edges.shape) == (7, 1000, (51,))
    assert a.block_len_used == {"circular": 10, "stationary": 10}
    for name in MODELS:
        model = getattr(a, name)
        assert tuple(model.bands) == BAND_KEYS and all(v.shape == (T + 1,) for v in model.bands.values())
        assert model.sample_paths.shape == (20, T + 1)
        assert model.hist_counts.shape == (50,) and model.terminal.shape == (N,)
    for x, y in ((getattr(a, name), getattr(b, name)) for name in MODELS):
        assert np.array_equal(x.terminal, y.terminal) and np.array_equal(x.sample_paths, y.sample_paths)
        assert all(np.array_equal(x.bands[k], y.bands[k]) for k in BAND_KEYS)
        assert np.array_equal(x.hist_counts, y.hist_counts)
        assert (x.var95, x.var99, x.cvar95, x.cvar99) == (y.var95, y.var99, y.cvar95, y.cvar99)
    assert np.array_equal(a.hist_edges, b.hist_edges)
    for name in MODELS:
        assert not np.array_equal(getattr(a, name).terminal, getattr(c, name).terminal)


def test_bands_and_risk_are_ordered() -> None:
    res = mc.simulate(history(), horizon=126, n_paths=5000, block_len=10, seed=3)
    for model in (res.gbm, res.bootstrap, res.stationary):
        bands = np.vstack([model.bands[k] for k in BAND_KEYS])
        assert (np.diff(bands, axis=0) >= 0).all()  # p5 <= p25 <= ... <= p95 on every day
        assert (bands[:, 0] == 1.0).all() and (model.sample_paths[:, 0] == 1.0).all()
        assert model.var99 >= model.var95
        assert model.cvar95 >= model.var95 and model.cvar99 >= model.var99
    cf = res.closed_form_gbm
    assert cf["var99"] >= cf["var95"] and cf["cvar95"] >= cf["var95"] and cf["cvar99"] >= cf["var99"]


def test_gbm_increments_have_the_right_mean_and_std() -> None:
    mu, sigma, N, T = 0.0004, 0.02, 10_000, 252
    inc = mc.gbm_increments(mu, sigma, np.random.default_rng(11), np.empty((T, N)))
    se_mean, se_std = sigma / math.sqrt(N * T), sigma / math.sqrt(2 * N * T)
    assert abs(inc.mean() - mu) < 5 * se_mean
    assert abs(inc.std(ddof=1) - sigma) < 5 * se_std
    assert sigma**2 / 2 > 15 * se_mean  # so a wrongly added -sigma^2/2 drift would fail the mean check


def test_bootstrap_increments_are_historical_values_in_contiguous_blocks() -> None:
    n, T, N, block = 97, 50, 400, 7
    ell = np.arange(n, dtype=float)  # each value reveals its index
    inc = mc.bootstrap_increments(ell, block, np.random.default_rng(5), np.empty((T, N)))
    assert np.isin(inc, ell).all()
    continues = (np.arange(1, T) % block) != 0  # rows t that continue the block started before t
    step = (inc[1:] - inc[:-1]) % n
    assert (step[continues] == 1).all()  # contiguous modulo n
    assert (step[~continues] != 1).any()  # new blocks start at random positions
    wraps = (inc[:-1] == n - 1) & (inc[1:] == 0)
    assert wraps[continues].any()  # blocks wrap around the end of the history (circular)
    real = history(n=500, seed=6)
    ell_real = np.log1p(real)
    inc_real = mc.bootstrap_increments(ell_real, 10, np.random.default_rng(2), np.empty((40, 300)))
    assert np.isin(inc_real, ell_real).all()


def test_simulated_gbm_risk_matches_the_closed_form() -> None:
    N, T = 200_000, 21
    res = mc.simulate(history(seed=21), horizon=T, n_paths=N, block_len=10, seed=2026)
    m, s = res.mu_daily * T, res.sigma_daily * math.sqrt(T)
    terminal = res.gbm.terminal
    for alpha in (0.95, 0.99):
        p, key = 1 - alpha, round(alpha * 100)
        q = m + s * norm.ppf(p)
        var_cf = 1 - math.exp(q)
        cvar_cf = 1 - math.exp(m + s * s / 2) * norm.cdf((q - m - s * s) / s) / p
        assert res.closed_form_gbm[f"var{key}"] == pytest.approx(var_cf, rel=1e-12)
        assert res.closed_form_gbm[f"cvar{key}"] == pytest.approx(cvar_cf, rel=1e-12)
        Q = np.quantile(terminal, p)
        tail = terminal[terminal <= Q]
        es = tail.mean()
        sim_var, sim_cvar = getattr(res.gbm, f"var{key}"), getattr(res.gbm, f"cvar{key}")
        assert (sim_var, sim_cvar) == (-Q, -es)
        se_var = math.exp(q) * math.sqrt(p * (1 - p) / N) * s / norm.pdf(norm.ppf(p))  # delta method
        se_cvar = math.sqrt((tail.var(ddof=1) + (1 - p) * (Q - es) ** 2) / (N * p))  # full asymptotic SE
        assert abs(sim_var - var_cf) <= 4 * se_var, (alpha, sim_var, var_cf, se_var)
        assert abs(sim_cvar - cvar_cf) <= 4 * se_cvar, (alpha, sim_cvar, cvar_cf, se_cvar)


def test_zero_drift_centres_the_terminal_log_return() -> None:
    R, N, T = history(mu=0.002, sigma=0.01, seed=4), 10_000, 252  # a strong drift: about +0.5 a year
    on = mc.simulate(R, horizon=T, n_paths=N, block_len=10, seed=1, zero_drift=True)
    off = mc.simulate(R, horizon=T, n_paths=N, block_len=10, seed=1)
    se = on.sigma_daily * math.sqrt(T / N)
    assert on.mu_daily == 0.0 and on.sigma_daily == off.sigma_daily
    for model in (on.gbm, on.bootstrap, on.stationary):
        assert abs(np.log1p(model.terminal).mean()) < 5 * se
    for model in (off.gbm, off.bootstrap, off.stationary):
        assert np.log1p(model.terminal).mean() > 50 * se


def test_histogram_keeps_every_path_on_heavy_tailed_input() -> None:
    R = np.expm1(np.random.default_rng(9).standard_t(3, 2000) * 0.01)
    N = 10_000
    res = mc.simulate(R, horizon=252, n_paths=N, block_len=10, seed=5)
    assert all(getattr(res, name).hist_counts.sum() == N for name in MODELS)
    assert np.all(np.diff(res.hist_edges) > 0)
    pooled = np.concatenate([getattr(res, name).terminal for name in MODELS])
    lo, hi = np.quantile(pooled, [0.001, 0.999])  # the edges come from all three models (§10.4)
    assert (res.hist_edges[0], res.hist_edges[-1]) == (lo, hi)
    assert (pooled < res.hist_edges[0]).any() and (pooled > res.hist_edges[-1]).any()  # clipped, not dropped


# ---------------------------------------------------------------- stationary bootstrap (Phase 7)


def reference_stationary(
    ell: np.ndarray, block_len: int, rng: np.random.Generator, T: int, N: int
) -> np.ndarray:
    """The §10.7 snippet for one chunk holding every path: (T, N) increments."""
    t = np.arange(T)[:, None]
    new = rng.random((T, N)) < 1.0 / block_len
    new[0] = True
    start = rng.integers(0, ell.size, size=(T, N))
    last = np.maximum.accumulate(np.where(new, t, 0), axis=0)
    idx = np.take_along_axis(start, last, axis=0) + (t - last)
    return np.take(ell, idx, mode="wrap")


def test_stationary_increments_are_historical_values_in_contiguous_runs() -> None:
    n, T, N, block = 97, 60, 3000, 8
    ell = np.arange(n, dtype=float)  # each value reveals its index
    inc = mc.stationary_increments(ell, block, np.random.default_rng(4), np.empty((T, N)), chunk=700)
    assert np.isin(inc, ell).all()
    step = (inc[1:] - inc[:-1]) % n
    continues = step == 1  # the path continued the block (contiguous modulo n), or a new block started there
    wraps = (inc[:-1] == n - 1) & (inc[1:] == 0)
    assert wraps.any()  # blocks wrap around the end of the history (circular)
    # Restart rate: the share of steps t >= 1 that continue the previous index is (1 - p) + p / n,
    # where p / n counts new blocks that happen to start at the next index.
    p = 1.0 / block
    expected = (1 - p) + p / n
    se = math.sqrt(expected * (1 - expected) / continues.size)
    assert abs(continues.mean() - expected) < 5 * se
    # Each path splits into runs that are contiguous modulo n, one per block, except where a new
    # block starts at the next index by chance (probability 1 / n). With runs cut at the horizon,
    # the pooled mean run length is T / (1 + (T - 1) p (1 - 1/n)), not block_len (§12).
    runs = N + int((~continues).sum())
    expected_len = T / (1 + (T - 1) * p * (1 - 1 / n))
    assert abs(T * N / runs / expected_len - 1) < 0.02


def test_a_block_length_of_one_restarts_every_step() -> None:
    n, T, N = 250, 100, 2000
    inc = mc.stationary_increments(np.arange(n, dtype=float), 1, np.random.default_rng(8), np.empty((T, N)))
    continues = ((inc[1:] - inc[:-1]) % n) == 1
    se = math.sqrt((1 / n) * (1 - 1 / n) / continues.size)
    assert abs(continues.mean() - 1 / n) < 5 * se


def test_chunked_stationary_increments_match_the_one_chunk_reference() -> None:
    ell = np.log1p(history(n=400, seed=12))
    T, N = 40, 1500
    one = mc.stationary_increments(ell, 6, np.random.default_rng(3), np.empty((T, N)), chunk=N)
    assert np.array_equal(one, reference_stationary(ell, 6, np.random.default_rng(3), T, N))
    # Chunks consume the generator chunk by chunk, so the first chunk's paths equal a reference run
    # on that chunk alone.
    chunked = mc.stationary_increments(ell, 6, np.random.default_rng(3), np.empty((T, N)), chunk=500)
    assert np.array_equal(chunked[:, :500], reference_stationary(ell, 6, np.random.default_rng(3), T, 500))
    assert not np.array_equal(chunked, one)


@pytest.mark.parametrize("block_len", [7, "auto"])
def test_the_third_model_leaves_gbm_and_the_block_bootstrap_unchanged(block_len: int | str) -> None:
    # An MA(5) history, so the automatic lengths are above 1 and differ between the two bootstraps.
    g = np.random.default_rng(31)
    R = np.expm1(0.004 * np.convolve(g.normal(size=1504), np.ones(5), mode="valid"))
    T, N, seed = 50, 2000, 99
    res = mc.simulate(R, horizon=T, n_paths=N, block_len=block_len, seed=seed)
    lengths = res.block_len_used
    if block_len == "auto":
        assert lengths["circular"] > lengths["stationary"] > 1
    # The draw order rebuilt by hand: GBM increments, its sample-path choice, bootstrap increments,
    # its sample-path choice. The stationary model draws only after these, with its own length.
    est = mc.estimate(np.asarray(R), zero_drift=False)
    rng = np.random.default_rng(seed)
    for model, fill in (
        (res.gbm, lambda out: mc.gbm_increments(est.mu, est.sigma, rng, out)),
        (res.bootstrap, lambda out: mc.bootstrap_increments(est.ell, lengths["circular"], rng, out)),
        (res.stationary, lambda out: mc.stationary_increments(est.ell, lengths["stationary"], rng, out)),
    ):
        P = mc.log_paths(T, N, fill)
        cols = rng.choice(N, 20, replace=False)
        assert np.array_equal(model.terminal, np.expm1(P[-1]))
        assert np.array_equal(model.sample_paths, np.exp(P[:, cols]).T)
        assert np.array_equal(model.bands["p50"], np.exp(np.quantile(P, 0.5, axis=1)))


def test_automatic_block_lengths_follow_arch_rounded_up_and_clipped() -> None:
    g = np.random.default_rng(6)
    autocorrelated = np.expm1(0.01 * np.convolve(g.normal(size=1600), np.ones(8) / 8, mode="valid"))
    for R in (autocorrelated, history(n=1200, seed=2)):
        res = mc.simulate(R, horizon=21, n_paths=1000, block_len="auto", seed=1)
        est = optimal_block_length(np.log1p(R)).iloc[0]
        for model in ("circular", "stationary"):
            want = int(min(max(math.ceil(est[model]), 1), 63, len(R) // 2))
            assert res.block_len_used[model] == want, (model, est[model])
        assert res.block_len_used["circular"] >= res.block_len_used["stationary"]  # (3/2)^(1/3) apart
    assert (
        mc.simulate(autocorrelated, horizon=21, n_paths=1000, block_len="auto", seed=1).block_len_used[
            "circular"
        ]
        > 3
    )  # the moving average makes the dependence real
    # Clipping: a persistent AR(1) (phi = 0.95) over 2,000 days asks for blocks longer than 63 days.
    # The history is at least 252 days, so floor(n / 2) >= 126 never binds before 63.
    persistent = np.expm1(0.01 * lfilter([1.0], [1.0, -0.95], np.random.default_rng(6).normal(size=2000)))
    assert optimal_block_length(np.log1p(persistent)).iloc[0].min() > 63
    for zero_drift in (False, True):  # the lengths come from ell itself, so zero drift changes nothing
        res = mc.simulate(
            persistent, horizon=21, n_paths=1000, block_len="auto", seed=1, zero_drift=zero_drift
        )
        assert res.block_len_used == {"circular": 63, "stationary": 63}


def test_automatic_block_lengths_round_up_and_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """§10.8 on fixed estimates: round up, at least 1, at most min(63, floor(n_hist / 2))."""
    ell = np.zeros(400)  # only its length matters here; floor(400 / 2) = 200 > 63

    def fixed(stationary: float, circular: float):
        return lambda _x: pd.DataFrame({"stationary": [stationary], "circular": [circular]})

    for (sb, cb), want in [
        ((2.1, 2.4), {"stationary": 3, "circular": 3}),  # 2.1 and 2.4 round down but must go up
        ((0.2, 0.0), {"stationary": 1, "circular": 1}),
        ((5.0, 70.3), {"stationary": 5, "circular": 63}),
        ((62.01, np.inf), {"stationary": 63, "circular": 63}),
    ]:
        monkeypatch.setattr(mc, "optimal_block_length", fixed(sb, cb))
        assert mc.auto_block_lengths(ell) == want, (sb, cb)
    monkeypatch.setattr(mc, "optimal_block_length", fixed(np.nan, 3.0))
    with pytest.raises(MonteCarloInputError, match="undefined"):
        mc.auto_block_lengths(ell)


@pytest.mark.parametrize(
    "R, block_len, message",
    [
        (np.full(300, 0.001), 10, "constant"),  # sigma-hat below 1e-8
        (np.r_[history(n=299), -1.0], 10, "greater than -100%"),
        (history(n=300), 151, "block_len 151"),
        (history(n=251), 10, "at least 252"),
    ],
    ids=["constant", "minus-100pct", "long-blocks", "short-history"],
)
def test_invalid_histories_are_rejected(R: np.ndarray, block_len: int, message: str) -> None:
    with pytest.raises(MonteCarloInputError, match=message):
        mc.simulate(R, horizon=21, n_paths=1000, block_len=block_len, seed=0)


def test_a_strategy_window_without_exposure_is_rejected(prices: pd.DataFrame, varying_rf) -> None:
    rf = varying_rf(prices.index)
    never = StrategyParams(strategy="mean_reversion", tickers=("Y",), L=20, z_entry=5.0)  # |z| <= 4.25
    R, exposed = strategy_mc_returns(run_strategy(prices, rf, never), "full")
    assert not exposed
    assert np.log1p(R).std(ddof=1) > 1e-5  # cash alone varies with the T-bill rate, far above the floor
    with pytest.raises(MonteCarloInputError, match="no position"):
        mc.simulate(R, horizon=21, n_paths=1000, block_len=10, seed=0, has_exposure=exposed)


def test_strategy_windows_feed_the_simulation(prices: pd.DataFrame, rf: pd.Series) -> None:
    run = run_strategy(prices, rf, StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60))
    bt = run.backtest
    full, exposed_full = strategy_mc_returns(run, "full")
    is_, exposed_is = strategy_mc_returns(run, "is")
    assert exposed_full and exposed_is
    assert len(full) == bt.stop - bt.e and len(is_) == bt.split_pos - bt.e
    assert np.array_equal(full, bt.total.iloc[bt.e : bt.stop].to_numpy())
    res = mc.simulate(is_, horizon=21, n_paths=1000, block_len=10, seed=0, has_exposure=exposed_is)
    assert res.n_hist == len(is_)


@pytest.mark.parametrize("seed", [-1, 2**53])
def test_seed_must_fit_in_a_javascript_number(seed: int) -> None:
    with pytest.raises(ValueError):
        mc.simulate(history(), horizon=21, n_paths=1000, block_len=10, seed=seed)
