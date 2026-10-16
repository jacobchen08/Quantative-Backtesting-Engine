"""Monte Carlo: GBM, block bootstrap and closed-form VaR/CVaR (§12, test_monte_carlo.py)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from app.services import monte_carlo as mc
from app.services.monte_carlo import MonteCarloInputError
from app.services.pipeline import StrategyParams, run_strategy, strategy_mc_returns

BAND_KEYS = ("p5", "p25", "p50", "p75", "p95")


def history(n: int = 1000, mu: float = 0.0003, sigma: float = 0.012, seed: int = 0) -> np.ndarray:
    """Daily simple returns whose log returns are iid normal."""
    return np.expm1(np.random.default_rng(seed).normal(mu, sigma, n))


def test_shapes_and_seeds() -> None:
    R, T, N = history(), 63, 2000
    a = mc.simulate(R, horizon=T, n_paths=N, block_len=10, seed=7)
    b = mc.simulate(R, horizon=T, n_paths=N, block_len=10, seed=7)
    c = mc.simulate(R, horizon=T, n_paths=N, block_len=10, seed=8)
    assert (a.seed, a.n_hist, a.hist_edges.shape) == (7, 1000, (51,))
    for model in (a.gbm, a.bootstrap):
        assert tuple(model.bands) == BAND_KEYS and all(v.shape == (T + 1,) for v in model.bands.values())
        assert model.sample_paths.shape == (20, T + 1)
        assert model.hist_counts.shape == (50,) and model.terminal.shape == (N,)
    for x, y in ((a.gbm, b.gbm), (a.bootstrap, b.bootstrap)):
        assert np.array_equal(x.terminal, y.terminal) and np.array_equal(x.sample_paths, y.sample_paths)
        assert all(np.array_equal(x.bands[k], y.bands[k]) for k in BAND_KEYS)
        assert np.array_equal(x.hist_counts, y.hist_counts)
        assert (x.var95, x.var99, x.cvar95, x.cvar99) == (y.var95, y.var99, y.cvar95, y.cvar99)
    assert np.array_equal(a.hist_edges, b.hist_edges)
    assert not np.array_equal(a.gbm.terminal, c.gbm.terminal)
    assert not np.array_equal(a.bootstrap.terminal, c.bootstrap.terminal)


def test_bands_and_risk_are_ordered() -> None:
    res = mc.simulate(history(), horizon=126, n_paths=5000, block_len=10, seed=3)
    for model in (res.gbm, res.bootstrap):
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
    for model in (on.gbm, on.bootstrap):
        assert abs(np.log1p(model.terminal).mean()) < 5 * se
    for model in (off.gbm, off.bootstrap):
        assert np.log1p(model.terminal).mean() > 50 * se


def test_histogram_keeps_every_path_on_heavy_tailed_input() -> None:
    R = np.expm1(np.random.default_rng(9).standard_t(3, 2000) * 0.01)
    N = 10_000
    res = mc.simulate(R, horizon=252, n_paths=N, block_len=10, seed=5)
    assert res.gbm.hist_counts.sum() == N and res.bootstrap.hist_counts.sum() == N
    assert np.all(np.diff(res.hist_edges) > 0)
    pooled = np.concatenate([res.gbm.terminal, res.bootstrap.terminal])
    assert (pooled < res.hist_edges[0]).any() and (pooled > res.hist_edges[-1]).any()  # clipped, not dropped


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
