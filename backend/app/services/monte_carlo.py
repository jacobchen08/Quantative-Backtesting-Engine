"""GBM, circular block bootstrap and closed-form VaR/CVaR (§10). Pure (§0, rule 8).

Inputs are daily simple returns R as decimals, and one simulation step is one trading day.
A path array P holds cumulative log returns, P[t, j] for t = 0..T with P[0] = 0, so the
equity multiple is exp(P) and the terminal simple return is expm1(P[T]). VaR and CVaR are
positive loss fractions of the starting equity.

The seed must be given: resolving a missing seed (secrets.randbelow) happens at the API
boundary, so this module stays deterministic.
"""

from __future__ import annotations

import math
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

MIN_HIST = 252
SIGMA_FLOOR = 1e-8  # a floor on degenerate input; also keeps the histogram edges increasing
MAX_SEED = 2**53  # JavaScript loses integer precision above this
N_SAMPLE_PATHS = 20
N_BINS = 50
BAND_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
BAND_KEYS = ("p5", "p25", "p50", "p75", "p95")
ALPHAS = (0.95, 0.99)


class MonteCarloInputError(Exception):
    """The return series can't be simulated (§10.1). The API maps it to HTTP 422."""


def validate_returns(R: object, *, block_len: int, has_exposure: bool = True) -> np.ndarray:
    """§10.1 validation. Returns R as a float array.

    Args:
        R: daily simple returns (decimals) of the history window.
        block_len: bootstrap block length in days.
        has_exposure: for source="strategy", whether the strategy held a position in the
            window. Cash alone still varies with the T-bill rate, so a sigma floor can't
            detect a strategy that never trades.
    """
    r = np.asarray(R, dtype=float)
    if r.ndim != 1:
        raise ValueError("R must be one-dimensional")
    if block_len < 1:
        raise ValueError(f"block_len must be >= 1, got {block_len}")
    n = r.size
    if n < MIN_HIST:
        raise MonteCarloInputError(
            f"Monte Carlo needs at least {MIN_HIST} daily returns; the window has {n}."
        )
    if not np.isfinite(r).all():
        raise MonteCarloInputError("The return series contains NaN or infinite values.")
    if not (r > -1).all():
        raise MonteCarloInputError(
            "Every daily return must be greater than -100%; ln(1 + R) is undefined otherwise."
        )
    if not has_exposure:
        raise MonteCarloInputError(
            "The strategy holds no position in this window, so there is no strategy risk to simulate."
        )
    sigma = float(np.log1p(r).std(ddof=1))
    if not sigma >= SIGMA_FLOOR:
        raise MonteCarloInputError(
            f"The daily log returns are (almost) constant (std {sigma:.1e} < {SIGMA_FLOOR:.0e})."
        )
    if block_len > n / 2:
        raise MonteCarloInputError(f"block_len {block_len} must be at most half the history ({n} days).")
    return r


@dataclass(frozen=True)
class Estimate:
    """Per-day estimates from the history (Delta t = 1 trading day).

    ell: daily log returns ln(1 + R) that the bootstrap resamples (demeaned with zero_drift).
    mu: GBM drift of the log price per day (0 with zero_drift).
    sigma: standard deviation of the daily log returns, ddof=1.
    """

    ell: np.ndarray
    mu: float
    sigma: float


def estimate(r: np.ndarray, *, zero_drift: bool) -> Estimate:
    """mu-hat = mean(ell) and sigma-hat = std(ell, ddof=1) of ell = ln(1 + R) (§10.1)."""
    ell = np.log1p(r)
    mu, sigma = float(ell.mean()), float(ell.std(ddof=1))
    if zero_drift:
        return Estimate(ell=ell - mu, mu=0.0, sigma=sigma)
    return Estimate(ell=ell, mu=mu, sigma=sigma)


def gbm_increments(mu: float, sigma: float, rng: np.random.Generator, out: np.ndarray) -> np.ndarray:
    """Fill out (T, N) in place with iid GBM log increments mu + sigma * Z (§10.2).

    There is no -sigma^2 / 2 term: mu is already the drift of the log price.
    """
    rng.standard_normal(out=out)
    out *= sigma
    out += mu
    return out


def bootstrap_increments(
    ell: np.ndarray, block_len: int, rng: np.random.Generator, out: np.ndarray
) -> np.ndarray:
    """Fill out (T, N) in place with a circular moving-block bootstrap of ell (§10.3).

    Each path is a sequence of blocks of block_len consecutive historical log returns,
    starting at uniform random positions and wrapping around the end of the history.
    """
    T, N = out.shape
    nb = -(-T // block_len)
    idx = rng.integers(0, ell.size, size=(nb, 1, N)) + np.arange(block_len)[None, :, None]  # int64
    idx = idx.reshape(nb * block_len, N)[:T]  # time-major (T, N), C-contiguous
    np.take(ell, idx, out=out, mode="wrap")  # "wrap" = circular, and no output buffering
    return out


def log_paths(T: int, N: int, fill: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Build the (T + 1, N) cumulative log-return paths in place; only one such array exists."""
    P = np.empty((T + 1, N))
    P[0] = 0.0
    fill(P[1:])
    np.cumsum(P, axis=0, out=P)
    return P


def empirical_var_cvar(terminal: np.ndarray, alpha: float) -> tuple[float, float]:
    """VaR_a = -Q_{1-a}(terminal) and CVaR_a = -mean(terminal | terminal <= Q_{1-a}) (§10.4)."""
    q = float(np.quantile(terminal, 1 - alpha))
    return -q, -float(terminal[terminal <= q].mean())


def closed_form_gbm(mu: float, sigma: float, T: int, alpha: float) -> tuple[float, float]:
    """Closed-form VaR and CVaR of the GBM terminal simple return (§10.5).

    X = terminal log return ~ N(m, s^2), m = mu * T, s = sigma * sqrt(T), p = 1 - alpha,
    q = m + s * Phi^-1(p). VaR = 1 - e^q; CVaR = 1 - e^(m + s^2/2) * Phi((q - m - s^2) / s) / p.
    """
    m, s, p = mu * T, sigma * math.sqrt(T), 1 - alpha
    q = m + s * float(norm.ppf(p))
    var = 1 - math.exp(q)
    cvar = 1 - math.exp(m + s * s / 2) * float(norm.cdf((q - m - s * s) / s)) / p
    return var, cvar


@dataclass(frozen=True)
class ModelResult:
    """§10.4 outputs of one model.

    bands: p5..p95 of the equity multiple for t = 0..T (all 1 at t = 0).
    sample_paths: (20, T + 1) equity multiples.
    hist_counts: 50 counts on the shared edges; they sum to N.
    terminal: the N terminal simple returns (kept for tests; not part of the API response).
    """

    bands: dict[str, np.ndarray]
    sample_paths: np.ndarray
    hist_counts: np.ndarray
    var95: float
    var99: float
    cvar95: float
    cvar99: float
    elapsed_ms: float
    terminal: np.ndarray


@dataclass(frozen=True)
class MonteCarloResult:
    """Both models, the closed-form GBM benchmark and the shared histogram edges (§10)."""

    seed: int
    n_hist: int
    mu_daily: float  # the GBM drift used (0 with zero_drift)
    sigma_daily: float
    gbm: ModelResult
    bootstrap: ModelResult
    closed_form_gbm: dict[str, float]  # var95, var99, cvar95, cvar99
    hist_edges: np.ndarray  # 51 edges shared by both models
    elapsed_ms_total: float


def _run_model(T: int, N: int, fill: Callable[[np.ndarray], np.ndarray], rng: np.random.Generator) -> dict:
    """Paths, then (in this order) terminal, sample paths, bands and risk for one model."""
    start = time.perf_counter()
    P = log_paths(T, N, fill)
    terminal = np.expm1(P[-1])  # a copy
    cols = rng.choice(N, N_SAMPLE_PATHS, replace=False)
    sample_paths = np.ascontiguousarray(np.exp(P[:, cols]).T)
    # overwrite_input avoids a full copy but leaves P unusable, so P is dropped right after.
    bands = np.exp(np.quantile(P, BAND_QUANTILES, axis=1, overwrite_input=True))
    del P
    risk = {alpha: empirical_var_cvar(terminal, alpha) for alpha in ALPHAS}
    return {
        "bands": dict(zip(BAND_KEYS, bands, strict=True)),
        "sample_paths": sample_paths,
        "var95": risk[0.95][0],
        "var99": risk[0.99][0],
        "cvar95": risk[0.95][1],
        "cvar99": risk[0.99][1],
        "elapsed_ms": (time.perf_counter() - start) * 1000,
        "terminal": terminal,
    }


def simulate(
    R: object,
    *,
    horizon: int,
    n_paths: int,
    block_len: int,
    seed: int,
    zero_drift: bool = False,
    has_exposure: bool = True,
) -> MonteCarloResult:
    """Run GBM and the block bootstrap from one history (§10).

    Args:
        R: daily simple returns (decimals) of the history window.
        horizon: T, the number of simulated trading days.
        n_paths: N, the number of paths per model (at least 20).
        block_len: bootstrap block length in days.
        seed: 0 <= seed < 2^53. Draws come from one generator in a fixed order: GBM
            increments, GBM sample-path choice, bootstrap increments, bootstrap sample-path choice.
        zero_drift: mu-hat := 0 for GBM and ell := ell - mean(ell) for the bootstrap.
        has_exposure: see validate_returns.

    Raises:
        MonteCarloInputError: the history fails a §10.1 check.
    """
    start = time.perf_counter()
    seed, T, N = operator.index(seed), operator.index(horizon), operator.index(n_paths)
    if not 0 <= seed < MAX_SEED:
        raise ValueError(f"seed must be in [0, 2^53), got {seed}")
    if T < 1 or N < N_SAMPLE_PATHS:
        raise ValueError(f"need horizon >= 1 and n_paths >= {N_SAMPLE_PATHS}, got {T} and {N}")
    r = validate_returns(R, block_len=block_len, has_exposure=has_exposure)
    est = estimate(r, zero_drift=zero_drift)
    rng = np.random.default_rng(seed)

    gbm = _run_model(T, N, lambda out: gbm_increments(est.mu, est.sigma, rng, out), rng)
    boot = _run_model(T, N, lambda out: bootstrap_increments(est.ell, block_len, rng, out), rng)

    # The 50 bins share edges across both models: the 0.1%..99.9% quantiles of the pooled
    # terminal returns. Values outside go into the first or last bin (np.histogram would drop them).
    lo, hi = np.quantile(np.concatenate([gbm["terminal"], boot["terminal"]]), [0.001, 0.999])
    edges = np.linspace(lo, hi, N_BINS + 1)
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError("histogram edges are not strictly increasing")  # the sigma floor rules this out
    for model in (gbm, boot):
        model["hist_counts"] = np.histogram(np.clip(model["terminal"], edges[0], edges[-1]), bins=edges)[0]

    closed: dict[str, float] = {}
    for alpha in ALPHAS:
        var, cvar = closed_form_gbm(est.mu, est.sigma, T, alpha)
        closed[f"var{round(alpha * 100)}"], closed[f"cvar{round(alpha * 100)}"] = var, cvar
    return MonteCarloResult(
        seed=seed,
        n_hist=r.size,
        mu_daily=est.mu,
        sigma_daily=est.sigma,
        gbm=ModelResult(**gbm),
        bootstrap=ModelResult(**boot),
        closed_form_gbm=closed,
        hist_edges=edges,
        elapsed_ms_total=(time.perf_counter() - start) * 1000,
    )
