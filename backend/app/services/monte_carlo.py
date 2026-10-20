"""GBM, circular block bootstrap, stationary bootstrap and closed-form VaR/CVaR (§10). Pure (§0, rule 8).

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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
from arch.bootstrap import optimal_block_length
from scipy.stats import norm

MIN_HIST = 252
MAX_BLOCK_LEN = 63  # the API's block_len range is 1..63 (§11.2); "auto" is clipped to it too
STATIONARY_CHUNK = 2_048  # §10.7: paths per chunk, which bounds the temporaries
SIGMA_FLOOR = 1e-8  # a floor on degenerate input; also keeps the histogram edges increasing
MAX_SEED = 2**53  # JavaScript loses integer precision above this
N_SAMPLE_PATHS = 20
N_BINS = 50
BAND_THREADS = 4  # the bands' quantiles are computed in this many row blocks at once
BAND_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
BAND_KEYS = ("p5", "p25", "p50", "p75", "p95")
ALPHAS = (0.95, 0.99)


class MonteCarloInputError(Exception):
    """The return series can't be simulated (§10.1). The API maps it to HTTP 422."""


def validate_returns(R: object, *, block_len: int | Literal["auto"], has_exposure: bool = True) -> np.ndarray:
    """§10.1 validation. Returns R as a float array.

    Args:
        R: daily simple returns (decimals) of the history window.
        block_len: bootstrap block length in days, or "auto" (§10.8), which is clipped to the
            history instead of checked against it.
        has_exposure: for source="strategy", whether the strategy held a position in the
            window. Cash alone still varies with the T-bill rate, so a sigma floor can't
            detect a strategy that never trades.
    """
    r = np.asarray(R, dtype=float)
    if r.ndim != 1:
        raise ValueError("R must be one-dimensional")
    auto = isinstance(block_len, str)
    if auto and block_len != "auto":
        raise ValueError(f"block_len must be an int or 'auto', got {block_len!r}")
    if not auto and block_len < 1:
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
    if not auto and block_len > n / 2:
        raise MonteCarloInputError(f"block_len {block_len} must be at most half the history ({n} days).")
    return r


def auto_block_lengths(ell: np.ndarray) -> dict[str, int]:
    """Automatic block lengths for the two bootstraps (§10.8), in days.

    The Politis & White (2004) estimator with the Patton, Politis & White (2009) correction,
    from arch.bootstrap.optimal_block_length on the daily log returns ell: its "circular" value
    for the moving-block bootstrap and its "stationary" value (the mean block length) for the
    stationary bootstrap. Each is rounded up and clipped to [1, min(63, floor(n_hist / 2))].
    """
    estimate_ = optimal_block_length(np.asarray(ell, dtype=float)).iloc[0]
    upper = min(MAX_BLOCK_LEN, ell.size // 2)
    out: dict[str, int] = {}
    for model in ("circular", "stationary"):
        value = float(estimate_[model])
        if math.isnan(value):
            raise MonteCarloInputError("The automatic block length is undefined for this return series.")
        out[model] = upper if value >= upper else max(1, math.ceil(value))
    return out


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


def stationary_increments(
    ell: np.ndarray,
    block_len: int,
    rng: np.random.Generator,
    out: np.ndarray,
    chunk: int = STATIONARY_CHUNK,
) -> np.ndarray:
    """Fill out (T, N) in place with a stationary bootstrap of ell (§10.7, Politis & Romano 1994).

    At each step, with probability p = 1 / block_len, a new block starts at a uniform random
    position; otherwise the path continues with the next historical value, circularly. Step 0
    always starts a block, so block lengths are geometric with mean block_len.

    The paths are filled in column chunks of at most `chunk`, in order, and each chunk draws its
    own numbers: the uniforms that decide new blocks, then the start positions. The temporaries
    are then about 3 x T x chunk values instead of 3 x T x N.
    """
    T, N = out.shape
    p = 1.0 / block_len
    t = np.arange(T)[:, None]
    for c0 in range(0, N, chunk):
        cols = slice(c0, min(N, c0 + chunk))
        width = cols.stop - cols.start
        new = rng.random((T, width)) < p
        new[0] = True
        start = rng.integers(0, ell.size, size=(T, width))  # int64
        last = _running_max_down(np.where(new, t, 0))  # the row where the current block began
        idx = np.take_along_axis(start, last, axis=0) + (t - last)
        np.take(ell, idx, out=out[:, cols], mode="wrap")
    return out


def _running_max_down(a: np.ndarray) -> np.ndarray:
    """np.maximum.accumulate(a, axis=0), in place, one row at a time.

    NumPy accumulates down the rows of a row-major array with strided reads; a loop of
    whole-row maxima gives the same values about 20 times faster at (252, 2048).
    """
    for i in range(1, len(a)):
        np.maximum(a[i - 1], a[i], out=a[i])
    return a


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
    """The three models, the closed-form GBM benchmark and the shared histogram edges (§10)."""

    seed: int
    n_hist: int
    mu_daily: float  # the GBM drift used (0 with zero_drift)
    sigma_daily: float
    block_len_used: dict[str, int]  # "circular" and "stationary" (the stationary one is a mean)
    gbm: ModelResult
    bootstrap: ModelResult
    stationary: ModelResult
    closed_form_gbm: dict[str, float]  # var95, var99, cvar95, cvar99
    hist_edges: np.ndarray  # 51 edges shared by all three models
    elapsed_ms_total: float


def _run_model(T: int, N: int, fill: Callable[[np.ndarray], np.ndarray], rng: np.random.Generator) -> dict:
    """Paths, then (in this order) terminal, sample paths, bands and risk for one model."""
    start = time.perf_counter()
    P = log_paths(T, N, fill)
    terminal = np.expm1(P[-1])  # a copy
    cols = rng.choice(N, N_SAMPLE_PATHS, replace=False)
    sample_paths = np.ascontiguousarray(np.exp(P[:, cols]).T)
    bands = np.exp(_daily_quantiles(P))  # leaves P unusable, so it is dropped right after
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


def _daily_quantiles(P: np.ndarray) -> np.ndarray:
    """The BAND_QUANTILES of each row of P (each day), shape (5, T + 1), partitioning P in place.

    np.quantile treats each row on its own, so row blocks give exactly the one-call result, and
    NumPy releases the GIL while it partitions, so the blocks run in parallel: about 3.5 times
    faster at 10,000 paths. overwrite_input avoids copying P.
    """
    blocks = np.array_split(np.arange(P.shape[0]), BAND_THREADS)
    with ThreadPoolExecutor(BAND_THREADS) as pool:
        parts = pool.map(
            lambda rows: np.quantile(P[rows[0] : rows[-1] + 1], BAND_QUANTILES, axis=1, overwrite_input=True),
            [rows for rows in blocks if rows.size],
        )
        return np.concatenate(list(parts), axis=1)


def simulate(
    R: object,
    *,
    horizon: int,
    n_paths: int,
    block_len: int | Literal["auto"],
    seed: int,
    zero_drift: bool = False,
    has_exposure: bool = True,
) -> MonteCarloResult:
    """Run GBM, the circular block bootstrap and the stationary bootstrap from one history (§10).

    Args:
        R: daily simple returns (decimals) of the history window.
        horizon: T, the number of simulated trading days.
        n_paths: N, the number of paths per model (at least 20).
        block_len: the block length in days of both bootstraps (for the stationary bootstrap,
            the mean block length), or "auto" for the §10.8 estimates.
        seed: 0 <= seed < 2^53. Draws come from one generator in a fixed order: GBM
            increments, GBM sample-path choice, bootstrap increments, bootstrap sample-path
            choice, stationary-bootstrap increments, its sample-path choice. The stationary
            model's draws come last, so a seed's GBM and block-bootstrap outputs don't depend on it.
        zero_drift: mu-hat := 0 for GBM and ell := ell - mean(ell) for both bootstraps.
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
    if block_len == "auto":  # on ell = ln(1 + R) itself, so zero_drift can't move it by rounding
        lengths = auto_block_lengths(np.log1p(r))
    else:
        lengths = {"circular": operator.index(block_len), "stationary": operator.index(block_len)}
    rng = np.random.default_rng(seed)

    gbm = _run_model(T, N, lambda out: gbm_increments(est.mu, est.sigma, rng, out), rng)
    boot = _run_model(T, N, lambda out: bootstrap_increments(est.ell, lengths["circular"], rng, out), rng)
    stat = _run_model(T, N, lambda out: stationary_increments(est.ell, lengths["stationary"], rng, out), rng)

    # The 50 bins share edges across the models: the 0.1%..99.9% quantiles of the pooled
    # terminal returns. Values outside go into the first or last bin (np.histogram would drop them).
    models = (gbm, boot, stat)
    lo, hi = np.quantile(np.concatenate([m["terminal"] for m in models]), [0.001, 0.999])
    edges = np.linspace(lo, hi, N_BINS + 1)
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError("histogram edges are not strictly increasing")  # the sigma floor rules this out
    for model in models:
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
        block_len_used=lengths,
        gbm=ModelResult(**gbm),
        bootstrap=ModelResult(**boot),
        stationary=ModelResult(**stat),
        closed_form_gbm=closed,
        hist_edges=edges,
        elapsed_ms_total=(time.perf_counter() - start) * 1000,
    )
