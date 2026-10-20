"""Pairs diagnostics (§5.4), the Kalman-filter hedge ratio (§5.7) and the deflated Sharpe ratio
(§9.1). Pure (§0, rule 8).

Engle–Granger test, static OLS and half-life: the inputs are log prices on the run calendar,
and the results are only displayed; no trading decision uses them.
Kalman filter: the inputs are rebased log prices; the value at t uses data through the close of t.
Deflated Sharpe ratio: the inputs are daily excess returns and daily (not annualized) Sharpe
ratios.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from statsmodels.tools.sm_exceptions import CollinearityWarning
from statsmodels.tsa.stattools import coint

VAR_FLOOR = 1e-12  # same floor as the rolling beta (§5.3)
RESID_VAR_RATIO = 1e-8  # §5.3's collinearity threshold, reused for the Kalman R (§5.7)
EULER_GAMMA = 0.5772156649  # the Euler–Mascheroni constant, to the digits §9.1 gives
SHARPE_VAR_FLOOR = 1e-24  # §9.1: below this, the trials' Sharpe ratios are all the same
STD_FLOOR = 1e-12  # the §8 Sharpe is undefined below this std


@dataclass(frozen=True)
class CointResult:
    """Engle–Granger p-value, or None with collinear=True when statsmodels flags collinearity."""

    p_value: float | None
    collinear: bool


def engle_granger(y: object, x: object) -> CointResult:
    """Engle–Granger test of y regressed on x with a constant (§5.4).

    Uses statsmodels' coint(y, x, trend="c", autolag="aic"), whose critical values account for
    the estimated beta (plain adfuller on OLS residuals does not). On (almost) exactly collinear
    inputs coint returns p = 0 with a CollinearityWarning; that is reported as None.
    A constant input leaves the test undefined and also gives None.
    """
    yv, xv = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if np.ptp(yv) == 0 or np.ptp(xv) == 0:
        return CointResult(p_value=None, collinear=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = float(coint(yv, xv, trend="c", autolag="aic")[1])
    if any(issubclass(w.category, CollinearityWarning) for w in caught):
        return CointResult(p_value=None, collinear=True)
    return CointResult(p_value=p if math.isfinite(p) else None, collinear=False)


def static_ols(y: object, x: object) -> tuple[float, float] | None:
    """Static OLS fit y = alpha + beta * x on the whole window. None when Var(x) <= 1e-12."""
    yv, xv = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    xd = xv - xv.mean()
    sxx = float(np.sum(xd**2))
    if not sxx / (xv.size - 1) > VAR_FLOOR:
        return None
    beta = float(np.sum(xd * (yv - yv.mean())) / sxx)
    return float(yv.mean() - beta * xv.mean()), beta


def half_life(resid: object) -> float | None:
    """Mean-reversion half-life, in trading days, of a residual series s (§5.4).

    Fits ds_t = a + b * s_{t-1} by least squares and returns -ln 2 / ln(1 + b).
    None unless -1 < b < 0.
    """
    s = np.asarray(resid, dtype=float)
    lag, ds = s[:-1], np.diff(s)
    lag_d = lag - lag.mean()
    sxx = float(np.sum(lag_d**2))
    if not sxx > 0:
        return None
    b = float(np.sum(lag_d * (ds - ds.mean())) / sxx)
    if not -1 < b < 0:
        return None
    return -math.log(2) / math.log1p(b)


@dataclass(frozen=True)
class PairsDiagnostics:
    """§5.4 diagnostics. IS = [first loaded date, split_date); OOS = [split_date, last date].

    static_alpha_is and static_beta_is describe ln Y = alpha + beta * ln X on raw (not rebased)
    log adjusted closes. half_life_is is in trading days.
    """

    coint_p_is: float | None
    coint_p_oos: float | None
    half_life_is: float | None
    static_beta_is: float | None
    static_alpha_is: float | None
    collinear_is: bool
    collinear_oos: bool


def pairs_diagnostics(
    y_close: pd.Series, x_close: pd.Series, split_date: date | pd.Timestamp
) -> PairsDiagnostics:
    """Engle–Granger p-values (IS and OOS), static OLS and half-life for Y on X (§5.4)."""
    y, x = np.log(y_close), np.log(x_close)
    in_is = (y.index < pd.Timestamp(split_date)).astype(bool)
    eg_is = engle_granger(y[in_is], x[in_is])
    eg_oos = engle_granger(y[~in_is], x[~in_is])
    fit = static_ols(y[in_is], x[in_is])
    alpha, beta = fit if fit is not None else (None, None)
    hl = half_life(y[in_is] - alpha - beta * x[in_is]) if fit is not None else None
    return PairsDiagnostics(
        coint_p_is=eg_is.p_value,
        coint_p_oos=eg_oos.p_value,
        half_life_is=hl,
        static_beta_is=beta,
        static_alpha_is=alpha,
        collinear_is=eg_is.collinear,
        collinear_oos=eg_oos.collinear,
    )


# ---------------------------------------------------------------- Kalman-filter hedge ratio (§5.7)


@dataclass(frozen=True)
class KalmanFilterResult:
    """The §5.7 filter on positions 0..n-1; every array is NaN for t < L, or everywhere when the
    model is undefined (a flat-x or exactly collinear warm-up window).

    z: the standardized innovation e_t / sqrt(S_t). Its prediction uses data through t - 1.
    alpha, beta: the filtered state (alpha_t, beta_t), using data through the close of t.
    R: the observation-noise variance estimated on days 0..L-1; None when undefined.
    cov: the filtered state covariance P_t as an (n, 2, 2) array, only with keep_cov=True.
    """

    z: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    R: float | None
    cov: np.ndarray | None


def kalman_filter(
    y: object, x: object, L: int, delta: float, *, keep_cov: bool = False
) -> KalmanFilterResult:
    """Kalman filter for a random-walk regression y_t = alpha_t + beta_t * x_t + e_t (§5.7, Chan 2013).

    Args:
        y, x: rebased log prices (y = ln(Y / Y_0), x = ln(X / X_0)) on the run calendar.
        L: initialization window: days 0..L-1 seed the state with their OLS fit. L >= 3.
        delta: state-noise parameter; the state moves by w_t ~ N(0, q I) with q = delta / (1 - delta).
            delta = 0 reproduces expanding-window OLS.
        keep_cov: also return the filtered covariance P_t of every day (for tests).

    The initialization is the posterior of a flat prior after days 0..L-1: s = OLS of y on (1, x),
    R = SSR / (L - 2), P = R (H'H)^-1. Then, for t = L..n-1: predict P + q I, innovation
    e = y_t - h s with h = (1, x_t), S = h P h' + R, z_t = e / sqrt(S), and the Joseph-form update,
    which keeps P symmetric positive definite. The recursion is unrolled into scalar arithmetic
    on the 2x2 matrices; it matches the matrix form within 1e-12 (§12).
    """
    yv, xv = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    n = yv.size
    if xv.size != n:
        raise ValueError("y and x must have the same length")
    if L < 3:
        raise ValueError(f"L must be >= 3 (R divides by L - 2), got {L}")
    if not 0 <= delta < 1:
        raise ValueError(f"delta must be in [0, 1), got {delta}")
    z, alpha, beta = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    cov = np.full((n, 2, 2), np.nan) if keep_cov else None
    if n <= L:
        return KalmanFilterResult(z=z, alpha=alpha, beta=beta, R=None, cov=cov)

    # Initialization from days 0..L-1, with centred sums for accuracy.
    yw, xw = yv[:L], xv[:L]
    x_bar, y_bar = float(xw.mean()), float(yw.mean())
    xd, yd = xw - x_bar, yw - y_bar
    sxx, syy = float(xd @ xd), float(yd @ yd)
    if not sxx / (L - 1) > VAR_FLOOR:  # a flat-x window: H'H is singular in exact arithmetic
        return KalmanFilterResult(z=z, alpha=alpha, beta=beta, R=None, cov=cov)
    b = float(xd @ yd) / sxx
    a = y_bar - b * x_bar
    resid = yw - a - b * xw
    R = float(resid @ resid) / (L - 2)
    if not R > RESID_VAR_RATIO * (syy / (L - 1)):  # exactly collinear legs
        return KalmanFilterResult(z=z, alpha=alpha, beta=beta, R=None, cov=cov)
    # P = R (H'H)^-1, written out: the OLS covariance of (alpha, beta).
    p00, p01, p11 = R * (1.0 / L + x_bar * x_bar / sxx), -R * x_bar / sxx, R / sxx

    q = delta / (1.0 - delta)
    xs, ys = xv.tolist(), yv.tolist()
    sqrt = math.sqrt
    for t in range(L, n):
        xt = xs[t]
        p00 += q  # predict: the state mean is unchanged
        p11 += q
        e = ys[t] - (a + b * xt)  # innovation: the prediction uses data through t - 1
        g0 = p00 + p01 * xt  # P h
        g1 = p01 + p11 * xt
        S = g0 + g1 * xt + R  # h P h' + R
        z[t] = e / sqrt(S)
        k0, k1 = g0 / S, g1 / S  # gain K = P h / S
        a += k0 * e
        b += k1 * e
        # Joseph form P = A P A' + R K K' with A = I - K h, h = (1, xt).
        m00, m01 = 1.0 - k0, -k0 * xt
        m10, m11 = -k1, 1.0 - k1 * xt
        ap00 = m00 * p00 + m01 * p01
        ap01 = m00 * p01 + m01 * p11
        ap10 = m10 * p00 + m11 * p01
        ap11 = m10 * p01 + m11 * p11
        p00 = ap00 * m00 + ap01 * m01 + R * k0 * k0
        p01 = ap00 * m10 + ap01 * m11 + R * k0 * k1
        p11 = ap10 * m10 + ap11 * m11 + R * k1 * k1
        alpha[t], beta[t] = a, b
        if cov is not None:
            cov[t] = ((p00, p01), (p01, p11))
    return KalmanFilterResult(z=z, alpha=alpha, beta=beta, R=R, cov=cov)


# ---------------------------------------------------------------- deflated Sharpe ratio (§9.1)


def expected_max_sharpe(sharpes: object) -> float | None:
    """SR0*: the Sharpe ratio the best of N skill-less trials is expected to show (§9.1).

    SR0* = sqrt(V) * [(1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e))], where V is the
    variance across trials of their daily Sharpe ratios (ddof=1) and g is the Euler–Mascheroni
    constant (Bailey & López de Prado 2014).

    Args:
        sharpes: the daily (not annualized) Sharpe ratio of each trial with a defined Sharpe.

    Returns:
        SR0* as a daily Sharpe ratio, or None when N < 2 or V < 1e-24.
    """
    sr = np.asarray(sharpes, dtype=float)
    n = sr.size
    if n < 2:
        return None
    v = float(np.var(sr, ddof=1))
    if not v >= SHARPE_VAR_FLOOR:
        return None
    factor = (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * math.e))
    return math.sqrt(v) * float(factor)


def probabilistic_sharpe(excess: object, sr_star: float) -> float | None:
    """PSR(SR*): the probability that the true Sharpe ratio exceeds sr_star (§9.1).

    PSR = Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2)) (Bailey & López de
    Prado 2012), where SR is the daily Sharpe ratio of the n daily excess returns, g3 their
    sample skewness and g4 their sample (non-excess) kurtosis, both bias-corrected.

    Args:
        excess: daily excess returns of one configuration.
        sr_star: the benchmark Sharpe ratio, daily.

    Returns:
        The probability, or None when the Sharpe ratio is undefined (n < 2 or std < 1e-12) or
        the term under the square root is <= 0. The caller handles the §8 exposure rule.
    """
    x = np.asarray(excess, dtype=float)
    n = x.size
    if n < 2:
        return None
    sd = float(x.std(ddof=1))
    if not sd >= STD_FLOOR:
        return None
    sr = float(x.mean()) / sd
    g3 = float(skew(x, bias=False))
    g4 = float(kurtosis(x, fisher=False, bias=False))
    term = 1 - g3 * sr + (g4 - 1) / 4 * sr**2
    if not term > 0:  # also catches a NaN moment
        return None
    return float(norm.cdf((sr - sr_star) * math.sqrt(n - 1) / math.sqrt(term)))
