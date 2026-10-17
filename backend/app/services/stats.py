"""Pairs diagnostics (§5.4) and the deflated Sharpe ratio (§9.1). Pure (§0, rule 8).

Engle–Granger test, static OLS and half-life: the inputs are log prices on the run calendar,
and the results are only displayed; no trading decision uses them.
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
