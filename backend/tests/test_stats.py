"""Engle–Granger, static OLS and half-life (§12, test_stats.py)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import CollinearityWarning
from statsmodels.tsa.stattools import coint

from app.services import stats
from app.services.pipeline import StrategyParams, build_report, run_strategy


def ar1(phi: float, eps: np.ndarray) -> np.ndarray:
    """AR(1) noise s_t = phi * s_{t-1} + eps_t, generated with a linear filter (no Python loop)."""
    return lfilter([1.0], [1.0, -phi], eps)


def test_engle_granger_rarely_flags_independent_random_walks() -> None:
    g = np.random.default_rng(42)
    p = [
        stats.engle_granger(np.cumsum(g.normal(0, 0.01, 750)), np.cumsum(g.normal(0, 0.01, 750))).p_value
        for _ in range(100)
    ]
    assert np.mean(np.array(p) < 0.05) <= 0.12


def test_engle_granger_detects_cointegrated_pairs() -> None:
    g = np.random.default_rng(7)
    hits = []
    for _ in range(100):
        x = np.cumsum(g.normal(0, 0.01, 750))
        y = 0.5 + 1.3 * x + ar1(0.95, g.normal(0, 0.01, 750))
        hits.append(stats.engle_granger(y, x).p_value < 0.05)
    assert np.mean(hits) >= 0.95


def test_collinear_inputs_give_a_null_p_value() -> None:
    x = np.cumsum(np.random.default_rng(1).normal(0, 0.01, 500))
    y = 2.0 * x
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p_raw = coint(y, x, trend="c", autolag="aic")[1]
    assert p_raw == 0.0  # statsmodels calls this "perfectly cointegrated"
    assert any(issubclass(w.category, CollinearityWarning) for w in caught)
    result = stats.engle_granger(y, x)
    assert result.p_value is None and result.collinear


def test_collinear_pair_reports_null_plus_a_warning() -> None:
    g = np.random.default_rng(5)
    n = 1000
    idx = pd.bdate_range("2018-01-01", periods=n).as_unit("ns")
    x = np.log(50.0) + np.cumsum(g.normal(0, 0.01, n))
    y = 2.0 * x + g.normal(0, 7e-5, n)  # the rolling z exists, but coint sees R^2 within 1.5e-6 of 1
    spy = 300.0 * np.exp(np.cumsum(g.normal(0.0003, 0.01, n)))
    prices = pd.DataFrame({"Y": np.exp(y), "X": np.exp(x), "SPY": spy}, index=idx)
    rf = pd.Series(0.02 / 252, index=idx)
    report = build_report(
        run_strategy(prices, rf, StrategyParams(strategy="pairs", tickers=("Y", "X"), L=60))
    )
    assert report.diagnostics.coint_p_is is None and report.diagnostics.collinear_is
    assert any("in-sample Engle–Granger test raised a collinearity warning" in w for w in report.warnings)
    assert not any("no evidence the pair is cointegrated" in w for w in report.warnings)


@pytest.mark.parametrize("seed", range(10))
def test_half_life_recovers_the_true_value(seed: int) -> None:
    phi, n = 0.95, 20_000
    g = np.random.default_rng(seed)
    s = ar1(phi, g.normal(0, 0.01, n))
    true_half_life = -np.log(2) / np.log(phi)  # 13.5 days
    assert stats.half_life(s) == pytest.approx(true_half_life, rel=0.15)
    x = np.cumsum(g.normal(0, 0.01, n))  # the same residual inside a cointegrated pair
    y = 0.4 + 1.1 * x + s
    alpha, beta = stats.static_ols(y, x)
    assert stats.half_life(y - alpha - beta * x) == pytest.approx(true_half_life, rel=0.15)


def test_half_life_is_null_without_mean_reversion() -> None:
    t = np.arange(300, dtype=float)
    assert stats.half_life(1.01**t) is None  # explosive: b = 0.01 > 0
    assert stats.half_life(t) is None  # a straight trend: b = 0
    assert stats.half_life(np.full(100, 3.0)) is None  # constant: undefined


def test_pairs_diagnostics_use_the_is_and_oos_windows(prices: pd.DataFrame) -> None:
    split = prices.index[700]
    d = stats.pairs_diagnostics(prices["Y"], prices["X"], split)
    y, x = np.log(prices["Y"]), np.log(prices["X"])
    is_ = prices.index < split
    assert d.coint_p_is == stats.engle_granger(y[is_], x[is_]).p_value
    assert d.coint_p_oos == stats.engle_granger(y[~is_], x[~is_]).p_value
    assert d.coint_p_is < 0.05  # the fixture pair is cointegrated
    slope, intercept = np.polyfit(x[is_], y[is_], 1)
    assert d.static_beta_is == pytest.approx(slope, rel=1e-10)
    assert d.static_alpha_is == pytest.approx(intercept, rel=1e-9)
    assert d.half_life_is == pytest.approx(stats.half_life(y[is_] - intercept - slope * x[is_]), rel=1e-8)
    assert not d.collinear_is and not d.collinear_oos


def test_engle_granger_is_null_for_a_constant_leg() -> None:
    x = np.cumsum(np.random.default_rng(2).normal(0, 0.01, 300))
    assert stats.engle_granger(np.full(300, 1.5), x) == stats.CointResult(p_value=None, collinear=False)
    assert stats.static_ols(x, np.full(300, 1.5)) is None
