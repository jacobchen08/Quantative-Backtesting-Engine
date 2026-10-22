"""run_strategy(): the §5.0 steps that turn prices into a backtest. Pure (§0, rule 8).

Prices and the risk-free rate come in as pandas objects on the run calendar; results go out as
dataclasses. run_strategy is the single place that runs the §5.0 steps; everything else (the
report, the sensitivity grid, the walk-forward, the tests) goes through it:

    report.py         build_report(): metrics, benchmark, trades and chart series of one run
    sensitivity.py    run_sensitivity(): the §9 parameter grid and its deflated Sharpe ratios
    walk_forward.py   run_walk_forward(): re-optimization over that grid, fold by fold (§9.2)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from app.services import backtester, sizing, strategies
from app.services.backtester import BacktestResult, NoValidScoreError

MARKET = "SPY"  # always loaded: the pairs benchmark and the beta reference (§4.5, §6.4)
STRATEGIES = ("mean_reversion", "pairs", "trend", "trend_portfolio")
TREND_STRATEGIES = ("trend", "trend_portfolio")  # no z_entry axis (§9)
PORTFOLIO_LEGS = (2, 12)  # the trend portfolio's number of tickers (§5.8)


@dataclass(frozen=True)
class StrategyParams:
    """A resolved strategy configuration; the API fills in every default before calling.

    Units: L and vol_halflife in trading days; z thresholds in standard deviations; cost_bps
    one-way per leg in basis points; borrow_bps an annual fee on short notional in basis points;
    vol_target annualized as a decimal (per asset for the trend portfolio, which needs one);
    split_date is the requested split (None = the 70% default). allow_short=False clips the
    signal at 0 (§5.9). pairs_model picks the rolling-OLS (§5.3) or Kalman (§5.7) score, with
    kalman_delta its state-noise parameter. Fields that don't apply to the strategy are ignored.
    """

    strategy: Literal["mean_reversion", "pairs", "trend", "trend_portfolio"]
    tickers: tuple[str, ...]
    L: int
    z_entry: float | None = 2.0
    z_exit: float | None = 0.5
    signal_frequency: Literal["monthly", "daily"] = "monthly"
    hedge_mode: Literal["fixed_at_entry", "rolling"] = "fixed_at_entry"
    pairs_model: Literal["ols", "kalman"] = "ols"
    kalman_delta: float = 1e-6
    cost_bps: float = 10.0
    execution_lag: int = 1
    borrow_bps: float = 0.0
    allow_short: bool = True
    vol_target: float | None = None
    vol_halflife: int = 42
    vol_refresh: Literal["daily", "weekly", "monthly"] = "weekly"
    max_leverage: float = 2.0
    split_date: date | None = None
    initial_capital: float = 100_000.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "tickers", tuple(self.tickers))
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        n_distinct = len(set(self.tickers))
        if self.strategy == "trend_portfolio":
            lo, hi = PORTFOLIO_LEGS
            if not lo <= len(self.tickers) <= hi or n_distinct != len(self.tickers):
                raise ValueError(f"trend_portfolio needs {lo} to {hi} distinct tickers, got {self.tickers}")
            if self.vol_target is None:
                raise ValueError("trend_portfolio is always vol-scaled per asset, so it needs a vol_target")
        else:
            n_legs = 2 if self.strategy == "pairs" else 1
            if len(self.tickers) != n_legs or n_distinct != n_legs:
                raise ValueError(f"{self.strategy} needs {n_legs} distinct ticker(s), got {self.tickers}")
        if self.L < 2:
            raise ValueError(f"L must be >= 2, got {self.L}")
        if self.strategy in TREND_STRATEGIES:
            if self.signal_frequency not in ("monthly", "daily"):
                raise ValueError(f"unknown signal_frequency {self.signal_frequency!r}")
        elif self.z_entry is None or self.z_exit is None or not 0 <= self.z_exit < self.z_entry:
            raise ValueError(f"need 0 <= z_exit < z_entry, got z_exit={self.z_exit}, z_entry={self.z_entry}")
        if self.strategy == "pairs":
            if self.hedge_mode not in ("fixed_at_entry", "rolling"):
                raise ValueError(f"unknown hedge_mode {self.hedge_mode!r}")
            if self.pairs_model not in ("ols", "kalman"):
                raise ValueError(f"unknown pairs_model {self.pairs_model!r}")
            if self.pairs_model == "kalman" and (not 0 <= self.kalman_delta < 1 or self.L < 3):
                raise ValueError("the Kalman model needs 0 <= kalman_delta < 1 and L >= 3")
        if not 0 <= self.execution_lag <= 5:
            raise ValueError(f"execution_lag must be in 0..5, got {self.execution_lag}")
        if not self.cost_bps >= 0:
            raise ValueError(f"cost_bps must be >= 0, got {self.cost_bps}")
        if not self.borrow_bps >= 0:
            raise ValueError(f"borrow_bps must be >= 0, got {self.borrow_bps}")
        if self.strategy == "pairs" and not self.allow_short:
            raise ValueError("pairs needs a short leg, so allow_short=false is not available")
        if self.vol_target is not None:
            if not self.vol_target > 0 or not self.max_leverage > 0 or self.vol_halflife < 1:
                raise ValueError("need vol_target > 0, max_leverage > 0 and vol_halflife >= 1")
            if self.vol_refresh not in ("daily", "weekly", "monthly"):
                raise ValueError(f"unknown vol_refresh {self.vol_refresh!r}")
        if not self.initial_capital > 0:
            raise ValueError(f"initial_capital must be > 0, got {self.initial_capital}")


@dataclass(frozen=True)
class StrategyRun:
    """Everything the §5.0 steps produce.

    Strategy outputs (score, signal, weights, beta, vol_forecast, scale_used and the leg_*
    frames) are unlagged: the value at date t uses data through the close of t. The backtest
    holds the executed, lagged series and the evaluation window.

    For the trend portfolio (§5.8) score, vol_forecast and scale_used are None; each leg's are in
    leg_scores, leg_signals, leg_vol_forecasts and leg_scale_used (one column per ticker), and
    signal is the "invested" flag, 1 where any leg's signal is non-zero.
    """

    params: StrategyParams
    prices: pd.DataFrame  # adjusted closes of the legs and SPY on the run calendar
    rf: pd.Series
    returns: pd.DataFrame  # simple daily returns of the same columns (first row NaN)
    score: pd.Series | None  # reported score: masked z (A, B) or masked, sampled m (C); None for D
    signal: pd.Series
    unit_weights: pd.DataFrame
    weights: pd.DataFrame  # W: sized target weights (the unit weights when the overlay is off)
    beta: pd.Series | None  # pairs: rolling (or filtered, with the Kalman model) beta
    beta_used: pd.Series | None  # pairs: the hedge ratio held
    vol_forecast: pd.Series | None  # overlay: annualized sigma
    scale_used: pd.Series | None  # overlay: the scale applied
    t0: int
    split_date: pd.Timestamp
    backtest: BacktestResult
    leg_scores: pd.DataFrame | None = None  # trend portfolio: each leg's masked, sampled m
    leg_signals: pd.DataFrame | None = None
    leg_vol_forecasts: pd.DataFrame | None = None
    leg_scale_used: pd.DataFrame | None = None


@dataclass(frozen=True)
class Scored:
    """Steps 1 and 2 of §5.0 for strategies A, B and C: the score after masking, the pairs beta
    and the vol forecast (None without the overlay). They depend on L but not on the thresholds."""

    score: pd.Series
    beta: pd.Series | None
    sigma: pd.Series | None


ScoreCache = dict[int, Scored]  # Scored by L, shared by the runs of one grid


def run_strategy(
    prices: pd.DataFrame,
    rf: pd.Series,
    params: StrategyParams,
    *,
    trades: bool = True,
    score_cache: ScoreCache | None = None,
) -> StrategyRun:
    """Run the §5.0 steps in order: score, vol forecast and mask, signal, unit weights, sizing,
    then t0, resolve_split and the backtester.

    Args:
        prices: adjusted closes with one column per ticker, including SPY, indexed by the run
            calendar (tz-naive datetime64[ns], ascending, unique). No NaN, all > 0.
        rf: daily risk-free rate (decimal) on the same calendar (§2.3, §4.5).
        params: the resolved configuration.
        trades: False skips the trade bookkeeping (run_backtest's trades), for the grids.
        score_cache: for the grids, whose configurations differ only in L and z_entry: steps 1
            and 2 are computed once per L and reused for every z_entry. The caller passes the
            same prices, rf and other parameters to every run that shares the cache.

    Raises:
        TypeError: rf is not a pandas Series.
        NoValidScoreError: the score is NaN on every day (§6.2).
        InsufficientDataError: fewer than 252 IS or 126 OOS evaluated days (§6.3).
    """
    cols = _check_inputs(prices, rf, params)
    p = params
    legs = list(p.tickers)
    prices = prices[cols]
    returns = prices / prices.shift(1) - 1  # §2.1 simple returns; no pct_change
    if p.strategy == "trend_portfolio":
        return _run_trend_portfolio(prices, rf, returns, p, trades=trades)
    overlay = p.vol_target is not None

    # 1-2. Score and vol forecast.
    scored = score_cache.get(p.L) if score_cache is not None else None
    if scored is None:
        scored = _score(prices, rf, returns, p)
        if score_cache is not None:
            score_cache[p.L] = scored
    score, beta, sigma = scored.score, scored.beta, scored.sigma

    # 3. Signal. For trend the sampled m becomes the reported score. The long-only switch
    # clips the signal last (§5.9); the score is unchanged.
    if p.strategy == "trend":
        score, signal = strategies.trend_signal(score, p.signal_frequency)
    else:
        signal = strategies.hysteresis_signal(score, p.z_entry, p.z_exit)
    if not p.allow_short:
        signal = strategies.long_only(signal)

    # 4. Unit weights, with tickers as column labels.
    beta_used = None
    if p.strategy == "pairs":
        beta_used = strategies.hedge_ratio_used(beta, signal, p.hedge_mode)
        unit_weights = strategies.pair_unit_weights(signal, beta_used, (legs[0], legs[1]))
    else:
        unit_weights = strategies.single_unit_weights(signal, legs[0])

    # 5. Sizing.
    scale_used = None
    W = unit_weights
    if overlay:
        sized = sizing.size_weights(
            unit_weights,
            signal,
            sigma,
            vol_target=p.vol_target,
            max_leverage=p.max_leverage,
            vol_refresh=p.vol_refresh,
        )
        W, scale_used = sized.weights, sized.scale_used

    # 6. Evaluation: t0, the split, then the backtester.
    valid = score.notna().to_numpy()
    if not valid.any():
        raise NoValidScoreError(
            f"The {p.strategy} score is undefined on every day: the date range is too short for "
            f"L = {p.L}{' and the volatility half-life' if overlay else ''}, or every window is degenerate."
        )
    t0 = int(np.argmax(valid))
    bt = _backtest(p, returns, W, signal, rf, t0, trades=trades)
    return StrategyRun(
        params=p,
        prices=prices,
        rf=rf,
        returns=returns,
        score=score.rename("score"),
        signal=signal,
        unit_weights=unit_weights,
        weights=W,
        beta=beta,
        beta_used=beta_used,
        vol_forecast=sigma,
        scale_used=scale_used,
        t0=t0,
        split_date=bt.split_date,
        backtest=bt,
    )


def _score(prices: pd.DataFrame, rf: pd.Series, returns: pd.DataFrame, p: StrategyParams) -> Scored:
    """Steps 1 and 2 of §5.0 for strategies A, B and C."""
    legs = list(p.tickers)

    # 1. Raw score: the rolling-OLS (§5.3) or Kalman (§5.7) z and beta for pairs.
    beta = None
    if p.strategy == "mean_reversion":
        score = strategies.mean_reversion_score(prices[legs[0]], p.L)
    elif p.strategy == "pairs":
        if p.pairs_model == "kalman":
            pair = strategies.kalman_pairs_score(prices[legs[0]], prices[legs[1]], p.L, p.kalman_delta)
        else:
            pair = strategies.pairs_score(prices[legs[0]], prices[legs[1]], p.L)
        score, beta = pair.z, pair.beta
    else:
        score = strategies.trend_score(prices[legs[0]], rf, p.L)

    # 2. Vol forecast, then mask the score where the forecast is undefined.
    sigma = None
    if p.vol_target is not None:
        if p.strategy == "pairs":
            u = sizing.pair_unit_return(returns[legs[0]], returns[legs[1]], beta)
        else:
            u = returns[legs[0]]
        sigma, score = sizing.forecast_and_mask(u, score, p.vol_halflife)
    return Scored(score=score, beta=beta, sigma=sigma)


def _run_trend_portfolio(
    prices: pd.DataFrame, rf: pd.Series, returns: pd.DataFrame, p: StrategyParams, *, trades: bool
) -> StrategyRun:
    """Strategy D (§5.8): the §5.0 steps per asset, then one backtest with per-leg trades (§7.4).

    Per leg i: m_i (§5.5); sigma_i from the leg's own return, masking m_i where it is undefined
    (§5.6); sampling and sign give s_i, clipped by the long-only switch (§5.9). t0 is the first
    date on which every leg's score is valid, and every s_i is 0 before it. The unit weight is
    s_i / N and the sized weight w_i = s_i * scale_used_i / N, where scale_used_i follows §5.6's
    refresh rule with the leg's own signal; it is exactly 0 where s_i = 0. Each leg is at most
    max_leverage / N, so the gross exposure never exceeds max_leverage.
    """
    legs = list(p.tickers)
    n_legs = len(legs)
    scores, signals, sigmas = {}, {}, {}
    for leg in legs:
        m = strategies.trend_score(prices[leg], rf, p.L)  # 1. raw score
        sigma, m = sizing.forecast_and_mask(returns[leg], m, p.vol_halflife)  # 2. forecast, mask
        score, signal = strategies.trend_signal(m, p.signal_frequency)  # 3. sampling and sign
        if not p.allow_short:
            signal = strategies.long_only(signal)
        scores[leg], signals[leg], sigmas[leg] = score, signal, sigma
    leg_scores = pd.DataFrame(scores, index=prices.index)
    valid = leg_scores.notna().all(axis=1).to_numpy()
    if not valid.any():
        raise NoValidScoreError(
            f"The trend_portfolio score is undefined on every day: the date range is too short for "
            f"L = {p.L} and the volatility half-life."
        )
    t0 = int(np.argmax(valid))  # the first date on which every leg's score is valid
    leg_signals = pd.DataFrame(signals, index=prices.index)
    leg_signals.iloc[:t0] = 0.0  # the portfolio starts with all its legs

    # 4-5. Unit weights s_i / N; sized weights s_i * scale_used_i / N, each leg with its own refresh.
    unit_weights = leg_signals / n_legs
    vol_forecasts = pd.DataFrame(sigmas, index=prices.index)
    W, scale_used = sizing.size_portfolio(
        leg_signals,
        vol_forecasts,
        vol_target=p.vol_target,
        max_leverage=p.max_leverage,
        vol_refresh=p.vol_refresh,
    )
    invested = leg_signals.ne(0).any(axis=1).astype(float).rename("signal")

    # 6. Evaluation.
    bt = _backtest(p, returns, W, invested, rf, t0, leg_signals=leg_signals, trades=trades)
    return StrategyRun(
        params=p,
        prices=prices,
        rf=rf,
        returns=returns,
        score=None,
        signal=invested,
        unit_weights=unit_weights,
        weights=W,
        beta=None,
        beta_used=None,
        vol_forecast=None,
        scale_used=None,
        t0=t0,
        split_date=bt.split_date,
        backtest=bt,
        leg_scores=leg_scores,
        leg_signals=leg_signals,
        leg_vol_forecasts=vol_forecasts,
        leg_scale_used=scale_used,
    )


def _backtest(
    p: StrategyParams,
    returns: pd.DataFrame,
    W: pd.DataFrame,
    signal: pd.Series,
    rf: pd.Series,
    t0: int,
    *,
    leg_signals: pd.DataFrame | None = None,
    trades: bool,
) -> BacktestResult:
    """Step 6 of §5.0: resolve the split on the days from e = t0 + k, then run the backtester."""
    k = p.execution_lag
    split = backtester.resolve_split(returns.index[t0 + k :], p.split_date)
    return backtester.run_backtest(
        R=returns[list(p.tickers)],
        W=W,
        s=signal,
        rf=rf,
        t0=t0,
        k=k,
        cost_bps=p.cost_bps,
        borrow_bps=p.borrow_bps,
        E0=p.initial_capital,
        split_date=split,
        leg_signals=leg_signals,
        trades=trades,
    )


def _check_inputs(prices: pd.DataFrame, rf: pd.Series, params: StrategyParams) -> list[str]:
    """Assert the §2.1 index contract where data enters the core; return the columns used."""
    if not isinstance(rf, pd.Series):
        raise TypeError(f"rf must be a pandas Series on the run calendar (§2.3), not {type(rf).__name__}")
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"prices must be a pandas DataFrame, not {type(prices).__name__}")
    idx = prices.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is not None or idx.unit != "ns":
        raise ValueError("prices must be indexed by a tz-naive datetime64[ns] DatetimeIndex")
    if not (idx.is_monotonic_increasing and idx.is_unique):
        raise ValueError("the price index must be sorted ascending and unique")
    cols = list(dict.fromkeys([*params.tickers, MARKET]))
    missing = [c for c in cols if c not in prices.columns]
    if missing:
        raise ValueError(f"prices has no column for {missing}")
    values = prices[cols].to_numpy(dtype=float)
    if not (np.isfinite(values).all() and (values > 0).all()):
        raise ValueError("prices must be finite and > 0 (clean and inner-join them first)")
    if not rf.index.equals(idx) or not np.isfinite(rf.to_numpy(dtype=float)).all():
        raise ValueError("rf must be finite and indexed by the same calendar as prices")
    return cols
