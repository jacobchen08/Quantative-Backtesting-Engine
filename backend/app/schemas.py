"""Pydantic v2 request and response models (§11).

Fields that don't apply to the chosen mode are ignored, never rejected (§11.2): a "before"
validator drops them from the input, so none of their rules run, and resolved() echoes them
as null. Response models mirror §11.3, which frontend/src/types.ts mirrors in turn.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from app.services import calendar
from app.services.pipeline import StrategyParams

TICKER = re.compile(r"^[A-Z0-9.^=-]{1,15}$")
EARLIEST_START = dt.date(1990, 1, 1)
L_RANGE = {"mean_reversion": (10, 252), "pairs": (10, 252), "trend": (21, 504), "trend_portfolio": (21, 504)}
L_DEFAULT = {"mean_reversion": 60, "pairs": 60, "trend": 252, "trend_portfolio": 252}
PORTFOLIO_LEGS = (2, 12)  # distinct tickers in a trend portfolio (§5.8)
VOL_TARGET_RANGE = (0.01, 0.5)  # annualized, for the optional overlay (§5.6)
PORTFOLIO_VOL_TARGET_RANGE = (0.05, 0.6)  # per asset, for the trend portfolio (§5.8)
PORTFOLIO_VOL_TARGET_DEFAULT = 0.40
MAX_GRID_VALUES = 8
MAX_PATH_CELLS = 12_700_000  # n_paths * (horizon + 1); keeps Monte Carlo near 200 MB
MAX_SEED = 2**53  # JavaScript loses integer precision above this

Strategy = Literal["mean_reversion", "pairs", "trend", "trend_portfolio"]
RiskFree = Literal["tbill"] | Annotated[float, Field(ge=-0.01, le=0.25)]
BlockLen = Literal["auto"] | Annotated[int, Field(ge=1, le=63)]


def _not_applicable(strategy: object, vol_target: object, pairs_model: object) -> set[str]:
    """BacktestRequest fields that the strategy, the pairs model or a missing vol_target makes
    irrelevant. The trend portfolio's vol settings always apply: its vol_target resolves to 0.40."""
    fields: set[str] = set()
    if strategy in ("trend", "trend_portfolio"):
        fields |= {"z_entry", "z_exit", "hedge_mode"}
    if strategy in ("mean_reversion", "pairs"):
        fields.add("signal_frequency")
    if strategy == "mean_reversion":
        fields.add("hedge_mode")
    if strategy != "pairs":
        fields |= {"pairs_model", "kalman_delta"}
    elif pairs_model != "kalman":
        fields.add("kalman_delta")
    if vol_target is None and strategy != "trend_portfolio":
        fields |= {"vol_halflife", "vol_refresh", "max_leverage"}
    return fields


def _check_dates(start: dt.date, end: dt.date) -> dt.date:
    """start >= 1990-01-01; returns end clamped to today_ny() - 1 day, which must stay after start."""
    if start < EARLIEST_START:
        raise ValueError(f"start must be on or after {EARLIEST_START}")
    end = calendar.last_final_day(end)
    if not start < end:
        raise ValueError(
            f"start ({start}) must be before end ({end}, after clamping end to yesterday in New York)"
        )
    return end


def _check_ticker(ticker: str) -> None:
    if not TICKER.match(ticker):
        raise ValueError(f"invalid ticker {ticker!r}: use 1-15 characters from A-Z, 0-9 and . ^ = -")


# ---------------------------------------------------------------- requests (§11.2)


class BacktestRequest(BaseModel):
    """POST /api/backtest. Dates are calendar dates; rates and targets are decimals."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {"strategy": "pairs", "tickers": ["KO", "PEP"], "start": "2015-01-02", "end": "2025-12-31"}
            ]
        },
    )

    strategy: Strategy
    tickers: list[str]
    start: dt.date
    end: dt.date
    L: int | None = Field(
        None,
        description="Lookback in trading days; null -> 60 (mean_reversion, pairs) "
        "or 252 (trend, trend_portfolio)",
    )
    z_entry: float = Field(2.0, gt=0, le=5)
    z_exit: float = Field(0.5, ge=0)
    signal_frequency: Literal["monthly", "daily"] = "monthly"
    cost_bps: float = Field(10.0, ge=0, le=100, description="One-way cost per leg, basis points")
    execution_lag: int = Field(1, ge=0, le=5)
    hedge_mode: Literal["fixed_at_entry", "rolling"] = "fixed_at_entry"
    pairs_model: Literal["ols", "kalman"] = Field(
        "ols", description="Pairs only: rolling OLS or Kalman filter"
    )
    kalman_delta: float = Field(
        1e-6, ge=1e-7, le=1e-2, description="Kalman state noise; pairs_model='kalman' only"
    )
    borrow_bps: float = Field(0.0, ge=0, le=2000, description="Annual fee on short notional, basis points")
    allow_short: bool = Field(True, description="false clips the signal at 0 (long or flat); not for pairs")
    vol_target: float | None = Field(
        None,
        description="Annualized; null turns the overlay off. 0.01-0.5, or 0.05-0.6 per asset for "
        "trend_portfolio, where null -> 0.40",
    )
    vol_halflife: int = Field(42, ge=5, le=126)
    vol_refresh: Literal["daily", "weekly", "monthly"] = "weekly"
    max_leverage: float = Field(2.0, ge=1, le=4)
    split_date: dt.date | None = None
    risk_free: RiskFree = "tbill"
    initial_capital: float = Field(100_000.0, gt=0)

    @classmethod
    def _ignored(cls, data: dict[str, Any]) -> set[str]:
        return _not_applicable(data.get("strategy"), data.get("vol_target"), data.get("pairs_model", "ols"))

    @model_validator(mode="before")
    @classmethod
    def _drop_ignored(cls, data: Any) -> Any:
        if isinstance(data, dict):
            ignored = cls._ignored(data)
            return {key: value for key, value in data.items() if key not in ignored}
        return data

    @field_validator("tickers", mode="before")
    @classmethod
    def _uppercase(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [t.upper() if isinstance(t, str) else t for t in value]
        return value

    @model_validator(mode="after")
    def _resolve(self) -> Self:
        self._resolve_common()
        return self

    def _resolve_common(self) -> None:
        if self.strategy == "trend_portfolio":
            lo, hi = PORTFOLIO_LEGS
            if not lo <= len(self.tickers) <= hi:
                raise ValueError(f"trend_portfolio needs {lo} to {hi} tickers, got {len(self.tickers)}")
        else:
            n_legs = 2 if self.strategy == "pairs" else 1
            if len(self.tickers) != n_legs:
                raise ValueError(f"{self.strategy} needs exactly {n_legs} ticker(s), got {len(self.tickers)}")
        for ticker in self.tickers:
            _check_ticker(ticker)
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError(
                "pairs needs two distinct tickers"
                if self.strategy == "pairs"
                else f"{self.strategy} needs distinct tickers"
            )
        if self.strategy == "trend_portfolio" and self.vol_target is None:
            self.vol_target = PORTFOLIO_VOL_TARGET_DEFAULT  # the overlay is always on (§5.8)
        if self.vol_target is not None:
            portfolio = self.strategy == "trend_portfolio"
            lo, hi = PORTFOLIO_VOL_TARGET_RANGE if portfolio else VOL_TARGET_RANGE
            if not lo <= self.vol_target <= hi:
                where = " per asset for trend_portfolio" if portfolio else ""
                raise ValueError(f"vol_target must be between {lo} and {hi}{where}, got {self.vol_target}")
        self.end = _check_dates(self.start, self.end)
        lo, hi = L_RANGE[self.strategy]
        if self.L is None:
            self.L = L_DEFAULT[self.strategy]
        elif not lo <= self.L <= hi:
            raise ValueError(f"L must be between {lo} and {hi} for {self.strategy}, got {self.L}")
        if self.strategy in ("mean_reversion", "pairs") and not self.z_exit < self.z_entry:
            raise ValueError(f"z_exit ({self.z_exit}) must be less than z_entry ({self.z_entry})")
        if self.strategy == "pairs" and not self.allow_short:
            raise ValueError("pairs needs a short leg, so allow_short=false is not available")

    def resolved(self) -> dict[str, Any]:
        """The request with every default filled in, end clamped and non-applicable fields null."""
        out = self.model_dump(mode="json", include=set(BacktestRequest.model_fields))
        for key in _not_applicable(self.strategy, self.vol_target, self.pairs_model):
            out[key] = None
        return out

    def rf_source(self) -> str:
        return (
            "13-week T-bill (^IRX)" if self.risk_free == "tbill" else f"constant {self.risk_free:.2%} a year"
        )

    def to_params(self) -> StrategyParams:
        return StrategyParams(
            strategy=self.strategy,
            tickers=tuple(self.tickers),
            L=self.L,
            z_entry=self.z_entry,
            z_exit=self.z_exit,
            signal_frequency=self.signal_frequency,
            hedge_mode=self.hedge_mode,
            pairs_model=self.pairs_model,
            kalman_delta=self.kalman_delta,
            cost_bps=self.cost_bps,
            execution_lag=self.execution_lag,
            borrow_bps=self.borrow_bps,
            allow_short=self.allow_short,
            vol_target=self.vol_target,
            vol_halflife=self.vol_halflife,
            vol_refresh=self.vol_refresh,
            max_leverage=self.max_leverage,
            split_date=self.split_date,
            initial_capital=self.initial_capital,
        )


class SensitivityRequest(BacktestRequest):
    """POST /api/sensitivity: a backtest request plus optional grid overrides (§9)."""

    L_values: list[int] | None = Field(None, min_length=1, max_length=MAX_GRID_VALUES)
    z_entry_values: list[float] | None = Field(None, min_length=1, max_length=MAX_GRID_VALUES)

    @classmethod
    def _ignored(cls, data: dict[str, Any]) -> set[str]:
        ignored = super()._ignored(data)
        if data.get("strategy") in ("trend", "trend_portfolio"):
            ignored.add("z_entry_values")  # no z_entry axis
        return ignored

    @model_validator(mode="after")
    def _resolve(self) -> Self:  # replaces BacktestRequest._resolve
        self._resolve_common()
        lo, hi = L_RANGE[self.strategy]
        if self.L_values is not None and not all(lo <= L <= hi for L in self.L_values):
            raise ValueError(f"every L in L_values must be between {lo} and {hi} for {self.strategy}")
        if self.z_entry_values is not None and not all(self.z_exit < z <= 5 for z in self.z_entry_values):
            raise ValueError(f"every z_entry in z_entry_values must be in ({self.z_exit}, 5]")
        return self


class WalkForwardRequest(SensitivityRequest):
    """POST /api/walk-forward: a sensitivity request plus the fold windows (§9.2)."""

    train_days: int = Field(756, ge=252, le=2520, description="Training window, trading days")
    test_days: int = Field(63, ge=21, le=252, description="Test window, trading days")
    anchored: bool = Field(False, description="true: every training window starts at the grid's common start")

    def resolved(self) -> dict[str, Any]:
        """The backtest fields as BacktestRequest.resolved() echoes them, plus the fold windows.

        The route adds the grid actually used (L_values, z_entry_values).
        """
        return {
            **super().resolved(),
            "train_days": self.train_days,
            "test_days": self.test_days,
            "anchored": self.anchored,
        }


class MonteCarloRequest(BaseModel):
    """POST /api/monte-carlo. The history is a strategy's returns or one ticker's returns (§10)."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [{"source": "ticker", "ticker": "SPY", "start": "2015-01-02", "end": "2025-12-31"}]
        },
    )

    source: Literal["strategy", "ticker"]
    backtest: BacktestRequest | None = Field(None, description="Required iff source is 'strategy'")
    window: Literal["full", "is"] = "full"
    ticker: str | None = Field(None, description="Required iff source is 'ticker'")
    start: dt.date | None = None
    end: dt.date | None = None
    n_paths: int = Field(10_000, ge=1_000, le=50_000)
    horizon: int = Field(252, ge=5, le=504, description="Trading days")
    block_len: BlockLen = Field(
        10, description="Days, 1-63, or 'auto' (§10.8); the mean block length for the stationary bootstrap"
    )
    seed: int | None = Field(None, ge=0, lt=MAX_SEED)
    zero_drift: bool = False

    @model_validator(mode="before")
    @classmethod
    def _drop_ignored(cls, data: Any) -> Any:
        if isinstance(data, dict):
            source = data.get("source")
            ignored = {"backtest", "window"} if source == "ticker" else set()
            if source == "strategy":
                ignored = {"ticker", "start", "end"}
            return {key: value for key, value in data.items() if key not in ignored}
        return data

    @field_validator("ticker", mode="before")
    @classmethod
    def _uppercase(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _resolve(self) -> Self:
        if self.n_paths * (self.horizon + 1) > MAX_PATH_CELLS:
            raise ValueError(
                f"n_paths * (horizon + 1) = {self.n_paths * (self.horizon + 1):,} exceeds {MAX_PATH_CELLS:,}"
            )
        if self.source == "strategy":
            if self.backtest is None:
                raise ValueError("backtest is required when source is 'strategy'")
            return self
        missing = [name for name in ("ticker", "start", "end") if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{', '.join(missing)} required when source is 'ticker'")
        _check_ticker(self.ticker)
        self.end = _check_dates(self.start, self.end)
        return self

    def resolved(self) -> dict[str, Any]:
        out = self.model_dump(mode="json", exclude={"backtest"})
        out["backtest"] = None if self.backtest is None else self.backtest.resolved()
        if self.source == "ticker":
            out["window"] = None
        else:
            out["ticker"] = out["start"] = out["end"] = None
        return out


# ---------------------------------------------------------------- responses (§11.3)


class MetricsOut(BaseModel):
    n_days: int
    total_return: float
    cagr: float | None
    ann_vol: float | None  # null when n_days < 2
    sharpe: float | None
    sharpe_ci95: tuple[float, float] | None
    sortino: float | None
    max_drawdown: float
    beta_spy: float | None
    corr_spy: float | None
    pct_days_in_market: float
    n_closed_trades: int | None  # trade and cost fields are null for benchmarks
    n_open_trades: int | None
    win_rate: float | None
    avg_holding_days: float | None
    ann_turnover: float | None
    ann_cost_drag: float | None
    ann_borrow_drag: float | None
    avg_gross_exposure: float  # 1 for benchmarks


class SegmentMetricsOut(BaseModel):
    is_: MetricsOut = Field(alias="is")
    oos: MetricsOut | None  # null only if ruin happened in IS
    full: MetricsOut


class TradeOut(BaseModel):
    id: int
    side: Literal["long", "short"]
    segment: Literal["is", "oos"]
    open: bool
    signal_date: str
    entry_date: str
    exit_signal_date: str | None
    exit_date: str | None
    holding_days: int
    pnl: float
    beta_at_entry: float | None
    ticker: str | None  # the leg, for trend_portfolio only (§7.4)


class LegSeriesOut(BaseModel):
    """One trend-portfolio leg on the series dates: score and vol forecast as of each close,
    executed position and weight."""

    ticker: str
    score: list[float | None]
    position: list[float]
    weight: list[float]
    vol_forecast: list[float | None]


class SeriesOut(BaseModel):
    """Aligned arrays from s0 = min(t0, e - 1) to the end (or the ruin day)."""

    dates: list[str]
    equity: list[float | None]  # null before the baseline day e - 1
    benchmark_equity: list[float | None]
    drawdown: list[float | None]
    daily_return: list[float | None]  # daily total return; null up to and including the baseline
    position: list[float]  # executed position (trend_portfolio: 1 when any leg is held)
    gross_exposure: list[float]  # executed sum of |w| at the end of each day
    score: list[float | None] | None  # null for trend_portfolio, whose scores are in legs
    beta: list[float | None] | None  # pairs only (the filtered beta with pairs_model="kalman")
    beta_used: list[float | None] | None  # pairs only
    vol_forecast: list[float | None] | None  # null when the overlay is off, and for trend_portfolio
    is_oos: list[bool]
    rolling_sharpe: list[float | None]  # trailing 252 evaluated days (§8.2)
    rolling_beta: list[float | None]
    legs: list[LegSeriesOut] | None  # trend_portfolio only


class SizingOut(BaseModel):
    vol_target: float
    max_leverage: float
    pct_days_at_cap: float | None


class DiagnosticsOut(BaseModel):
    coint_p_is: float | None
    coint_p_oos: float | None
    half_life_is: float | None
    static_beta_is: float | None
    static_alpha_is: float | None


class ResolvedBacktestRequest(BaseModel):
    strategy: Strategy
    tickers: list[str]
    start: str
    end: str
    L: int
    z_entry: float | None
    z_exit: float | None
    signal_frequency: Literal["monthly", "daily"] | None
    cost_bps: float
    execution_lag: int
    hedge_mode: Literal["fixed_at_entry", "rolling"] | None
    pairs_model: Literal["ols", "kalman"] | None
    kalman_delta: float | None
    borrow_bps: float
    allow_short: bool
    vol_target: float | None
    vol_halflife: int | None
    vol_refresh: Literal["daily", "weekly", "monthly"] | None
    max_leverage: float | None
    split_date: str | None
    risk_free: Literal["tbill"] | float
    initial_capital: float


class ResolvedBacktestRun(ResolvedBacktestRequest):
    split_date: str
    first_evaluated_date: str
    rf_source: str


class SegmentSharpeListOut(BaseModel):
    is_: list[float | None] = Field(alias="is")
    oos: list[float | None] | None
    full: list[float | None]


class SegmentValueOut(BaseModel):
    is_: float | None = Field(alias="is")
    oos: float | None
    full: float | None


class CostSensitivityOut(BaseModel):
    """§8.1: Sharpe against one-way cost from one run with positions held fixed."""

    cost_bps: list[float]
    sharpe: SegmentSharpeListOut  # one value per cost_bps entry
    break_even_bps: SegmentValueOut  # null: no turnover, or losing before costs


class MonthlyReturnOut(BaseModel):
    year: int
    month: int
    strategy: float
    benchmark: float
    partial: bool


class AnnualReturnOut(BaseModel):
    year: int
    strategy: float
    benchmark: float
    partial: bool


class CalendarOut(BaseModel):
    monthly: list[MonthlyReturnOut]
    annual: list[AnnualReturnOut]


class BacktestResponse(BaseModel):
    run_id: int
    engine_version: str
    resolved: ResolvedBacktestRun
    series: SeriesOut
    sizing: SizingOut | None  # null when the overlay is off
    trades: list[TradeOut]
    metrics: SegmentMetricsOut
    benchmark_metrics: SegmentMetricsOut
    diagnostics: DiagnosticsOut | None  # pairs only
    cost_sensitivity: CostSensitivityOut | None  # null if ruined
    calendar: CalendarOut
    ruined: bool
    ruin_date: str | None
    warnings: list[str]


class WindowOut(BaseModel):
    start: str
    end: str  # the split date, excluded from the window


class SelectedCellOut(BaseModel):
    L: int
    z_entry: float | None


class BestCellOut(BaseModel):
    L: int
    z_entry: float | None
    sharpe: float
    dsr: float | None


class DeflatedOut(BaseModel):
    """§9.1 deflated Sharpe ratios on the common window."""

    n_trials: int
    sr0_star: float | None  # annualized for display
    selected_dsr: float | None
    best: BestCellOut | None


class SensitivityResponse(BaseModel):
    L_values: list[int]
    z_entry_values: list[float] | None  # null for trend and trend_portfolio
    sharpe: list[list[float | None]]  # rows = L; one column for trend and trend_portfolio
    n_configs_tested: int
    window: WindowOut
    selected: SelectedCellOut | None
    deflated: DeflatedOut


class ResolvedWalkForwardRun(ResolvedBacktestRequest):
    train_days: int
    test_days: int
    anchored: bool
    L_values: list[int]  # the grid actually used
    z_entry_values: list[float] | None  # null for trend and trend_portfolio


class FoldOut(BaseModel):
    """One walk-forward fold (§9.2); dates are inclusive."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected: SelectedCellOut | None  # null = flat fold: no configuration was eligible
    train_sharpe: float | None
    test_sharpe: float | None


class WalkForwardSeriesOut(BaseModel):
    """Aligned arrays from the baseline day b0 - 2 to the last stitched day."""

    dates: list[str]
    equity: list[float | None]
    fixed_equity: list[float | None]  # null after the request's own configuration was ruined
    benchmark_equity: list[float | None]
    fold: list[int | None]  # the fold that owns each day; null on the baseline


class WalkForwardResponse(BaseModel):
    resolved: ResolvedWalkForwardRun
    folds: list[FoldOut]
    n_configs: int
    n_distinct_selected: int
    series: WalkForwardSeriesOut
    metrics: MetricsOut  # the stitched series: the trade fields are null
    fixed_metrics: MetricsOut | None  # null if the request's own configuration was ruined before the window
    benchmark_metrics: MetricsOut
    ruined: bool
    ruin_date: str | None
    warnings: list[str]


class BandsOut(BaseModel):
    p5: list[float]
    p25: list[float]
    p50: list[float]
    p75: list[float]
    p95: list[float]


class ModelResultOut(BaseModel):
    bands: BandsOut  # equity multiples for t = 0..T
    sample_paths: list[list[float]]  # 20 x (T + 1)
    hist_counts: list[int]  # 50
    var95: float
    var99: float
    cvar95: float
    cvar99: float
    elapsed_ms: float


class RiskOut(BaseModel):
    var95: float
    var99: float
    cvar95: float
    cvar99: float


class BlockLenUsedOut(BaseModel):
    """The block lengths actually used: the request's block_len, or the §10.8 estimates."""

    circular: int
    stationary: int  # the mean block length


class ResolvedMonteCarloRun(BaseModel):
    source: Literal["strategy", "ticker"]
    backtest: ResolvedBacktestRequest | None
    window: Literal["full", "is"] | None
    ticker: str | None
    start: str | None
    end: str | None
    n_paths: int
    horizon: int
    block_len: int | Literal["auto"]  # as requested
    seed: int
    zero_drift: bool
    n_hist: int
    mu_daily: float
    sigma_daily: float
    block_len_used: BlockLenUsedOut


class MonteCarloResponse(BaseModel):
    resolved: ResolvedMonteCarloRun
    gbm: ModelResultOut
    bootstrap: ModelResultOut
    stationary: ModelResultOut  # §10.7
    closed_form_gbm: RiskOut
    hist_edges: list[float]  # 51, shared by the three models
    elapsed_ms_total: float


# ---------------------------------------------------------------- run history (§11.4)


def _nullable_copy(model: type[BaseModel], name: str) -> type[BaseModel]:
    """The same fields as `model`, each nullable and defaulting to null.

    A run stored by an earlier engine version can lack fields added later (ann_borrow_drag,
    pairs_model, ...). Those read back as null; stored rows are never rewritten (§11.4).
    """
    fields: dict[str, Any] = {
        key: (info.annotation | None, Field(default=None, alias=info.alias))
        for key, info in model.model_fields.items()
    }
    return create_model(name, **fields)


StoredMetrics = _nullable_copy(MetricsOut, "StoredMetrics")
StoredResolvedRun = _nullable_copy(ResolvedBacktestRun, "StoredResolvedRun")


class StoredSegmentMetrics(BaseModel):
    is_: StoredMetrics | None = Field(default=None, alias="is")
    oos: StoredMetrics | None = None
    full: StoredMetrics | None = None


class RunSummary(BaseModel):
    """One row of GET /api/runs, read from the stored JSON."""

    id: int
    created_at: str  # ISO 8601, UTC
    engine_version: str
    strategy: Strategy
    tickers: list[str]
    start: str
    end: str
    sharpe_full: float | None
    sharpe_oos: float | None  # null when the run has no OOS segment (ruin in IS)
    total_return_full: float
    max_drawdown_full: float


class RunDetail(BaseModel):
    """GET /api/runs/{id}: the stored resolved request and metrics. Equity curves aren't stored."""

    id: int
    created_at: str  # ISO 8601, UTC
    engine_version: str
    resolved: StoredResolvedRun
    metrics: StoredSegmentMetrics
    benchmark_metrics: StoredSegmentMetrics
