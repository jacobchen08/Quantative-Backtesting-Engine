"""to_jsonable() and the response payloads built from engine results (§2.4, §11.3).

Starlette answers HTTP 500 if a response contains NaN or inf, so every payload goes through
to_jsonable(): non-finite floats become None, numpy values become Python values, and dates
become "YYYY-MM-DD".
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import Any

import numpy as np
import pandas as pd

from app.services.monte_carlo import ModelResult, MonteCarloResult
from app.services.pipeline import BacktestReport, SensitivityResult, WalkForwardResult


def to_jsonable(obj: Any) -> Any:
    """Convert engine output to JSON-safe Python: None for NaN/inf/NaT, plain numbers, ISO dates."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if obj is pd.NaT or (isinstance(obj, np.datetime64) and np.isnat(obj)):
        return None
    if isinstance(obj, (dt.date, np.datetime64)):  # dt.datetime and pd.Timestamp are dt.date subclasses
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, (pd.Series, pd.Index)):
        return to_jsonable(obj.to_numpy())
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "M":
            return [None if pd.isna(day) else day.strftime("%Y-%m-%d") for day in pd.DatetimeIndex(obj)]
        if obj.dtype.kind == "f":
            return _finite(obj.tolist())
        return to_jsonable(obj.tolist())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {field.name: to_jsonable(getattr(obj, field.name)) for field in dataclasses.fields(obj)}
    raise TypeError(f"cannot convert {type(obj).__name__} to JSON")


def _finite(values: Any) -> Any:
    """Float lists from ndarray.tolist(), with NaN and inf replaced by None (any nesting)."""
    if isinstance(values, list):
        return [_finite(value) for value in values]
    return values if math.isfinite(values) else None


def backtest_payload(
    report: BacktestReport, resolved: dict[str, Any], data_warnings: list[str]
) -> dict[str, Any]:
    """BacktestResponse without run_id and engine_version (§11.3)."""
    s = report.series
    d = report.diagnostics
    series = {
        "dates": s.index,
        **{name: s[name] for name in ("equity", "benchmark_equity", "drawdown", "daily_return")},
        **{name: s[name] for name in ("position", "gross_exposure")},
        **{name: s[name] if name in s else None for name in ("score", "beta", "beta_used", "vol_forecast")},
        "is_oos": s["is_oos"],
        "rolling_sharpe": s["rolling_sharpe"],
        "rolling_beta": s["rolling_beta"],
        "legs": None
        if report.legs is None
        else [
            {
                "ticker": ticker,
                **{name: leg[name] for name in ("score", "position", "weight", "vol_forecast")},
            }
            for ticker, leg in report.legs.items()
        ],
    }
    diagnostics = None
    if d is not None:
        diagnostics = {
            "coint_p_is": d.coint_p_is,
            "coint_p_oos": d.coint_p_oos,
            "half_life_is": d.half_life_is,
            "static_beta_is": d.static_beta_is,
            "static_alpha_is": d.static_alpha_is,
        }
    return to_jsonable(
        {
            "resolved": {
                **resolved,
                "split_date": report.split_date,
                "first_evaluated_date": report.first_evaluated_date,
            },
            "series": series,
            "sizing": report.sizing,
            "trades": report.trades.to_dict("records"),
            "metrics": report.metrics,
            "benchmark_metrics": report.benchmark_metrics,
            "diagnostics": diagnostics,
            "cost_sensitivity": report.cost_sensitivity,
            "calendar": {
                "monthly": [
                    {
                        "year": r.year,
                        "month": r.month,
                        "strategy": r.strategy,
                        "benchmark": r.benchmark,
                        "partial": r.partial,
                    }
                    for r in report.calendar_monthly
                ],
                "annual": [
                    {"year": r.year, "strategy": r.strategy, "benchmark": r.benchmark, "partial": r.partial}
                    for r in report.calendar_annual
                ],
            },
            "ruined": report.ruined,
            "ruin_date": report.ruin_date,
            "warnings": [*data_warnings, *report.warnings],
        }
    )


def sensitivity_payload(result: SensitivityResult) -> dict[str, Any]:
    """SensitivityResponse (§11.3). window.end is the split date, which the window excludes."""
    return to_jsonable(
        {
            "L_values": result.L_values,
            "z_entry_values": result.z_entry_values,
            "sharpe": result.sharpe,
            "n_configs_tested": result.n_configs_tested,
            "window": {"start": result.window_start, "end": result.window_end},
            "selected": _cell(result.selected),
            "deflated": result.deflated,
        }
    )


def _cell(cell: tuple[int, float | None] | None) -> dict[str, Any] | None:
    return None if cell is None else {"L": cell[0], "z_entry": cell[1]}


def walk_forward_payload(
    result: WalkForwardResult, resolved: dict[str, Any], data_warnings: list[str]
) -> dict[str, Any]:
    """WalkForwardResponse (§11.3). The stitched daily components stay server-side."""
    s = result.series
    return to_jsonable(
        {
            "resolved": {
                **resolved,
                "L_values": result.L_values,
                "z_entry_values": result.z_entry_values,
            },
            "folds": [
                {
                    "train_start": f.train_start,
                    "train_end": f.train_end,
                    "test_start": f.test_start,
                    "test_end": f.test_end,
                    "selected": _cell(f.selected),
                    "train_sharpe": f.train_sharpe,
                    "test_sharpe": f.test_sharpe,
                }
                for f in result.folds
            ],
            "n_configs": result.n_configs,
            "n_distinct_selected": result.n_distinct_selected,
            "series": {
                "dates": s.index,
                "equity": s["equity"],
                "fixed_equity": s["fixed_equity"],
                "benchmark_equity": s["benchmark_equity"],
                "fold": [None if np.isnan(j) else int(j) for j in s["fold"].to_numpy()],
            },
            "metrics": result.metrics,
            "fixed_metrics": result.fixed_metrics,
            "benchmark_metrics": result.benchmark_metrics,
            "ruined": result.ruined,
            "ruin_date": result.ruin_date,
            "warnings": [*data_warnings, *result.warnings],
        }
    )


def _model_payload(model: ModelResult) -> dict[str, Any]:
    return {
        "bands": model.bands,
        "sample_paths": model.sample_paths,
        "hist_counts": model.hist_counts,
        "var95": model.var95,
        "var99": model.var99,
        "cvar95": model.cvar95,
        "cvar99": model.cvar99,
        "elapsed_ms": model.elapsed_ms,
    }


def monte_carlo_payload(result: MonteCarloResult, resolved: dict[str, Any]) -> dict[str, Any]:
    """MonteCarloResponse (§11.3). The raw terminal returns stay server-side."""
    return to_jsonable(
        {
            "resolved": {
                **resolved,
                "seed": result.seed,
                "n_hist": result.n_hist,
                "mu_daily": result.mu_daily,
                "sigma_daily": result.sigma_daily,
                "block_len_used": result.block_len_used,
            },
            "gbm": _model_payload(result.gbm),
            "bootstrap": _model_payload(result.bootstrap),
            "stationary": _model_payload(result.stationary),
            "closed_form_gbm": result.closed_form_gbm,
            "hist_edges": result.hist_edges,
            "elapsed_ms_total": result.elapsed_ms_total,
        }
    )
