"""Execution lag, daily returns, costs, borrow fees, equity, ruin and trades (§6, §7). Pure (§0, rule 8).

The execution lag k is applied here and nowhere else (§2.2): the executed position is
p_t = s_{t-k}, and day t's P&L comes from the position held into day t, p_{t-1} * r_t.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

MIN_IS_DAYS = 252
MIN_OOS_DAYS = 126


class InsufficientDataError(Exception):
    """Too little data for the requested evaluation. The API maps it to HTTP 422.

    n_is and n_oos carry the evaluated-day counts when the IS/OOS minimums caused the error.
    """

    def __init__(self, message: str, *, n_is: int | None = None, n_oos: int | None = None) -> None:
        super().__init__(message)
        self.n_is = n_is
        self.n_oos = n_oos


class NoValidScoreError(InsufficientDataError):
    """The strategy's score is NaN on every day, so there is no t0 (§6.2)."""


def resolve_split(eval_dates: pd.DatetimeIndex, requested: date | pd.Timestamp | None) -> pd.Timestamp:
    """Resolve the IS/OOS split date and enforce the 252 IS / 126 OOS minimums (§6.3).

    Args:
        eval_dates: the evaluated days, from the first evaluated day e to the end of the
            calendar, untruncated by ruin.
        requested: the requested split date, or None for the default.

    Returns:
        The first OOS date. With no request it is eval_dates[floor(0.7 * n)]; a requested date
        that isn't a trading day rolls forward to the first evaluated day on or after it.
        IS = evaluated days before it, OOS = evaluated days on or after it.

    Raises:
        InsufficientDataError: fewer than 252 IS or 126 OOS evaluated days.
    """
    n = len(eval_dates)
    if requested is None:
        # floor(0.7 n) exactly. The float expression int(0.7 * n) is one too small for
        # n = 90, 170, 180, ... because 0.7 * 90 evaluates to 62.99999999999999.
        pos = (7 * n) // 10
    else:
        pos = int(eval_dates.searchsorted(pd.Timestamp(requested), side="left"))
    n_is, n_oos = pos, n - pos
    if n_is < MIN_IS_DAYS or n_oos < MIN_OOS_DAYS:
        raise InsufficientDataError(
            f"The split leaves {n_is} in-sample and {n_oos} out-of-sample evaluated days; at least "
            f"{MIN_IS_DAYS} and {MIN_OOS_DAYS} are needed. Use a longer date range"
            + (" or a different split date." if requested is not None else "."),
            n_is=n_is,
            n_oos=n_oos,
        )
    return eval_dates[pos]


@dataclass(frozen=True)
class BacktestResult:
    """Output of run_backtest.

    Series are on the full run calendar (the index of R) unless noted. Only positions
    [e, stop) form the evaluation window: e = t0 + k is the first evaluated day and stop is
    one past the last evaluated day (one past the ruin day if ruin happened). Values outside
    the window are intermediate values, not results.

    Returns and P&L are decimals and fractions of equity. The value at date t of every daily
    series is for the day ending at the close of t.

    With per-leg trades (the trend portfolio, §7.4), trade_id and held_id are DataFrames with
    one column per leg, holding the portfolio-wide trade ids, and leg_pos holds each leg's
    executed position. Otherwise they are Series and leg_pos is None.

    trade_id, held_id, pnl and trades are None when run_backtest was called with trades=False.
    """

    w: pd.DataFrame  # executed end-of-day weights, w_t = W_{t-k}
    pos: pd.Series  # executed position, pos_t = s_{t-k} (for per-leg trades: the invested flag)
    held: pd.Series  # True when a position (any leg) is held into day t and earns its P&L
    gross_held: pd.Series  # sum_i |w_{i,t-1}|: gross exposure held into day t
    gross_excess: pd.Series  # sum_i w_{i,t-1} (r_{i,t} - rf_t), before costs and borrow fees
    borrow: pd.Series  # borrow fee on the short notional held into day t (§6.1)
    turnover: pd.Series  # sum_i |w_{i,t} - w_{i,t-1}|
    cost: pd.Series  # c * turnover, charged on the trade day
    excess: pd.Series  # daily return in excess of rf; clipped on the ruin day
    total: pd.Series  # daily total return rf + excess; exactly -1 on the ruin day
    trade_id: pd.Series | pd.DataFrame | None  # trade held at the close of t (NaN = flat)
    held_id: pd.Series | pd.DataFrame | None  # trade that earns day t's P&L, i.e. trade_id shifted by one day
    leg_pos: pd.DataFrame | None  # per-leg trades only: each leg's executed position s_i shifted by k
    pnl: pd.Series | None  # P&L per trade id as a fraction of equity (§7.1), within the window
    trades: pd.DataFrame | None  # one row per trade: the §7.3 fields and ticker (beta_at_entry comes later)
    equity: pd.Series  # baseline E0 dated e - 1, then E_t through stop - 1
    drawdown: pd.Series  # E_t / running max - 1, same index as equity
    t0: int
    k: int
    e: int
    stop: int
    split_pos: int  # calendar position of split_date
    split_date: pd.Timestamp
    ruined: bool
    ruin_date: pd.Timestamp | None

    def segment_bounds(self) -> dict[str, tuple[int, int] | None]:
        """Calendar positions [start, end) of the IS, OOS and full segments (§6.3).

        OOS is None when ruin happened in IS, because the window ends before the split.
        """
        oos = (self.split_pos, self.stop) if self.split_pos < self.stop else None
        return {"is": (self.e, min(self.split_pos, self.stop)), "oos": oos, "full": (self.e, self.stop)}


@dataclass(frozen=True)
class DailyReturns:
    """The §6.1 daily components of a path of executed weights, on the full run calendar.

    The value at date t is for the day ending at the close of t. Returns and costs are
    decimals, as fractions of the day's starting equity.
    """

    w_prev: pd.DataFrame  # weights held into day t
    gross_held: pd.Series  # sum_i |w_{i,t-1}|: gross exposure held into day t
    gross_excess: pd.Series  # sum_i w_{i,t-1} (r_{i,t} - rf_t), before costs and borrow fees
    borrow: pd.Series  # borrow fee on the short notional held into day t
    turnover: pd.Series  # sum_i |w_{i,t} - w_{i,t-1}|
    cost: pd.Series  # c * turnover, charged on the trade day
    excess: pd.Series  # daily return in excess of rf; clip_ruin_day changes the ruin day in place
    total: pd.Series  # rf + excess, likewise


def daily_returns(
    *, w: pd.DataFrame, R: pd.DataFrame, rf: pd.Series, cost_bps: float, borrow_bps: float
) -> DailyReturns:
    """§6.1 from the executed weights on: P&L, borrow fee, turnover, cost, excess and total returns.

    Args:
        w: executed end-of-day weights as fractions of equity, same index and columns as R. The
            value at date t is the weight held at the close of t (already lagged, §2.2).
        R: simple daily returns (decimals). Only the first row may be NaN.
        rf: daily risk-free rate (decimal), same index as R.
        cost_bps: one-way cost per unit of traded notional, per leg, in basis points.
        borrow_bps: annual fee on the short notional held into each day, in basis points.

    run_backtest applies this to W.shift(k), and the walk-forward (§9.2) to its stitched weights.
    """
    # Plain arrays: rows are days, columns are legs. pandas gives the same floats about 15
    # times slower, which adds up over the 21 backtests of a grid.
    c = cost_bps / 10_000  # one-way, per unit of traded notional, per leg
    w_now = w.to_numpy(dtype=float)
    w_prev = np.zeros_like(w_now)  # weights held into day t; flat into day 0
    w_prev[1:] = w_now[:-1]
    r = R.to_numpy(dtype=float, copy=True)
    r[0] = 0.0  # only the first row may be NaN
    rf_day = rf.to_numpy(dtype=float)
    if not np.isfinite(r).all() or not np.isfinite(rf_day).all():
        raise ValueError("R (after its first row) and rf must be finite")  # .sum() would hide NaN as 0
    gross_excess = (w_prev * (r - rf_day[:, None])).sum(axis=1)
    shorts = np.clip(-w_prev, 0.0, None).sum(axis=1)  # short notional held into day t
    borrow = (borrow_bps / 10_000 / 252) * shorts  # charged daily on it
    turnover = np.abs(w_now - w_prev).sum(axis=1)  # every leg counts
    cost = c * turnover  # charged on the trade day
    excess = gross_excess - borrow - cost  # daily return in excess of rf
    total = rf_day + excess  # daily total return (cash earns rf)

    def series(values: np.ndarray) -> pd.Series:
        return pd.Series(values, index=R.index)

    return DailyReturns(
        w_prev=pd.DataFrame(w_prev, index=R.index, columns=R.columns),
        gross_held=series(np.abs(w_prev).sum(axis=1)),
        gross_excess=series(gross_excess),
        borrow=series(borrow),
        turnover=series(turnover),
        cost=series(cost),
        excess=series(excess),
        total=series(total),
    )


def first_ruin_day(total: pd.Series, start: int, end: int | None = None) -> int | None:
    """Calendar position of the first day in [start, end) whose total return is <= -1 (§6.2), or None."""
    hits = np.flatnonzero(total.to_numpy()[start:end] <= -1.0)
    return start + int(hits[0]) if hits.size else None


def clip_ruin_day(excess: pd.Series, total: pd.Series, rf: pd.Series, r: int) -> float:
    """Clip day r's total return to exactly -1 in place (§6.2): limited liability, so E_r = 0.

    Returns the amount by which day r's excess return was raised (>= 0), which run_backtest
    credits to the trade held into r so that the trades still reconcile (§7.2).
    """
    clipped = -1.0 - rf.iloc[r]  # excess return that makes total exactly -1
    credit = clipped - excess.iloc[r]
    excess.iloc[r] = clipped
    total.iloc[r] = -1.0
    return credit


def run_backtest(
    *,
    R: pd.DataFrame,
    W: pd.DataFrame,
    s: pd.Series,
    rf: pd.Series,
    t0: int,
    k: int,
    cost_bps: float,
    borrow_bps: float,
    E0: float,
    split_date: date | pd.Timestamp,
    leg_signals: pd.DataFrame | None = None,
    trades: bool = True,
) -> BacktestResult:
    """Daily returns, costs, borrow fees, equity, ruin and trades (§6.1, §6.2, §7).

    Args:
        R: simple daily returns (decimals), one column per leg. Only the first row may be NaN.
        W: raw (unlagged) target weights as fractions of equity, same index and columns as R.
            The value at date t is the weight wanted at the close of t.
        s: raw signal in {-1, 0, +1}, same index as R.
        rf: daily risk-free rate (decimal), same index as R, no NaN. rf_t accrues over day t.
        t0: calendar position of the first non-NaN score (§6.2); t0 + k >= 1.
        k: execution lag in trading days.
        cost_bps: one-way cost per unit of traded notional, per leg, in basis points.
        borrow_bps: annual fee on the short notional held into each day, in basis points,
            charged daily at borrow_bps / 10,000 / 252. 0 reproduces the earlier phases exactly.
        E0: initial equity (currency units).
        split_date: an already-resolved split date leaving >= 1 IS and >= 1 OOS evaluated day.
        leg_signals: for the trend portfolio (§5.8, §7.4), each leg's raw signal, same index and
            columns as R. Trades are then identified and attributed leg by leg, and s must be the
            "invested" flag: 1 where any leg's signal is non-zero, else 0. None (the default) for
            the other strategies, whose trades span all legs.
        trades: False skips §7's attribution and trade records, leaving trade_id, held_id, pnl
            and trades None. The parameter grids (§9, §9.2) read only the daily series, and the
            bookkeeping is about a third of a backtest's time. Nothing else changes.

    Returns:
        BacktestResult. total_t = rf_t + sum_i w_{i,t-1} (r_{i,t} - rf_t) - borrow_t - cost_t,
        where cash and short proceeds earn rf. A flat day's excess return is exactly 0.

    Raises:
        TypeError: rf is not a pandas Series.
        ValueError: misaligned inputs, a signal before t0, or no IS or OOS evaluated day.
    """
    if not isinstance(rf, pd.Series):
        raise TypeError(f"rf must be a pandas Series on the run calendar (§2.3), not {type(rf).__name__}")
    t0, k = operator.index(t0), operator.index(k)
    _validate_inputs(R, W, s, rf, t0, k, cost_bps, borrow_bps, E0)
    if leg_signals is not None:
        _validate_leg_signals(R, W, s, leg_signals)
    calendar = R.index
    n = len(calendar)
    e = t0 + k
    split = pd.Timestamp(split_date)
    split_pos = int(calendar.searchsorted(split, side="left"))
    n_is = split_pos - e
    if n_is < 1 or split_pos >= n:
        raise ValueError(
            f"split_date {split.date()} leaves {max(n_is, 0)} in-sample and {max(n - max(split_pos, e), 0)} "
            "out-of-sample evaluated days; at least 1 of each is needed"
        )

    # §6.1, computed on the full run calendar.
    w = W.shift(k).fillna(0.0)  # executed end-of-day weights
    pos = s.shift(k).fillna(0.0)  # executed position (used to identify trades)
    d = daily_returns(w=w, R=R, rf=rf, cost_bps=cost_bps, borrow_bps=borrow_bps)
    gross_excess, borrow, cost, excess, total = d.gross_excess, d.borrow, d.cost, d.excess, d.total

    # A position held into day t earns its P&L; for the trend portfolio, any leg's (§5.8).
    leg_pos = None if leg_signals is None else leg_signals.shift(k).fillna(0.0)
    if leg_pos is None:
        held = pos.shift(1, fill_value=0.0) != 0
    else:
        held = (leg_pos.shift(1, fill_value=0.0) != 0).any(axis=1)

    # §6.2 ruin: the first evaluated day with total <= -1 ends the window.
    credit = 0.0
    r = first_ruin_day(total, e)
    ruined = r is not None
    stop = r + 1 if ruined else n
    if ruined:
        if not held.iloc[r]:
            raise ValueError("ruin on a day with no position held into it; the cost inputs are out of range")
        credit = clip_ruin_day(excess, total, rf, r)  # >= 0, credited to the trade(s) held into r

    # §7: trades are runs of non-zero, same-sign executed position (per leg for the trend
    # portfolio). §7.1 attribution follows P&L timing, within the evaluation window.
    window = slice(e, stop)  # every day before e is flat with zero weights
    trade_id = held_id = pnl = records = None
    if trades and leg_pos is None:
        trade_id = pd.Series(_trade_ids(pos.to_numpy()), index=calendar)
        held_id = trade_id.shift(1)  # the trade that earns day t's P&L
        credit_by_day = pd.Series(0.0, index=calendar)
        if ruined:
            credit_by_day.iloc[r] = credit
        pnl = _attribute(
            held_id=held_id,
            trade_id=trade_id,
            gross_excess=gross_excess,
            borrow=borrow,
            cost=cost,
            a=d.gross_held,
            b=w.abs().sum(axis=1),
            credit=credit_by_day,
            window=window,
        )
        records = _trade_records(pnl, trade_id.iloc[:stop], held_id.iloc[:stop], pos, calendar, k, split)
        records["ticker"] = None
        trade_id, held_id = trade_id.rename("trade_id"), held_id.rename("held_id")
    elif trades:
        trade_id, pnl, records = _leg_trades(
            local_id=_trade_ids(leg_pos.to_numpy()),
            leg_pos=leg_pos,
            w=w,
            w_prev=d.w_prev,
            R=R,
            rf=rf,
            cost_bps=cost_bps,
            borrow_bps=borrow_bps,
            credit=credit,
            ruin_day=r,
            window=window,
            stop=stop,
            k=k,
            split=split,
        )
        held_id = trade_id.shift(1)

    # §6.2 equity: a baseline point E0 dated e - 1, then compounding total returns.
    eq = np.empty(stop - e + 1)
    eq[0] = E0
    eq[1:] = E0 * np.cumprod(1.0 + total.to_numpy()[e:stop])
    equity = pd.Series(eq, index=calendar[e - 1 : stop], name="equity")
    drawdown = (equity / equity.cummax() - 1.0).rename("drawdown")  # running max includes the baseline

    return BacktestResult(
        w=w,
        pos=pos.rename("position"),
        held=held.rename("held"),
        gross_held=d.gross_held.rename("gross_held"),
        gross_excess=gross_excess.rename("gross_excess"),
        borrow=borrow.rename("borrow"),
        turnover=d.turnover.rename("turnover"),
        cost=cost.rename("cost"),
        excess=excess.rename("excess"),
        total=total.rename("total"),
        trade_id=trade_id,
        held_id=held_id,
        leg_pos=leg_pos,
        pnl=None if pnl is None else pnl.rename("pnl"),
        trades=records,
        equity=equity,
        drawdown=drawdown,
        t0=t0,
        k=k,
        e=e,
        stop=stop,
        split_pos=split_pos,
        split_date=calendar[split_pos],
        ruined=ruined,
        ruin_date=calendar[stop - 1] if ruined else None,
    )


def _validate_inputs(
    R: pd.DataFrame,
    W: pd.DataFrame,
    s: pd.Series,
    rf: pd.Series,
    t0: int,
    k: int,
    cost_bps: float,
    borrow_bps: float,
    E0: float,
) -> None:
    """§6.1 entry checks, plus three consistency checks the trade bookkeeping relies on."""
    if not (W.index.equals(R.index) and s.index.equals(R.index) and rf.index.equals(R.index)):
        raise ValueError("W, s and rf must have exactly the same index as R")
    if not W.columns.equals(R.columns):
        # pandas aligns on labels: mismatched labels give all-zero P&L with no error.
        raise ValueError(f"W columns {list(W.columns)} must equal R columns {list(R.columns)}")
    if k < 0 or t0 < 0 or t0 + k < 1:
        raise ValueError(f"need k >= 0, t0 >= 0 and t0 + k >= 1 (a baseline day); got t0={t0}, k={k}")
    if t0 + k >= len(R):
        raise ValueError(f"t0 + k = {t0 + k} leaves no evaluated day in {len(R)} rows")
    signal, weights = s.to_numpy(dtype=float), W.to_numpy(dtype=float)
    if not (signal[:t0] == 0).all():
        raise ValueError(f"the signal is non-zero before t0={t0}")
    if not np.isin(signal, (-1.0, 0.0, 1.0)).all():
        raise ValueError("the signal must take values in {-1, 0, +1}")
    if not np.isfinite(weights).all():
        raise ValueError("W must be finite")
    if not (weights[signal == 0] == 0).all():
        raise ValueError("W must be exactly 0 on days the signal is flat")
    if not cost_bps >= 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    if not borrow_bps >= 0:
        raise ValueError(f"borrow_bps must be >= 0, got {borrow_bps}")
    if not E0 > 0:
        raise ValueError(f"E0 must be > 0, got {E0}")


def _validate_leg_signals(R: pd.DataFrame, W: pd.DataFrame, s: pd.Series, leg_signals: pd.DataFrame) -> None:
    """Checks for per-leg trades (§5.8): aligned legs, and s the invested flag they imply."""
    if not (leg_signals.index.equals(R.index) and leg_signals.columns.equals(R.columns)):
        raise ValueError("leg_signals must have exactly the index and columns of R")
    legs, weights = leg_signals.to_numpy(dtype=float), W.to_numpy(dtype=float)
    if not np.isin(legs, (-1.0, 0.0, 1.0)).all():
        raise ValueError("every leg signal must take values in {-1, 0, +1}")
    if not (weights[legs == 0] == 0).all():
        raise ValueError("W must be exactly 0 wherever that leg's signal is flat")
    if not np.array_equal(s.to_numpy(), (legs != 0).any(axis=1).astype(float)):
        raise ValueError("s must be the invested flag: 1 where any leg's signal is non-zero, else 0")


def _trade_ids(pos: np.ndarray) -> np.ndarray:
    """§7: trade ids 1, 2, ... for runs of non-zero, same-sign executed position; NaN when flat.

    pos has one row per day, and optionally one column per leg, each numbered on its own.
    """
    changed = np.ones(pos.shape, dtype=bool)  # the first day differs from the day before it
    changed[1:] = pos[1:] != pos[:-1]
    ids = np.cumsum(changed & (pos != 0), axis=0)
    return np.where(pos != 0, ids, np.nan)


def _attribute(
    *,
    held_id: pd.Series,
    trade_id: pd.Series,
    gross_excess: pd.Series,
    borrow: pd.Series,
    cost: pd.Series,
    a: pd.Series,
    b: pd.Series,
    credit: pd.Series,
    window: slice,
) -> pd.Series:
    """§7.1 attribution of each day's excess return to trades, following P&L timing.

    a and b are the gross exposure held into and out of each day; credit is the ruin day's
    clipped amount (§6.2), credited to the trade held into it. Returns P&L per trade id.
    """
    exit_frac = (a / (a + b).where(a + b > 0)).fillna(0.0)
    to_held = gross_excess - borrow - cost * exit_frac + credit  # attributed to held_id
    to_trade = -(cost - cost * exit_frac)  # attributed to trade_id
    return (
        pd.concat(
            [
                pd.DataFrame({"id": held_id.iloc[window], "v": to_held.iloc[window]}),
                pd.DataFrame({"id": trade_id.iloc[window], "v": to_trade.iloc[window]}),
            ]
        )
        .dropna(subset=["id"])
        .groupby("id")["v"]
        .sum()
    )


def _leg_trades(
    *,
    local_id: np.ndarray,
    leg_pos: pd.DataFrame,
    w: pd.DataFrame,
    w_prev: pd.DataFrame,
    R: pd.DataFrame,
    rf: pd.Series,
    cost_bps: float,
    borrow_bps: float,
    credit: float,
    ruin_day: int | None,
    window: slice,
    stop: int,
    k: int,
    split: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Per-leg trades of the trend portfolio (§7.4).

    §7's identification, §7.1's attribution and §7.3's bookkeeping run leg by leg, each on that
    leg's own exposure, gross excess w_{i,t-1}(r_i - rf), borrow fee and cost c |w_{i,t} - w_{i,t-1}|.
    Every daily term is a sum over legs, so all legs' trades together reconcile with the excess
    return (§7.2). A ruin-day credit is split across the legs held into that day in proportion to
    their gross weight |w_{i,r-1}|.

    Args:
        local_id: _trade_ids of each leg's executed position, numbered within the leg.

    Returns:
        (trade_id, pnl, trades): trade ids per leg as a DataFrame, renumbered 1... across the
        portfolio in order of entry date and then ticker; P&L per renumbered id; the trade records
        with each trade's ticker.

    These are the formulas of _attribute and _trade_records written with numpy arrays, because
    they run once per leg.
    """
    calendar = R.index
    e = window.start
    Rv = R.to_numpy(dtype=float, copy=True)
    Rv[0] = 0.0  # as in daily_returns: only the first row may be NaN
    wv, wp = w.to_numpy(dtype=float), w_prev.to_numpy(dtype=float)
    # §7.1 per leg: each day's leg contribution, split between the trade held into it and the
    # trade at its close by the leg's own exit fraction a / (a + b).
    cost = (cost_bps / 10_000) * np.abs(wv - wp)
    a, b = np.abs(wp), np.abs(wv)
    exit_frac = np.divide(a, a + b, out=np.zeros_like(a), where=a + b > 0)
    gross_excess = wp * (Rv - rf.to_numpy()[:, None])
    borrow = (borrow_bps / 10_000 / 252) * np.maximum(-wp, 0.0)
    to_held = gross_excess - borrow - cost * exit_frac
    to_trade = -(cost - cost * exit_frac)
    if ruin_day is not None:  # split by the gross weight held into the ruin day
        to_held[ruin_day] += credit * a[ruin_day] / a[ruin_day].sum()

    # One row per trade, gathered leg by leg: the leg, its number within the leg, the calendar
    # positions of its entry and last held day (-1 while open), its days held and its P&L.
    n_legs = R.shape[1]
    local = np.nan_to_num(local_id, nan=0.0).astype(np.int64)  # 0 = flat
    pos = leg_pos.to_numpy(dtype=float)
    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in ("leg", "local", "entry", "exit", "days", "pnl")
    }
    sl = slice(e, stop)
    for j in range(n_legs):
        ids = local[:stop, j]
        held = np.r_[0, ids[:-1]]  # held_id: the trade that earns day t's P&L
        m = int(ids.max(initial=0))
        if m == 0:
            continue
        pnl = (
            np.bincount(held[sl], weights=to_held[sl, j], minlength=m + 1)
            + np.bincount(ids[sl], weights=to_trade[sl, j], minlength=m + 1)
        )[1:]
        last_held = np.full(m + 1, -1)
        held_days = np.flatnonzero(held)
        np.maximum.at(last_held, held[held_days], held_days)
        is_open = np.arange(1, m + 1) == ids[-1]  # still held at the end of the last day
        parts["leg"].append(np.full(m, j))
        parts["local"].append(np.arange(1, m + 1))
        parts["entry"].append(np.flatnonzero((ids != held) & (ids > 0)))  # trade i + 1 starts at entry[i]
        parts["exit"].append(np.where(is_open, -1, last_held[1:]))
        parts["days"].append(np.bincount(held, minlength=m + 1)[1:])
        parts["pnl"].append(pnl)

    if not parts["leg"]:  # no leg ever traded
        flat = pd.Series(np.nan, index=calendar[:stop])
        trades = _trade_records(pd.Series(dtype=float), flat, flat, leg_pos.iloc[:, 0], calendar, k, split)
        trades["ticker"] = None
        trade_id = pd.DataFrame(np.nan, index=calendar, columns=R.columns)
        return trade_id, pd.Series(dtype=float, index=pd.Index([], name="id")), trades

    # Portfolio-wide ids follow entry date, then ticker.
    leg, local_no, entry, exit_pos, days, pnl = (np.concatenate(parts[key]) for key in parts)
    tickers = R.columns.to_numpy()
    ticker_rank = np.argsort(np.argsort(tickers.astype(str)))  # alphabetical position of each leg
    order = np.lexsort((ticker_rank[leg], entry))  # the last key sorts first
    leg, local_no, entry, exit_pos, days, pnl = (
        x[order] for x in (leg, local_no, entry, exit_pos, days, pnl)
    )
    ids = np.arange(1, len(order) + 1, dtype=np.int64)
    renumber = np.full((n_legs, local.max() + 1), np.nan)  # trades after a ruin day stay unnumbered
    renumber[leg, local_no] = ids
    trade_id = pd.DataFrame(renumber[np.arange(n_legs), local], index=calendar, columns=R.columns)

    is_open = exit_pos < 0
    exit_date = np.full(len(ids), np.datetime64("NaT", "ns"))
    exit_signal = exit_date.copy()
    exit_date[~is_open] = calendar[exit_pos[~is_open]]
    exit_signal[~is_open] = calendar[exit_pos[~is_open] - k]
    entry_date = calendar[entry]
    trades = pd.DataFrame(
        {
            "id": ids,
            "side": np.where(pos[entry, leg] > 0, "long", "short"),
            "segment": np.where(entry_date < split, "is", "oos"),
            "open": is_open,
            "signal_date": calendar[entry - k],
            "entry_date": entry_date,
            "exit_signal_date": exit_signal,
            "exit_date": exit_date,
            "holding_days": days.astype(np.int64),
            "pnl": pnl,
            "ticker": tickers[leg],
        }
    )
    return trade_id, pd.Series(pnl, index=pd.Index(ids, name="id")), trades


def _trade_records(
    pnl: pd.Series,
    trade_id: pd.Series,
    held_id: pd.Series,
    pos: pd.Series,
    calendar: pd.DatetimeIndex,
    k: int,
    split: pd.Timestamp,
) -> pd.DataFrame:
    """Trade bookkeeping (§7.3) on series truncated at the last evaluated day."""
    ids = pnl.index  # every trade, including one entered on the final day
    entry_date = trade_id.index.to_series().groupby(trade_id).first()  # first date where trade_id == id
    last_held = held_id.index.to_series().groupby(held_id).last()  # last date where held_id == id
    holding_days = held_id.value_counts().reindex(ids, fill_value=0)
    is_open = pd.Series(ids == trade_id.iloc[-1], index=ids)  # still held at the end of the last day
    exit_date = last_held.reindex(ids).where(~is_open)  # NaT -> null for open trades

    entry = pd.DatetimeIndex(entry_date.reindex(ids))
    entry_pos = calendar.get_indexer(entry)
    exit_signal = pd.Series(pd.NaT, index=ids, dtype=calendar.dtype)
    closed = ~is_open.to_numpy()
    exit_signal[closed] = calendar[calendar.get_indexer(pd.DatetimeIndex(exit_date[closed])) - k]
    return pd.DataFrame(
        {
            "id": ids.to_numpy().astype(np.int64),
            "side": np.where(pos.to_numpy()[entry_pos] > 0, "long", "short"),
            "segment": np.where(entry < split, "is", "oos"),  # the segment in which it was entered
            "open": is_open.to_numpy(),
            "signal_date": calendar[entry_pos - k],
            "entry_date": entry,
            "exit_signal_date": exit_signal.to_numpy(),
            "exit_date": exit_date.to_numpy(),
            "holding_days": holding_days.to_numpy().astype(np.int64),
            "pnl": pnl.to_numpy(),
        }
    )
