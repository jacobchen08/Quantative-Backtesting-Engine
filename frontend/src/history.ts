// Run history (§11.4, §13 view 12): loading a stored run's settings into the sidebar, and the
// side-by-side comparison of up to four stored runs. Equity curves are not stored, so the
// comparison covers metrics only.
import { DASH, fmtCI, fmtInt, fmtMultiple, fmtNum, fmtPct, fmtPerYear, fmtRate, fmtSignedPct, fmtTurnover } from './format'
import { STRATEGY_LABEL, withStrategy, type FormState } from './form'
import type { Metrics, SegmentKey, StoredRun } from './types'

export const MAX_COMPARED = 4

/** A decimal rate as the percent string the sidebar holds: 0.1 -> "10", 0.035 -> "3.5". */
export function percentField(x: number): string {
  return String(Number((x * 100).toPrecision(12))) // 0.07 * 100 is 7.000000000000001
}

/**
 * The sidebar a stored run's resolved request describes (§13: "Load settings" fills the sidebar
 * without running). Settings a run doesn't store, Monte Carlo and walk-forward, keep the values in
 * `current`, as do controls hidden for the run's strategy, whose fields the API echoes as null.
 *
 * Fields an older engine didn't store come back null (§11.4). Each such field was added with a
 * default equal to the earlier behaviour, so a null loads as that default: no borrow fee, shorts
 * allowed, and the rolling-OLS pairs model. A field left out (the static demo passes a request,
 * not a resolved run) is read the same way as a null.
 */
export function resolvedToForm(r: Partial<StoredRun['resolved']>, current: FormState): FormState {
  const strategy = r.strategy ?? current.strategy
  const form: FormState = { ...withStrategy(current, strategy) }
  const tickers = r.tickers ?? []
  if (strategy === 'pairs') {
    form.pairY = tickers[0] ?? form.pairY
    form.pairX = tickers[1] ?? form.pairX
  } else if (strategy === 'trend_portfolio') {
    if (tickers.length > 0) form.portfolioTickers = tickers.join(' ')
  } else if (tickers.length > 0) {
    form.ticker = tickers[0]
  }
  if (r.start != null) form.start = r.start
  if (r.end != null) form.end = r.end
  form.split = r.split_date ?? '' // the split the run used, so a re-run evaluates the same windows
  if (r.L != null) form.L = String(r.L)
  if (r.z_entry != null) form.zEntry = String(r.z_entry)
  if (r.z_exit != null) form.zExit = String(r.z_exit)
  if (r.signal_frequency != null) form.signalFrequency = r.signal_frequency
  if (strategy === 'pairs') {
    if (r.hedge_mode != null) form.hedgeMode = r.hedge_mode
    form.pairsModel = r.pairs_model ?? 'ols'
    if (r.kalman_delta != null) form.kalmanDelta = String(r.kalman_delta)
  }
  if (r.cost_bps != null) form.costBps = String(r.cost_bps)
  form.borrowBps = String(r.borrow_bps ?? 0)
  if (strategy !== 'pairs') form.longOnly = r.allow_short === false
  if (r.execution_lag != null) form.lag = String(r.execution_lag)
  if (r.risk_free === 'tbill' || r.risk_free == null) form.rfMode = 'tbill'
  else {
    form.rfMode = 'constant'
    form.rfPct = percentField(r.risk_free)
  }
  if (strategy === 'trend_portfolio') {
    if (r.vol_target != null) form.portfolioVolPct = percentField(r.vol_target)
  } else {
    form.volOn = r.vol_target != null
    if (r.vol_target != null) form.volTargetPct = percentField(r.vol_target)
  }
  if (r.vol_target != null) {
    if (r.vol_halflife != null) form.volHalflife = String(r.vol_halflife)
    if (r.vol_refresh != null) form.volRefresh = r.vol_refresh
    if (r.max_leverage != null) form.maxLeverage = String(r.max_leverage)
  }
  return form
}

/** True when a stored run came from a different engine version than the API's (§11.4). */
export function isOtherEngine(runVersion: string, current: string | null): boolean {
  return current !== null && runVersion !== current
}

export type CompareRow = { label: string; values: string[]; group: 'settings' | 'strategy' | 'benchmark' }

type M = { [K in keyof Metrics]?: Metrics[K] | null } | null // a stored run's fields may be null

const METRIC_ROWS: readonly { label: string; cell: (m: NonNullable<M>) => string }[] = [
  { label: 'Sharpe', cell: (m) => fmtNum(m.sharpe) },
  { label: 'Sharpe 95% CI', cell: (m) => fmtCI(m.sharpe_ci95) },
  { label: 'Sortino', cell: (m) => fmtNum(m.sortino) },
  { label: 'CAGR', cell: (m) => fmtPct(m.cagr) },
  { label: 'Total return', cell: (m) => fmtSignedPct(m.total_return) },
  { label: 'Ann. vol', cell: (m) => fmtPct(m.ann_vol) },
  { label: 'Max drawdown', cell: (m) => fmtPct(m.max_drawdown) },
  { label: 'Beta to SPY', cell: (m) => fmtNum(m.beta_spy) },
  { label: 'Closed trades', cell: (m) => fmtInt(m.n_closed_trades) },
  { label: 'Win rate', cell: (m) => fmtPct(m.win_rate, 0) },
  { label: 'Days in market', cell: (m) => fmtPct(m.pct_days_in_market, 0) },
  { label: 'Avg gross exposure', cell: (m) => fmtMultiple(m.avg_gross_exposure) },
  { label: 'Turnover', cell: (m) => fmtTurnover(m.ann_turnover) },
  { label: 'Cost drag', cell: (m) => fmtPerYear(m.ann_cost_drag) },
  { label: 'Borrow drag', cell: (m) => fmtPerYear(m.ann_borrow_drag) },
  { label: 'Days', cell: (m) => fmtInt(m.n_days) },
]

const BENCHMARK_ROWS = METRIC_ROWS.filter((row) =>
  ['Sharpe', 'CAGR', 'Total return', 'Ann. vol', 'Max drawdown'].includes(row.label),
)

const num = (x: number | null | undefined) => (x == null ? DASH : String(x))

/** "Rolling OLS" or "Kalman δ 1e-6" for pairs; a run stored before Phase 7 had only rolling OLS. */
function pairsModelLabel(r: StoredRun['resolved']): string {
  if (r.strategy !== 'pairs') return DASH
  return r.pairs_model === 'kalman' ? `Kalman δ ${r.kalman_delta == null ? DASH : r.kalman_delta.toExponential()}` : 'Rolling OLS'
}

/**
 * The comparison table: one column per run, in the order given. Settings first, then the
 * strategy's metrics for the chosen segment, then its benchmark's. A run with no metrics for the
 * segment (OOS after ruin in IS, §6.2) shows dashes there.
 */
export function comparisonRows(runs: readonly StoredRun[], segment: SegmentKey): CompareRow[] {
  const settings: CompareRow[] = [
    { label: 'Strategy', values: runs.map((r) => (r.resolved.strategy ? STRATEGY_LABEL[r.resolved.strategy] : DASH)) },
    { label: 'Tickers', values: runs.map((r) => (r.resolved.tickers ?? []).join(' ') || DASH) },
    { label: 'Period', values: runs.map((r) => `${r.resolved.start ?? DASH} → ${r.resolved.end ?? DASH}`) },
    { label: 'Split', values: runs.map((r) => r.resolved.split_date ?? DASH) },
    { label: 'L', values: runs.map((r) => num(r.resolved.L)) },
    { label: 'z entry / exit', values: runs.map((r) => (r.resolved.z_entry == null ? DASH : `${r.resolved.z_entry} / ${num(r.resolved.z_exit)}`)) },
    { label: 'Pairs model', values: runs.map((r) => pairsModelLabel(r.resolved)) },
    { label: 'Vol target', values: runs.map((r) => (r.resolved.vol_target == null ? 'off' : fmtRate(r.resolved.vol_target))) },
    { label: 'Cost (bps)', values: runs.map((r) => num(r.resolved.cost_bps)) },
    { label: 'Engine', values: runs.map((r) => r.engine_version) },
  ].map((row) => ({ ...row, group: 'settings' as const }))
  const pick = (r: StoredRun, which: 'metrics' | 'benchmark_metrics'): M => r[which]?.[segment] ?? null
  const metricRows = (rows: typeof METRIC_ROWS, which: 'metrics' | 'benchmark_metrics', group: CompareRow['group']) =>
    rows.map((row) => ({
      label: row.label, // the table heads the benchmark rows with the benchmark's name
      values: runs.map((r) => {
        const m = pick(r, which)
        return m === null ? DASH : row.cell(m)
      }),
      group,
    }))
  return [...settings, ...metricRows(METRIC_ROWS, 'metrics', 'strategy'), ...metricRows(BENCHMARK_ROWS, 'benchmark_metrics', 'benchmark')]
}
