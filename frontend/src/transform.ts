// Pure transforms from API responses to chart rows, table orderings and labels.
import { fmtLag, fmtMultiple, fmtRate } from './format'
import type {
  ModelResult,
  ResolvedBacktestRun,
  SegmentKey,
  SensitivityResponse,
  Series,
  Strategy,
  Trade,
} from './types'

// ---------------------------------------------------------------- backtest series

export type SeriesRow = {
  date: string
  equity: number | null
  benchmark: number | null
  drawdown: number | null
  score: number | null
  beta: number | null
  betaUsed: number | null
  exposure: number
  volForecast: number | null
  oos: boolean
  // Trade markers at signal dates, placed on the score (§13); null on every other day.
  entryLong: number | null
  entryShort: number | null
  exit: number | null
}

export type SeriesRows = {
  rows: SeriesRow[]
  /** Markers whose signal date is not in the series. The API's s0 rule (§11.3) makes this 0. */
  unplacedMarkers: number
}

/**
 * One row per series date, in order. No row is ever dropped, even where every value is null,
 * because the synchronized charts line up by row index (§13).
 */
export function seriesRows(series: Series, trades: readonly Trade[]): SeriesRows {
  const n = series.dates.length
  const arrays = [
    series.equity,
    series.benchmark_equity,
    series.drawdown,
    series.daily_return,
    series.position,
    series.gross_exposure,
    series.score,
    series.is_oos,
    ...[series.beta, series.beta_used, series.vol_forecast].filter((a) => a !== null),
  ]
  if (arrays.some((a) => a.length !== n)) throw new Error('The series arrays have different lengths.')

  const rows: SeriesRow[] = series.dates.map((date, i) => ({
    date,
    equity: series.equity[i],
    benchmark: series.benchmark_equity[i],
    drawdown: series.drawdown[i],
    score: series.score[i],
    beta: series.beta?.[i] ?? null,
    betaUsed: series.beta_used?.[i] ?? null,
    exposure: series.gross_exposure[i],
    volForecast: series.vol_forecast?.[i] ?? null,
    oos: series.is_oos[i],
    entryLong: null,
    entryShort: null,
    exit: null,
  }))

  const index = new Map(series.dates.map((date, i) => [date, i]))
  // An exit forced by a NaN score (a degenerate window means flat) has no score to sit on: use 0.
  const markerY = (i: number) => series.score[i] ?? 0
  let unplacedMarkers = 0
  const place = (date: string, field: 'entryLong' | 'entryShort' | 'exit') => {
    const i = index.get(date)
    if (i === undefined) unplacedMarkers += 1
    else rows[i][field] = markerY(i)
  }
  for (const trade of trades) {
    place(trade.signal_date, trade.side === 'long' ? 'entryLong' : 'entryShort')
    if (trade.exit_signal_date !== null) place(trade.exit_signal_date, 'exit')
  }
  return { rows, unplacedMarkers }
}

/** Index of the first out-of-sample row, or -1 if the series ends before the split (ruin in IS). */
export function firstOosIndex(isOos: readonly boolean[]): number {
  return isOos.indexOf(true)
}

/** Strategies A and C are benchmarked against the traded ticker, pairs against SPY (§6.4). */
export function benchmarkName(strategy: Strategy, tickers: readonly string[]): string {
  return strategy === 'pairs' ? 'SPY' : tickers[0]
}

/**
 * The dates a metrics segment covers (§6.3): IS is every evaluated day before the split, OOS the
 * days from the split on. Ruin ends the evaluation window on the ruin day (§6.2).
 */
export function segmentRange(
  r: Pick<ResolvedBacktestRun, 'first_evaluated_date' | 'split_date' | 'end'>,
  ruinDate: string | null,
  segment: SegmentKey,
): string {
  const ruinedInIs = ruinDate !== null && ruinDate < r.split_date // ISO dates order as strings
  const last = ruinDate === null ? r.end : `${ruinDate} (ruin)`
  if (segment === 'is') return ruinedInIs ? `${r.first_evaluated_date} → ${last}` : `${r.first_evaluated_date} → before ${r.split_date}`
  if (segment === 'oos') return ruinedInIs ? 'none: ruined before the split' : `${r.split_date} → ${last}`
  return `${r.first_evaluated_date} → ${last}`
}

/** The assumptions footer (§13), from what the API actually used rather than from the form. */
export function assumptionLines(r: ResolvedBacktestRun): string[] {
  return [
    `Costs: ${r.cost_bps} bps one-way, per leg, on traded notional`,
    `Execution lag: ${fmtLag(r.execution_lag)}`,
    `Risk-free rate: ${r.rf_source}; idle cash and short-sale proceeds earn it`,
    'Rebalancing: daily, to target weights; turnover from price drift is ignored',
    r.vol_target === null
      ? 'Vol targeting: off'
      : `Vol targeting: ${fmtRate(r.vol_target)} annual target, ${r.vol_halflife}-day EWMA half-life, ` +
        `${r.vol_refresh} refresh, ${fmtMultiple(r.max_leverage, 1)} max leverage`,
  ]
}

// ---------------------------------------------------------------- trades table

export type TradeKey = keyof Trade
export type SortDir = 'asc' | 'desc'

/** A sorted copy. Nulls go last in both directions; ties keep trade-id order. */
export function sortTrades(trades: readonly Trade[], key: TradeKey, dir: SortDir): Trade[] {
  const sign = dir === 'asc' ? 1 : -1
  return [...trades].sort((a, b) => {
    const va = a[key]
    const vb = b[key]
    if (va === vb) return a.id - b.id
    if (va === null) return 1
    if (vb === null) return -1
    const cmp = typeof va === 'string' ? (va < (vb as string) ? -1 : 1) : Number(va) - Number(vb)
    return sign * cmp || a.id - b.id
  })
}

// ---------------------------------------------------------------- sensitivity heatmap

export type HeatCell = { sharpe: number | null; L: number; zEntry: number | null; selected: boolean }

export type HeatLayout = {
  rowLabels: string[]
  colLabels: string[]
  cells: HeatCell[][] // [row][col]
  rowAxis: string
  colAxis: string
}

/** "252 (12m)" when L is a whole number of 21-day months, else "250". */
export function lookbackLabel(L: number): string {
  return L % 21 === 0 ? `${L} (${L / 21}m)` : String(L)
}

/**
 * Rows = L and columns = z_entry for mean reversion and pairs. Trend has no z_entry axis, so its
 * one-column matrix is shown as a single row over L (§13).
 */
export function heatmapLayout(resp: SensitivityResponse): HeatLayout {
  const { selected } = resp
  const isSelected = (L: number, z: number | null) => selected !== null && selected.L === L && selected.z_entry === z
  const zs = resp.z_entry_values
  if (zs === null) {
    return {
      rowLabels: ['IS Sharpe'],
      colLabels: resp.L_values.map(lookbackLabel),
      cells: [resp.L_values.map((L, i) => ({ sharpe: resp.sharpe[i][0], L, zEntry: null, selected: isSelected(L, null) }))],
      rowAxis: '',
      colAxis: 'Lookback L (trading days)',
    }
  }
  return {
    rowLabels: resp.L_values.map((L) => `L ${L}`),
    colLabels: zs.map((z) => `z ${Number.isInteger(z) ? z.toFixed(1) : z}`), // 2 -> "z 2.0", 2.25 -> "z 2.25"
    cells: resp.L_values.map((L, i) =>
      zs.map((z, j) => ({ sharpe: resp.sharpe[i][j], L, zEntry: z, selected: isSelected(L, z) })),
    ),
    rowAxis: 'Lookback L',
    colAxis: 'z entry',
  }
}

/** The colour scale's half-width: the largest |Sharpe| in the grid, but at least 0.5. */
export function heatmapBound(cells: readonly HeatCell[][]): number {
  const values = cells.flat().map((c) => c.sharpe)
  return Math.max(0.5, ...values.filter((s): s is number => s !== null).map(Math.abs))
}

const NEUTRAL = [38, 38, 38] // neutral-800: a Sharpe of 0
const NEGATIVE = [239, 68, 68] // red-500
const POSITIVE = [16, 185, 129] // emerald-500
export const NULL_CELL = 'rgb(23, 23, 23)' // neutral-900

/** Diverging colour centred on a Sharpe of 0, saturating at ±bound (§13). */
export function divergingColor(value: number | null, bound: number): string {
  if (value === null || !Number.isFinite(value)) return NULL_CELL
  const t = Math.max(-1, Math.min(1, value / bound))
  const end = t < 0 ? NEGATIVE : POSITIVE
  const mix = NEUTRAL.map((c, k) => Math.round(c + (end[k] - c) * Math.abs(t)))
  return `rgb(${mix.join(', ')})`
}

// ---------------------------------------------------------------- Monte Carlo

export type FanRow = { t: number; p5: number; p25: number; p50: number; p75: number; p95: number; paths: number[] }

/** One row per horizon day t = 0..T: the percentile bands and the sample paths (equity multiples). */
export function fanRows(model: ModelResult): FanRow[] {
  const { p5, p25, p50, p75, p95 } = model.bands
  return p50.map((_, t) => ({
    t,
    p5: p5[t],
    p25: p25[t],
    p50: p50[t],
    p75: p75[t],
    p95: p95[t],
    paths: model.sample_paths.map((path) => path[t]),
  }))
}

export type HistRow = {
  lo: number
  hi: number
  mid: number
  count: number
  share: number // of all paths
  tail: 'low' | 'high' | null // the end bins also hold every path beyond them (§10.4)
}

/**
 * Round axis ticks inside [lo, hi]: the multiples of a step of 1, 2, 2.5, 5 or 10 × 10^k whose
 * count is closest to `target` (ties go to more ticks). The histogram's domain is its exact bin
 * edges, which are not round numbers.
 */
export function niceTicks(lo: number, hi: number, target = 6): number[] {
  if (!(hi > lo)) return [lo]
  const magnitude = 10 ** Math.floor(Math.log10((hi - lo) / target))
  const count = (step: number) => Math.floor(hi / step) - Math.ceil(lo / step) + 1
  let step = magnitude
  for (const m of [2, 2.5, 5, 10]) {
    const candidate = m * magnitude
    if (Math.abs(count(candidate) - target) < Math.abs(count(step) - target)) step = candidate
  }
  const ticks: number[] = []
  for (let i = Math.ceil(lo / step); i * step <= hi; i += 1) {
    ticks.push(Number((i * step).toPrecision(12)) || 0) // strip float noise (0.30000000000000004) and -0
  }
  return ticks
}

export function histogramRows(counts: readonly number[], edges: readonly number[]): HistRow[] {
  if (edges.length !== counts.length + 1) throw new Error('A histogram needs one more edge than counts.')
  const total = counts.reduce((sum, c) => sum + c, 0)
  return counts.map((count, i) => ({
    lo: edges[i],
    hi: edges[i + 1],
    mid: (edges[i] + edges[i + 1]) / 2,
    count,
    share: total > 0 ? count / total : 0,
    tail: i === 0 ? 'low' : i === counts.length - 1 ? 'high' : null,
  }))
}
