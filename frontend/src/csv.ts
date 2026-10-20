// CSV export (§13 view 13). Every table goes through one pure helper, toCsv: RFC 4180 quoting, CRLF
// line endings, numbers at full precision (String(x), never the formatted text) and nulls as empty
// fields. Building the file in the browser needs no API call.
import type { CalendarReturns, ResolvedBacktestRun, Series, Trade } from './types'

export type CsvValue = string | number | boolean | null | undefined

/** One field: quoted when it holds a comma, a double quote, CR or LF, with inner quotes doubled. */
export function csvField(value: CsvValue): string {
  if (value === null || value === undefined) return ''
  const s = typeof value === 'string' ? value : String(value)
  return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s
}

/** A header row and data rows as RFC 4180 text. Every record, the last included, ends in CRLF. */
export function toCsv(header: readonly string[], rows: readonly (readonly CsvValue[])[]): string {
  for (const row of rows) {
    if (row.length !== header.length) throw new Error(`CSV row has ${row.length} fields, header has ${header.length}`)
  }
  return [header, ...rows].map((row) => row.map(csvField).join(',') + '\r\n').join('')
}

export type CsvTable = 'trades' | 'series' | 'calendar'

/** `{strategy}-{tickers}-{start}-{end}-{table}.csv`, the tickers joined by "_" (tickers may contain "-"). */
export function csvFileName(resolved: Pick<ResolvedBacktestRun, 'strategy' | 'tickers' | 'start' | 'end'>, table: CsvTable) {
  return `${resolved.strategy}-${resolved.tickers.join('_')}-${resolved.start}-${resolved.end}-${table}.csv`
}

export const TRADE_COLUMNS = [
  'id',
  'ticker',
  'side',
  'segment',
  'open',
  'signal_date',
  'entry_date',
  'exit_signal_date',
  'exit_date',
  'holding_days',
  'pnl',
  'beta_at_entry',
] as const satisfies readonly (keyof Trade)[]

/** The trades table, one row per trade in id order, with the API's field names as the header. */
export function tradesCsv(trades: readonly Trade[]): string {
  const sorted = [...trades].sort((a, b) => a.id - b.id)
  return toCsv(TRADE_COLUMNS, sorted.map((t) => TRADE_COLUMNS.map((key) => t[key])))
}

type Column = { name: string; values: readonly CsvValue[] }

/**
 * The `series` arrays, one row per date. Arrays the API sends as null for this strategy (score for
 * the trend portfolio, beta outside pairs, vol_forecast with the overlay off) are left out; the
 * trend portfolio's legs add `{ticker}_score`, `_position`, `_weight` and `_vol_forecast` columns.
 */
export function seriesCsv(series: Series): string {
  const optional = (name: string, values: readonly CsvValue[] | null): Column[] => (values === null ? [] : [{ name, values }])
  const columns: Column[] = [
    { name: 'date', values: series.dates },
    { name: 'equity', values: series.equity },
    { name: 'benchmark_equity', values: series.benchmark_equity },
    { name: 'drawdown', values: series.drawdown },
    { name: 'daily_return', values: series.daily_return },
    { name: 'position', values: series.position },
    { name: 'gross_exposure', values: series.gross_exposure },
    ...optional('score', series.score),
    ...optional('beta', series.beta),
    ...optional('beta_used', series.beta_used),
    ...optional('vol_forecast', series.vol_forecast),
    { name: 'is_oos', values: series.is_oos },
    { name: 'rolling_sharpe', values: series.rolling_sharpe },
    { name: 'rolling_beta', values: series.rolling_beta },
    ...(series.legs ?? []).flatMap((leg) => [
      { name: `${leg.ticker}_score`, values: leg.score },
      { name: `${leg.ticker}_position`, values: leg.position },
      { name: `${leg.ticker}_weight`, values: leg.weight },
      { name: `${leg.ticker}_vol_forecast`, values: leg.vol_forecast },
    ]),
  ]
  const n = series.dates.length
  for (const c of columns) {
    if (c.values.length !== n) throw new Error(`series.${c.name} has ${c.values.length} values for ${n} dates`)
  }
  const rows = series.dates.map((_, i) => columns.map((c) => c.values[i]))
  return toCsv(
    columns.map((c) => c.name),
    rows,
  )
}

export const CALENDAR_COLUMNS = ['period', 'year', 'month', 'strategy', 'benchmark', 'partial'] as const

/** Monthly rows (period "month"), then annual rows (period "year", month empty). */
export function calendarCsv(calendar: CalendarReturns): string {
  return toCsv(CALENDAR_COLUMNS, [
    ...calendar.monthly.map((m) => ['month', m.year, m.month, m.strategy, m.benchmark, m.partial]),
    ...calendar.annual.map((y) => ['year', y.year, null, y.strategy, y.benchmark, y.partial]),
  ])
}
