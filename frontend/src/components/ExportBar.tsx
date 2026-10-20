// CSV export buttons (§13 view 13). The files are built in the browser from the backtest response
// by the pure helpers in csv.ts; nothing is requested from the API.
import { calendarCsv, csvFileName, seriesCsv, tradesCsv, type CsvTable } from '../csv'
import type { BacktestResponse } from '../types'

/** Saves text as a file through a temporary object URL and a[download]. */
function download(fileName: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = fileName
  document.body.append(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 0) // after the click has started the download
}

const TABLES: readonly { table: CsvTable; label: string; title: string; build: (bt: BacktestResponse) => string }[] = [
  { table: 'trades', label: 'Trades', title: 'Every trade, one row each', build: (bt) => tradesCsv(bt.trades) },
  { table: 'series', label: 'Series', title: 'The daily series arrays, one row per date', build: (bt) => seriesCsv(bt.series) },
  { table: 'calendar', label: 'Calendar', title: 'Monthly and annual returns', build: (bt) => calendarCsv(bt.calendar) },
]

export function ExportBar(props: { backtest: BacktestResponse }) {
  const bt = props.backtest
  return (
    <div className="flex items-center gap-2 text-xs text-neutral-500" role="group" aria-label="Export CSV">
      <span className="tracking-widest uppercase">Export CSV</span>
      {TABLES.map(({ table, label, title, build }) => (
        <button
          key={table}
          type="button"
          title={`${title}: ${csvFileName(bt.resolved, table)}`}
          onClick={() => download(csvFileName(bt.resolved, table), build(bt))}
          className="rounded border border-neutral-700 px-2 py-0.5 text-neutral-300 hover:text-neutral-100"
        >
          {label}
        </button>
      ))}
    </div>
  )
}
