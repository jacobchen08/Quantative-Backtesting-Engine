// Monthly and annual returns of the evaluation window (§8.2, §13 view 10): a years × months CSS
// grid on the diverging scale centred on 0, partial periods hatched, and each year's total.
import { memo, useMemo } from 'react'
import { fmtSignedPct } from '../format'
import { calendarGrid, divergingColor, type BenchmarkLabels, type CalendarCell } from '../transform'
import type { CalendarReturns } from '../types'
import { Panel } from './ui'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const HATCH = 'repeating-linear-gradient(135deg, rgba(10, 10, 10, 0.6) 0 3px, transparent 3px 7px)'

function MonthCell(props: { cell: CalendarCell | null; bound: number; label: string }) {
  const { cell } = props
  if (cell === null) return <div className="h-7" />
  return (
    <div
      title={`${props.label}: ${fmtSignedPct(cell.value, 2)}${cell.partial ? ' (partial month)' : ''}`}
      className="flex h-7 items-center justify-center rounded text-neutral-50"
      style={{ backgroundColor: divergingColor(cell.value, props.bound), backgroundImage: cell.partial ? HATCH : undefined }}
    >
      {fmtSignedPct(cell.value, 1)}
    </div>
  )
}

function YearCell(props: { value: number; partial: boolean; label: string; muted?: boolean }) {
  const tone = props.muted ? 'text-slate-400' : props.value < 0 ? 'text-red-300' : 'text-emerald-300'
  return (
    <div
      title={`${props.label}: ${fmtSignedPct(props.value, 2)}${props.partial ? ' (partial year)' : ''}`}
      className={`flex h-7 items-center justify-end rounded bg-neutral-900 px-2 ${tone}`}
      style={{ backgroundImage: props.partial ? HATCH : undefined }}
    >
      {fmtSignedPct(props.value, 1)}
    </div>
  )
}

export const CalendarPanel = memo(function CalendarPanel(props: {
  calendar: CalendarReturns
  benchmark: BenchmarkLabels
  showBenchmark: boolean
}) {
  const { years, bound } = useMemo(() => calendarGrid(props.calendar), [props.calendar])
  const { showBenchmark } = props
  const columns = `max-content repeat(12, minmax(3.25rem, 1fr)) minmax(4.5rem, 1fr)${showBenchmark ? ' minmax(4.5rem, 1fr)' : ''}`
  const stops = [-1, -0.5, 0, 0.5, 1].map((t) => divergingColor(t * bound, bound))
  return (
    <Panel
      title="Calendar returns"
      subtitle="Total returns compounded within each month and year of the evaluation window. Hatched: partial periods, which the window covers only in part."
    >
      <div className="space-y-3 overflow-x-auto">
        <div className="grid gap-1 text-[11px] tabular-nums" style={{ gridTemplateColumns: columns }}>
          <div />
          {MONTHS.map((m) => (
            <div key={m} className="text-center text-neutral-400">
              {m}
            </div>
          ))}
          <div className="text-right text-neutral-400">Year</div>
          {showBenchmark && <div className="text-right text-slate-400">{props.benchmark.short}</div>}
          {years.map((y) => [
            <div key={`label-${y.year}`} className="flex items-center pr-2 text-neutral-400">
              {y.year}
            </div>,
            ...y.months.map((cell, i) => (
              <MonthCell key={`${y.year}-${i}`} cell={cell} bound={bound} label={`${MONTHS[i]} ${y.year}`} />
            )),
            <YearCell key={`year-${y.year}`} value={y.strategy} partial={y.partial} label={`${y.year}`} />,
            ...(showBenchmark
              ? [
                  <YearCell
                    key={`bench-${y.year}`}
                    value={y.benchmark}
                    partial={y.partial}
                    label={`${y.year} ${props.benchmark.phrase}`}
                    muted
                  />,
                ]
              : []),
          ])}
        </div>
        <div className="flex items-center gap-2 text-[11px] text-neutral-500">
          <span>{fmtSignedPct(-bound, 1)}</span>
          <span className="h-2 w-40 rounded" style={{ background: `linear-gradient(to right, ${stops.join(', ')})` }} />
          <span>{fmtSignedPct(bound, 1)}</span>
          <span>monthly return (colour centred on 0)</span>
          <span className="ml-3 inline-block h-3 w-5 rounded bg-neutral-700" style={{ backgroundImage: HATCH }} />
          <span>partial period</span>
        </div>
      </div>
    </Panel>
  )
})
