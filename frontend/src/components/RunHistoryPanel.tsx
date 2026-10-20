// Run history (§11.4, §13 view 12): the stored backtests with their key metrics, "Load settings"
// into the sidebar, and a side-by-side metrics comparison of up to four runs. The list loads when the
// panel is opened and reloads after each successful Run (the reloadKey).
import { Fragment, useEffect, useRef, useState } from 'react'
import { getHealth, getRun, getRuns } from '../api'
import { fmtNum, fmtPct, fmtSignedPct } from '../format'
import { STRATEGY_LABEL } from '../form'
import { comparisonRows, isOtherEngine, MAX_COMPARED } from '../history'
import { failure, type Async } from '../state'
import type { RunSummary, SegmentKey, StoredRun } from '../types'
import { AsyncNotice, ErrorBox, Muted, Panel, Segmented } from './ui'

const SEGMENTS = [
  { value: 'is', label: 'IS' },
  { value: 'oos', label: 'OOS' },
  { value: 'full', label: 'Full' },
] as const

const LIST_LIMIT = 50

/** "2026-10-19T02:35:15+00:00" in the viewer's time zone, to the minute. */
function fmtCreated(iso: string): string {
  const d = new Date(iso)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function EngineCell(props: { version: string; current: string | null }) {
  if (!isOtherEngine(props.version, props.current)) return <span className="text-neutral-500">{props.version}</span>
  return (
    <span
      className="text-amber-300"
      title={`Stored by engine ${props.version}; the API now runs ${props.current}. Fields added since read as —, and re-running may give different numbers.`}
    >
      ⚠ {props.version}
    </span>
  )
}

export function RunHistoryPanel(props: { reloadKey: number; onLoadSettings: (run: StoredRun) => void }) {
  const [open, setOpen] = useState(false)
  const [runs, setRuns] = useState<Async<RunSummary[]>>({ status: 'idle' })
  const [engine, setEngine] = useState<string | null>(null)
  const [selected, setSelected] = useState<number[]>([])
  const [details, setDetails] = useState<Record<number, Async<StoredRun>>>({})
  const [segment, setSegment] = useState<SegmentKey>('full')
  const [loadError, setLoadError] = useState<string | null>(null)
  const requested = useRef(new Set<number>()) // details already requested, so each is fetched once

  useEffect(() => {
    if (!open) return
    const controller = new AbortController() // a reload keeps the old table up until the new list is in
    getRuns(LIST_LIMIT, controller.signal).then(
      (data) => setRuns({ status: 'done', data }),
      (error: unknown) => !controller.signal.aborted && setRuns(failure(error)),
    )
    getHealth(controller.signal).then(
      (h) => setEngine(h.engine_version),
      () => {}, // the list's own error state says the API is unreachable
    )
    return () => controller.abort()
  }, [open, props.reloadKey])

  function fetchDetail(id: number): Promise<StoredRun> {
    requested.current.add(id)
    setDetails((d) => ({ ...d, [id]: { status: 'loading' } }))
    return getRun(id).then(
      (data) => {
        setDetails((d) => ({ ...d, [id]: { status: 'done', data } }))
        return data
      },
      (error: unknown) => {
        requested.current.delete(id) // try again next time
        setDetails((d) => ({ ...d, [id]: failure(error) }))
        throw error
      },
    )
  }

  function toggle(id: number) {
    if (selected.includes(id)) {
      setSelected(selected.filter((s) => s !== id))
      return
    }
    if (selected.length >= MAX_COMPARED) return
    setSelected([...selected, id])
    if (!requested.current.has(id)) fetchDetail(id).catch(() => {})
  }

  async function load(id: number) {
    setLoadError(null)
    const cached = details[id]
    try {
      props.onLoadSettings(cached?.status === 'done' ? cached.data : await fetchDetail(id))
    } catch (error) {
      setLoadError(`Could not load run #${id}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  const compared = selected.map((id) => details[id]).filter((d): d is { status: 'done'; data: StoredRun } => d?.status === 'done')
  const comparing = selected.length > 0

  return (
    <Panel
      title="Run history"
      subtitle="Every backtest the API has stored. Equity curves aren't stored, so the comparison covers metrics only."
      actions={
        <button
          type="button"
          aria-expanded={open}
          onClick={() => {
            if (!open && runs.status !== 'done') setRuns({ status: 'loading' })
            setOpen(!open)
          }}
          className="rounded border border-neutral-700 px-2 py-0.5 text-xs text-neutral-300 hover:text-neutral-100"
        >
          {open ? 'Hide' : 'Show run history'}
        </button>
      }
    >
      {!open ? (
        <Muted>Hidden. Opening it loads the {LIST_LIMIT} most recent runs.</Muted>
      ) : (
        <div className="space-y-4">
          <AsyncNotice state={runs} what="the run history" errorTitle="Run history failed" />
          {loadError && <ErrorBox title="Load settings failed" detail={loadError} />}
          {runs.status === 'done' && runs.data.length === 0 && <Muted>No runs stored yet. Each Run stores its backtest.</Muted>}
          {runs.status === 'done' && runs.data.length > 0 && (
            <div className="max-h-96 overflow-y-auto">
              <table className="w-full text-xs tabular-nums" aria-label="Stored runs">
                <thead className="sticky top-0 bg-neutral-900 text-left text-neutral-500">
                  <tr>
                    <th className="px-2 py-1 font-normal" title={`Select up to ${MAX_COMPARED} runs to compare`}>
                      Cmp
                    </th>
                    <th className="px-2 py-1 text-right font-normal">#</th>
                    <th className="px-2 py-1 font-normal">Stored</th>
                    <th className="px-2 py-1 font-normal">Strategy</th>
                    <th className="px-2 py-1 font-normal">Tickers</th>
                    <th className="px-2 py-1 font-normal">Period</th>
                    <th className="px-2 py-1 text-right font-normal">Sharpe</th>
                    <th className="px-2 py-1 text-right font-normal">Sharpe OOS</th>
                    <th className="px-2 py-1 text-right font-normal">Total return</th>
                    <th className="px-2 py-1 text-right font-normal">Max DD</th>
                    <th className="px-2 py-1 font-normal">Engine</th>
                    <th className="px-2 py-1 font-normal" />
                  </tr>
                </thead>
                <tbody>
                  {runs.data.map((run) => {
                    const checked = selected.includes(run.id)
                    const full = !checked && selected.length >= MAX_COMPARED
                    return (
                      <tr key={run.id} className="border-t border-neutral-800 hover:bg-neutral-800/40" data-run-id={run.id}>
                        <td className="px-2 py-1">
                          <input
                            type="checkbox"
                            aria-label={`Compare run ${run.id}`}
                            checked={checked}
                            disabled={full}
                            title={full ? `At most ${MAX_COMPARED} runs can be compared` : undefined}
                            onChange={() => toggle(run.id)}
                          />
                        </td>
                        <td className="px-2 py-1 text-right text-neutral-400">{run.id}</td>
                        <td className="px-2 py-1 text-neutral-400">{fmtCreated(run.created_at)}</td>
                        <td className="px-2 py-1">{STRATEGY_LABEL[run.strategy]}</td>
                        <td className="max-w-48 truncate px-2 py-1" title={run.tickers.join(' ')}>
                          {run.tickers.join(' ')}
                        </td>
                        <td className="px-2 py-1 text-neutral-400">
                          {run.start} → {run.end}
                        </td>
                        <td className="px-2 py-1 text-right">{fmtNum(run.sharpe_full)}</td>
                        <td className="px-2 py-1 text-right">{fmtNum(run.sharpe_oos)}</td>
                        <td className="px-2 py-1 text-right">{fmtSignedPct(run.total_return_full)}</td>
                        <td className="px-2 py-1 text-right">{fmtPct(run.max_drawdown_full)}</td>
                        <td className="px-2 py-1">
                          <EngineCell version={run.engine_version} current={engine} />
                        </td>
                        <td className="px-2 py-1 text-right">
                          <button
                            type="button"
                            onClick={() => void load(run.id)}
                            title="Fill the sidebar with this run's settings, without running it"
                            className="rounded border border-neutral-700 px-1.5 py-0.5 whitespace-nowrap text-neutral-300 hover:border-emerald-700 hover:text-emerald-300"
                          >
                            Load settings
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
          {comparing && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-[11px] tracking-widest text-neutral-500 uppercase">
                  Comparing {selected.length} run{selected.length === 1 ? '' : 's'}
                </p>
                <Segmented label="Comparison segment" value={segment} options={SEGMENTS} onChange={setSegment} />
              </div>
              {selected.map((id) => {
                const d = details[id]
                return d?.status === 'error' ? <ErrorBox key={id} title={`Run #${id} failed to load`} detail={d.detail} httpStatus={d.httpStatus} /> : null
              })}
              {compared.length > 0 && <ComparisonTable runs={compared.map((d) => d.data)} segment={segment} engine={engine} />}
            </div>
          )}
        </div>
      )}
    </Panel>
  )
}

function ComparisonTable(props: { runs: StoredRun[]; segment: SegmentKey; engine: string | null }) {
  const rows = comparisonRows(props.runs, props.segment)
  const heading = (group: string) =>
    group === 'strategy' ? `Strategy, ${props.segment.toUpperCase()}` : group === 'benchmark' ? `Benchmark, ${props.segment.toUpperCase()}` : 'Settings'
  return (
    <table className="w-full text-xs tabular-nums" aria-label="Run comparison">
      <thead>
        <tr className="text-left text-neutral-500">
          <th className="px-2 py-1 font-normal" />
          {props.runs.map((r) => (
            <th key={r.id} className="px-2 py-1 text-right font-normal">
              Run #{r.id}
              {isOtherEngine(r.engine_version, props.engine) && (
                <span className="ml-1 text-amber-300" title={`Stored by engine ${r.engine_version}`}>
                  ⚠
                </span>
              )}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <Fragment key={`${row.group}-${row.label}`}>
            {(i === 0 || rows[i - 1].group !== row.group) && (
              <tr>
                <td colSpan={props.runs.length + 1} className="px-2 pt-2 text-[11px] tracking-widest text-neutral-500 uppercase">
                  {heading(row.group)}
                </td>
              </tr>
            )}
            <tr className="border-t border-neutral-800">
              <td className="px-2 py-0.5 text-neutral-400">{row.label}</td>
              {row.values.map((v, j) => (
                <td key={j} className="px-2 py-0.5 text-right">
                  {v}
                </td>
              ))}
            </tr>
          </Fragment>
        ))}
      </tbody>
    </table>
  )
}
