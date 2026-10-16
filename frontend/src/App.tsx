// Dashboard shell: the sidebar holds the settings, the main panel the views (§13). A Run sends the
// backtest first; once it succeeds, the sensitivity grid and Monte Carlo go out in parallel.
import { useCallback, useMemo, useRef, useState, type ReactNode } from 'react'
import { ApiError, postBacktest, postMonteCarlo, postSensitivity } from './api'
import { EquityPanel, SignalPanel } from './components/BacktestCharts'
import { KpiGrid } from './components/KpiGrid'
import { MonteCarloPanel } from './components/MonteCarloPanel'
import { SensitivityHeatmap } from './components/SensitivityHeatmap'
import { Sidebar } from './components/Sidebar'
import { TradesTable } from './components/TradesTable'
import { ErrorBoundary, ErrorBox, Loading } from './components/ui'
import { buildRequests, defaultForm, isoDate, STRATEGY_LABEL, type FormState } from './form'
import type { Async } from './state'
import { assumptionLines, benchmarkName, firstOosIndex, seriesRows } from './transform'
import type { BacktestResponse, MonteCarloResponse, ResolvedBacktestRun, SensitivityResponse } from './types'

function failure(error: unknown): { status: 'error'; detail: string; httpStatus: number } {
  if (error instanceof ApiError) return { status: 'error', detail: error.message, httpStatus: error.status }
  return { status: 'error', detail: error instanceof Error ? error.message : String(error), httpStatus: 0 }
}

function RunHeader(props: { bt: BacktestResponse }) {
  const { bt } = props
  const r = bt.resolved
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
      <h2 className="text-lg text-neutral-100">
        {STRATEGY_LABEL[r.strategy]} <span className="text-neutral-500">·</span> {r.tickers.join(' / ')}
      </h2>
      <p className="text-xs text-neutral-500">
        {r.start} → {r.end} · L {r.L} · first evaluated {r.first_evaluated_date} · split {r.split_date} · run #{bt.run_id} ·
        engine {bt.engine_version}
      </p>
    </div>
  )
}

function Warnings(props: { warnings: string[] }) {
  if (props.warnings.length === 0) return null
  return (
    <div role="status" className="rounded border border-amber-800/70 bg-amber-950/30 px-3 py-2 text-sm text-amber-200">
      <p className="mb-1 text-[11px] font-semibold tracking-widest text-amber-400 uppercase">
        {props.warnings.length} warning{props.warnings.length === 1 ? '' : 's'}
      </p>
      <ul className="list-disc space-y-0.5 pl-5">
        {props.warnings.map((w, i) => (
          <li key={i}>{w}</li>
        ))}
      </ul>
    </div>
  )
}

function AssumptionsFooter(props: { resolved: ResolvedBacktestRun }) {
  return (
    <footer className="border-t border-neutral-800 pt-3 text-[11px] text-neutral-500">
      <p className="mb-1 tracking-widest uppercase">Assumptions</p>
      <ul className="space-y-0.5">
        {assumptionLines(props.resolved).map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
    </footer>
  )
}

function EmptyState() {
  return (
    <div className="mt-24 max-w-xl space-y-2 text-sm text-neutral-400">
      <p className="text-neutral-200">No run yet.</p>
      <p>
        Pick a strategy and tickers in the sidebar, then press <span className="text-emerald-400">Run</span>. A run sends
        the backtest first, then the sensitivity grid and the Monte Carlo simulation.
      </p>
      <p className="text-neutral-500">Nothing is requested until you press Run.</p>
    </div>
  )
}

export default function App() {
  const [form, setForm] = useState<FormState>(() => defaultForm(isoDate(new Date())))
  const [formError, setFormError] = useState<string | null>(null)
  const [lastRun, setLastRun] = useState<FormState | null>(null)
  const [backtest, setBacktest] = useState<Async<BacktestResponse>>({ status: 'idle' })
  const [sensitivity, setSensitivity] = useState<Async<SensitivityResponse>>({ status: 'idle' })
  const [monteCarlo, setMonteCarlo] = useState<Async<MonteCarloResponse>>({ status: 'idle' })
  const inFlight = useRef<AbortController | null>(null)

  async function run() {
    const built = buildRequests(form)
    if (!built.ok) {
      setFormError(built.error)
      return
    }
    setFormError(null)
    inFlight.current?.abort() // a newer run replaces any request still in flight
    const controller = new AbortController()
    inFlight.current = controller
    const { signal } = controller
    setLastRun(form)
    setBacktest({ status: 'loading' })
    setSensitivity({ status: 'loading' })
    setMonteCarlo({ status: 'loading' })

    let bt: BacktestResponse
    try {
      bt = await postBacktest(built.backtest, signal)
    } catch (error) {
      if (signal.aborted) return
      setBacktest(failure(error))
      const skipped = { status: 'skipped', reason: 'Not run, because the backtest failed.' } as const
      setSensitivity(skipped)
      setMonteCarlo(skipped)
      return
    }
    if (signal.aborted) return
    setBacktest({ status: 'done', data: bt })

    // These two go out only now: the backtest has filled the price cache, whereas three
    // simultaneous cold-cache requests would each download the same tickers from Yahoo.
    postSensitivity(built.sensitivity, signal).then(
      (data) => !signal.aborted && setSensitivity({ status: 'done', data }),
      (error: unknown) => !signal.aborted && setSensitivity(failure(error)),
    )
    postMonteCarlo(built.monteCarlo, signal).then(
      (data) => !signal.aborted && setMonteCarlo({ status: 'done', data }),
      (error: unknown) => !signal.aborted && setMonteCarlo(failure(error)),
    )
  }

  const onUseSeed = useCallback((seed: number) => setForm((f) => ({ ...f, mcSeed: String(seed) })), [])

  const view = useMemo(() => {
    if (backtest.status !== 'done') return null
    const bt = backtest.data
    try {
      const { rows, unplacedMarkers } = seriesRows(bt.series, bt.trades)
      return {
        bt,
        rows,
        unplacedMarkers,
        firstOos: firstOosIndex(bt.series.is_oos),
        benchmark: benchmarkName(bt.resolved.strategy, bt.resolved.tickers),
        error: null,
      }
    } catch (error) {
      return { bt, error: error instanceof Error ? error.message : String(error) }
    }
  }, [backtest])

  const running = [backtest, sensitivity, monteCarlo].some((s) => s.status === 'loading')
  const stale = lastRun !== null && JSON.stringify(lastRun) !== JSON.stringify(form)

  let content
  if (backtest.status === 'idle') content = <EmptyState />
  else if (backtest.status === 'loading') content = <Loading what="the backtest" />
  else if (backtest.status === 'error') {
    content = <ErrorBox title="Backtest failed" detail={backtest.detail} httpStatus={backtest.httpStatus} />
  } else if (view?.error) {
    content = <ErrorBox title="The API response could not be drawn" detail={view.error} />
  } else if (view && view.error === null) {
    const { bt } = view
    // Each view gets its own boundary, reset by every new run (the key).
    const guard = (title: string, node: ReactNode) => (
      <ErrorBoundary key={`${bt.run_id}-${title}`} title={title}>
        {node}
      </ErrorBoundary>
    )
    content = (
      <>
        <RunHeader bt={bt} />
        <Warnings warnings={bt.warnings} />
        {guard('Performance', <KpiGrid backtest={bt} monteCarlo={monteCarlo} />)}
        {guard('Equity', <EquityPanel rows={view.rows} firstOos={view.firstOos} benchmark={view.benchmark} />)}
        {guard(
          'Signal',
          <SignalPanel rows={view.rows} firstOos={view.firstOos} backtest={bt} unplacedMarkers={view.unplacedMarkers} />,
        )}
        {guard('Trades', <TradesTable trades={bt.trades} pairs={bt.resolved.strategy === 'pairs'} />)}
        {guard('Sensitivity', <SensitivityHeatmap state={sensitivity} />)}
        {guard('Monte Carlo', <MonteCarloPanel state={monteCarlo} onUseSeed={onUseSeed} />)}
        <AssumptionsFooter resolved={bt.resolved} />
      </>
    )
  }

  return (
    <div className="flex h-screen min-w-[1280px] bg-neutral-950 font-mono text-neutral-200">
      <Sidebar
        form={form}
        setForm={setForm}
        onRun={() => void run()}
        running={running}
        formError={formError}
        stale={stale}
      />
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-[1600px] space-y-4 p-6">{content}</div>
      </main>
    </div>
  )
}
