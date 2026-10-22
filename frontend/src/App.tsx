// Dashboard shell: the sidebar holds the settings, the main panel the views (§13). A Run sends the
// backtest first; once it succeeds, the sensitivity grid and Monte Carlo go out in parallel. The
// walk-forward goes out only from its own panel's button. The settings live in the URL:
// loading a URL fills the sidebar without sending anything, and each Run writes them back.
//
// The static demo is this same dashboard with `demo` set: the sidebar offers only the
// recorded scenarios, requests are replayed (demo.ts), a banner dates the recording, the run history
// is hidden, and the chosen scenario is shown at once, since a replay costs nothing.
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { postBacktest, postMonteCarlo, postSensitivity, postWalkForward } from './api'
import { EquityPanel, PortfolioSignalPanel, RollingPanel, SignalPanel } from './components/BacktestCharts'
import { CalendarPanel } from './components/CalendarPanel'
import { CostSweepPanel } from './components/CostSweepPanel'
import { ExportBar } from './components/ExportBar'
import { KpiGrid } from './components/KpiGrid'
import { MonteCarloPanel } from './components/MonteCarloPanel'
import { RunHistoryPanel } from './components/RunHistoryPanel'
import { AssumptionsFooter, DemoBanner, EmptyState, Notices, RunHeader, Warnings } from './components/RunViews'
import { SensitivityHeatmap } from './components/SensitivityHeatmap'
import { Sidebar } from './components/Sidebar'
import { TradesTable } from './components/TradesTable'
import { ErrorBoundary, ErrorBox, Loading } from './components/ui'
import { WalkForwardPanel } from './components/WalkForwardPanel'
import { matchScenario, scenarioForm, type DemoManifest, type DemoScenario } from './demo'
import { buildRequests, buildWalkForward, defaultForm, isoDate, type FormState } from './form'
import { resolvedToForm } from './history'
import { failure, type Async } from './state'
import { benchmarkLabels, firstOosIndex, legRows, seriesRows } from './transform'
import type {
  BacktestResponse,
  MonteCarloResponse,
  SegmentKey,
  SensitivityResponse,
  StoredRun,
  WalkForwardResponse,
} from './types'
import { formToSearch, searchToForm } from './urlState'

/**
 * The sidebar defaults: today is the default end date, so they are rebuilt where they are needed.
 * In the demo they are the first recorded scenario, so its URL is the bare page.
 */
function defaults(demo: DemoManifest | null): FormState {
  return demo ? scenarioForm(demo.scenarios[0]) : defaultForm(isoDate(new Date()))
}

/** The page URL with the settings that differ from the defaults (§13), keeping the path and hash. */
function urlFor(form: FormState, demo: DemoManifest | null): string {
  const { pathname, hash } = window.location
  return `${pathname}${formToSearch(form, defaults(demo))}${hash}`
}

export default function App(props: { demo?: DemoManifest | null }) {
  const demo = props.demo ?? null
  // The URL fills the sidebar on load and sends nothing (§13); its ignored parameters become notices.
  const [initial] = useState(() => searchToForm(window.location.search, defaults(demo)))
  const [form, setForm] = useState<FormState>(initial.form)
  const [notices, setNotices] = useState<string[]>(initial.notices)
  const [formError, setFormError] = useState<string | null>(null)
  const [lastRun, setLastRun] = useState<FormState | null>(null)
  // The demo shows its scenario on load (the effect below), so it starts as loading, not empty.
  const [backtest, setBacktest] = useState<Async<BacktestResponse>>({ status: demo ? 'loading' : 'idle' })
  const [sensitivity, setSensitivity] = useState<Async<SensitivityResponse>>({ status: 'idle' })
  const [monteCarlo, setMonteCarlo] = useState<Async<MonteCarloResponse>>({ status: 'idle' })
  const [walkForward, setWalkForward] = useState<Async<WalkForwardResponse>>({ status: 'idle' })
  const [walkForwardSent, setWalkForwardSent] = useState<string | null>(null) // the request's JSON
  const [showBenchmark, setShowBenchmark] = useState(true) // kept across runs
  const [segment, setSegment] = useState<SegmentKey>('full') // the KPI toggle; the cost sweep follows it
  const [logScale, setLogScale] = useState(false) // the equity axis; kept across runs
  const [historyKey, setHistoryKey] = useState(0) // bumped by each stored backtest, to reload the history
  const inFlight = useRef<AbortController | null>(null)
  const walkForwardInFlight = useRef<AbortController | null>(null)

  async function run(settings: FormState = form) {
    const built = buildRequests(settings)
    if (!built.ok) {
      setFormError(built.error)
      return
    }
    setFormError(null)
    window.history.replaceState(null, '', urlFor(settings, demo)) // settings only, never results
    inFlight.current?.abort() // a newer run replaces any request still in flight
    const controller = new AbortController()
    inFlight.current = controller
    const { signal } = controller
    setLastRun(settings)
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
    setHistoryKey((k) => k + 1) // the backtest is now stored

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

  // The walk-forward reruns the whole grid, so only its own button sends it (§13). A new Run leaves
  // its result in place, marked stale when the sidebar no longer builds the same request.
  async function runWalkForward() {
    const built = buildWalkForward(form)
    if (!built.ok) {
      setWalkForward({ status: 'error', detail: built.error, httpStatus: 0 })
      return
    }
    walkForwardInFlight.current?.abort()
    const controller = new AbortController()
    walkForwardInFlight.current = controller
    setWalkForwardSent(JSON.stringify(built.request))
    setWalkForward({ status: 'loading' })
    try {
      const data = await postWalkForward(built.request, controller.signal)
      if (!controller.signal.aborted) setWalkForward({ status: 'done', data })
    } catch (error) {
      if (!controller.signal.aborted) setWalkForward(failure(error))
    }
  }

  const walkForwardStale = useMemo(() => {
    if (walkForward.status !== 'done') return false
    const built = buildWalkForward(form)
    return !built.ok || JSON.stringify(built.request) !== walkForwardSent
  }, [walkForward, walkForwardSent, form])

  // The demo replays its first scenario (or the URL's) on load; the live dashboard waits for Run.
  const runOnLoad = useRef(demo !== null)
  useEffect(() => {
    if (!runOnLoad.current) return
    runOnLoad.current = false
    void run(initial.form)
  })

  // The demo's scenario picker: a replay is instant, so choosing a scenario shows it.
  function onPickScenario(scenario: DemoScenario) {
    const settings = scenarioForm(scenario)
    setForm(settings)
    setNotices([])
    void run(settings)
  }
  const activeScenario = useMemo(() => (demo ? matchScenario(form, demo.scenarios) : null), [demo, form])

  const onUseSeed = useCallback((seed: number) => setForm((f) => ({ ...f, mcSeed: String(seed) })), [])

  // "Load settings" fills the sidebar from a stored run without running it (§13 view 12).
  const onLoadSettings = useCallback((run: StoredRun) => {
    setForm((f) => resolvedToForm(run.resolved, f))
    setFormError(null)
    setNotices([`Loaded the settings of run #${run.id} into the sidebar. Press Run to re-run it.`])
  }, [])

  const view = useMemo(() => {
    if (backtest.status !== 'done') return null
    const bt = backtest.data
    try {
      const { rows, unplacedMarkers } = seriesRows(bt.series, bt.trades)
      const legs = legRows(bt.series, bt.trades) // trend portfolio only; empty otherwise
      return {
        bt,
        rows,
        legs: legs.legs,
        unplacedMarkers: unplacedMarkers + legs.unplacedMarkers,
        firstOos: firstOosIndex(bt.series.is_oos),
        benchmark: benchmarkLabels(bt.resolved.strategy, bt.resolved.tickers),
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
        <RunHeader bt={bt} demo={demo !== null} />
        <ExportBar backtest={bt} />
        <Warnings warnings={bt.warnings} />
        {guard(
          'Performance',
          <KpiGrid
            backtest={bt}
            monteCarlo={monteCarlo}
            showBenchmark={showBenchmark}
            onShowBenchmark={setShowBenchmark}
            segment={segment}
            onSegment={setSegment}
          />,
        )}
        {guard('Equity', <EquityPanel
            rows={view.rows}
            firstOos={view.firstOos}
            benchmark={view.benchmark}
            showBenchmark={showBenchmark}
            onShowBenchmark={setShowBenchmark}
            logScale={logScale}
            onLogScale={setLogScale}
            ruined={bt.ruined}
          />)}
        {guard('Rolling statistics', <RollingPanel rows={view.rows} firstOos={view.firstOos} />)}
        {guard(
          'Calendar returns',
          <CalendarPanel calendar={bt.calendar} benchmark={view.benchmark} showBenchmark={showBenchmark} />,
        )}
        {guard(
          'Signal',
          bt.resolved.strategy === 'trend_portfolio' ? (
            <PortfolioSignalPanel
              rows={view.rows}
              legs={view.legs}
              firstOos={view.firstOos}
              backtest={bt}
              unplacedMarkers={view.unplacedMarkers}
            />
          ) : (
            <SignalPanel rows={view.rows} firstOos={view.firstOos} backtest={bt} unplacedMarkers={view.unplacedMarkers} />
          ),
        )}
        {guard('Trades', <TradesTable trades={bt.trades} strategy={bt.resolved.strategy} />)}
        {guard('Cost sweep', <CostSweepPanel backtest={bt} segment={segment} />)}
        {guard('Sensitivity', <SensitivityHeatmap state={sensitivity} />)}
        {guard(
          'Walk-forward',
          <WalkForwardPanel
            state={walkForward}
            onRun={() => void runWalkForward()}
            stale={walkForwardStale}
            showBenchmark={showBenchmark}
          />,
        )}
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
        demo={demo && { scenarios: demo.scenarios, active: activeScenario, onPick: onPickScenario }}
      />
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-[1600px] space-y-4 p-6">
          {demo && <DemoBanner demo={demo} />}
          <Notices notices={notices} onDismiss={() => setNotices([])} />
          {content}
          {/* The demo has no database, so no history (§13). */}
          {!demo && (
            <ErrorBoundary title="Run history">
              <RunHistoryPanel reloadKey={historyKey} onLoadSettings={onLoadSettings} />
            </ErrorBoundary>
          )}
        </div>
      </main>
    </div>
  )
}
