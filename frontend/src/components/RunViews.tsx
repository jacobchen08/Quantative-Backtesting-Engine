// The run's surrounding views (§13): its header, warnings and assumptions footer, the notices
// above the results, the demo's banner, and the empty state before the first run.
import type { DemoManifest } from '../demo'
import { STRATEGY_LABEL } from '../form'
import { assumptionLines } from '../transform'
import type { BacktestResponse, ResolvedBacktestRun } from '../types'

export function RunHeader(props: { bt: BacktestResponse; demo: boolean }) {
  const { bt } = props
  const r = bt.resolved
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
      <h2 className="text-lg text-neutral-100">
        {STRATEGY_LABEL[r.strategy]} <span className="text-neutral-500">·</span>{' '}
        {r.tickers.join(r.strategy === 'trend_portfolio' ? ', ' : ' / ')}
      </h2>
      <p className="text-xs text-neutral-500">
        {r.start} → {r.end} · L {r.L} · first evaluated {r.first_evaluated_date} · split {r.split_date} ·{' '}
        {/* the demo's run ids belong to the recorder's scratch database, not to any history */}
        {props.demo ? 'recorded' : `run #${bt.run_id}`} · engine {bt.engine_version}
      </p>
    </div>
  )
}

export function Warnings(props: { warnings: string[] }) {
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

export function AssumptionsFooter(props: { resolved: ResolvedBacktestRun }) {
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

/** Notices about the sidebar's settings: URL parameters that were ignored, or a loaded run. */
export function Notices(props: { notices: string[]; onDismiss: () => void }) {
  if (props.notices.length === 0) return null
  return (
    <div role="note" className="flex items-start justify-between gap-4 rounded border border-sky-900 bg-sky-950/30 px-3 py-2 text-sm text-sky-200">
      <ul className="space-y-0.5">
        {props.notices.map((n) => (
          <li key={n}>{n}</li>
        ))}
      </ul>
      <button type="button" onClick={props.onDismiss} className="text-xs text-sky-400 hover:text-sky-200">
        Dismiss
      </button>
    </div>
  )
}

/** The demo's banner (§13, "Static demo"). */
export function DemoBanner(props: { demo: DemoManifest }) {
  return (
    <div role="status" className="rounded border border-violet-900 bg-violet-950/30 px-3 py-2 text-sm text-violet-200">
      Demo: results recorded on {props.demo.recorded_on} from Yahoo data through {props.demo.data_end}. Run locally for
      live data.{' '}
      <a
        className="text-violet-400 underline hover:text-violet-200"
        href="https://github.com/jacobchen08/Quantitative-Backtesting-Engine#readme"
      >
        Source and setup
      </a>
    </div>
  )
}

export function EmptyState() {
  return (
    <div className="mt-24 max-w-xl space-y-2 text-sm text-neutral-400">
      <p className="text-neutral-200">No run yet.</p>
      <p>
        Pick a strategy and tickers in the sidebar, then press <span className="text-emerald-400">Run</span>. A run sends
        the backtest first, then the sensitivity grid and the Monte Carlo simulation. The walk-forward has its own
        button in its panel.
      </p>
      <p className="text-neutral-500">Nothing is requested until you press Run.</p>
    </div>
  )
}
