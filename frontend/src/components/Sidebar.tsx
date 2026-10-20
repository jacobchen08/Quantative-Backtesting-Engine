// Run settings (§13): the strategy switch, then one section per group of settings
// (SidebarSections.tsx). Nothing is sent until Run is pressed.
//
// In the static demo (Phase 9) a scenario picker heads the sidebar and every other input is
// read-only: only the recorded scenarios can be replayed (§13, "Static demo").
import type { Dispatch, SetStateAction } from 'react'
import type { DemoScenario } from '../demo'
import { STRATEGIES, withStrategy, type FormState } from '../form'
import { Field, INPUT } from './fields'
import {
  ExecutionSection,
  MonteCarloSection,
  PeriodSection,
  SignalSection,
  TickersSection,
  VolTargetingSection,
  WalkForwardSection,
  type SetField,
} from './SidebarSections'
import { Segmented } from './ui'

export function Sidebar(props: {
  form: FormState
  setForm: Dispatch<SetStateAction<FormState>>
  onRun: () => void
  running: boolean
  formError: string | null
  stale: boolean
  /** The static demo's scenarios; null in the live dashboard. */
  demo?: { scenarios: readonly DemoScenario[]; active: DemoScenario | null; onPick: (s: DemoScenario) => void } | null
}) {
  const demo = props.demo ?? null
  const { form, setForm } = props
  const set: SetField = (key) => (value) => setForm((f) => ({ ...f, [key]: value }))
  const sectionProps = { form, set, setForm }

  return (
    <form
      // min/max/step only drive the browser's spinners: without noValidate the browser would block Run
      // for values the API accepts (cost 2.5 bps against step 1) and hide the API's 422 for the rest.
      noValidate
      className="flex h-full w-80 shrink-0 flex-col border-r border-neutral-800 bg-neutral-950"
      onSubmit={(e) => {
        e.preventDefault()
        props.onRun()
      }}
    >
      <div className="border-b border-neutral-800 px-4 py-3">
        <h1 className="text-sm font-semibold tracking-widest text-emerald-400 uppercase">Quantitative Backtesting Engine</h1>
        <p className="text-[11px] text-neutral-500">Backtests · sensitivity · walk-forward · Monte&nbsp;Carlo</p>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-3">
        {demo && (
          <Field label="Demo scenario" hint="The demo replays recorded results, so the settings below are read-only.">
            <select
              className={INPUT}
              value={demo.active?.id ?? ''}
              onChange={(e) => {
                const scenario = demo.scenarios.find((s) => s.id === e.target.value)
                if (scenario) demo.onPick(scenario)
              }}
            >
              {demo.active === null && (
                <option value="" disabled>
                  Not a recorded scenario
                </option>
              )}
              {demo.scenarios.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </Field>
        )}
        {/* A disabled fieldset makes every control in it read-only, the chips and toggles included. */}
        <fieldset disabled={demo !== null} className="m-0 min-w-0 space-y-4 border-0 p-0 disabled:opacity-70">
          <Segmented
            label="Strategy"
            value={form.strategy}
            options={STRATEGIES}
            onChange={(s) => setForm((f) => withStrategy(f, s))}
          />

          <TickersSection {...sectionProps} />
          <PeriodSection {...sectionProps} />
          <SignalSection {...sectionProps} />
          <ExecutionSection {...sectionProps} />
          <VolTargetingSection {...sectionProps} />
          <WalkForwardSection {...sectionProps} />
          <MonteCarloSection {...sectionProps} />
        </fieldset>
      </div>

      <div className="space-y-2 border-t border-neutral-800 px-4 py-3">
        {props.formError && (
          <p role="alert" className="text-xs text-red-300">
            {props.formError}
          </p>
        )}
        {props.stale && !props.running && (
          <p className="text-[11px] text-amber-400/80">Settings changed since the last run.</p>
        )}
        <button
          type="submit"
          className="w-full rounded bg-emerald-600 px-3 py-2 text-sm font-semibold tracking-wider text-neutral-950 uppercase hover:bg-emerald-500 disabled:opacity-60"
        >
          {props.running ? 'Running…' : 'Run'}
        </button>
      </div>
    </form>
  )
}
