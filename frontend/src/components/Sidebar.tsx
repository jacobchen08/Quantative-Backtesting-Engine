// Run settings (§13). Only the controls that apply to the chosen strategy are shown, and the
// request builder (form.ts) sends only those. Nothing is sent until Run is pressed.
//
// In the static demo (Phase 9) a scenario picker heads the sidebar and every other input is
// read-only: only the recorded scenarios can be replayed (§13, "Static demo").
import type { Dispatch, ReactNode, SetStateAction } from 'react'
import type { DemoScenario } from '../demo'
import {
  isTrendLike,
  KALMAN_DELTA_DEFAULT,
  PAIR_PRESETS,
  parseTickerList,
  PORTFOLIO_PRESETS,
  STRATEGIES,
  TRADING_DAYS_PER_MONTH,
  TREND_MONTHS,
  TREND_PRESETS,
  withStrategy,
  type FormState,
} from '../form'
import { Segmented } from './ui'

const INPUT =
  'w-full rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm text-neutral-100 ' +
  'placeholder:text-neutral-600 focus:border-emerald-500 focus:outline-none'

const CHIP = 'rounded border px-1.5 py-0.5 text-[11px] '
const chipClass = (active: boolean) =>
  CHIP +
  (active
    ? 'border-emerald-600 bg-emerald-500/15 text-emerald-300'
    : 'border-neutral-700 text-neutral-400 hover:border-neutral-500 hover:text-neutral-100')

function Section(props: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section aria-label={props.title} className="space-y-3 border-t border-neutral-800 pt-3">
      <div className="flex items-center justify-between">
        <h3 className="text-[11px] font-semibold tracking-widest text-neutral-500 uppercase">{props.title}</h3>
        {props.aside}
      </div>
      {props.children}
    </section>
  )
}

function Field(props: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] text-neutral-400">{props.label}</span>
      {props.children}
      {props.hint && <span className="mt-1 block text-[11px] leading-snug text-neutral-500">{props.hint}</span>}
    </label>
  )
}

/** A number typed as text; form.ts parses it on Run. min/max/step are hints for the browser only. */
function NumberInput(props: {
  value: string
  onChange: (value: string) => void
  min?: number
  max?: number
  step?: number | 'any'
  placeholder?: string
}) {
  return (
    <input
      type="number"
      inputMode="decimal"
      className={INPUT}
      value={props.value}
      min={props.min}
      max={props.max}
      step={props.step ?? 'any'}
      placeholder={props.placeholder}
      onChange={(e) => props.onChange(e.target.value)}
    />
  )
}

function TickerInput(props: { value: string; onChange: (value: string) => void; placeholder?: string }) {
  return (
    <input
      type="text"
      className={INPUT + ' uppercase'}
      value={props.value}
      placeholder={props.placeholder}
      spellCheck={false}
      autoComplete="off"
      onChange={(e) => props.onChange(e.target.value.toUpperCase())}
    />
  )
}

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
  const set =
    <K extends keyof FormState>(key: K) =>
    (value: FormState[K]) =>
      setForm((f) => ({ ...f, [key]: value }))
  const { strategy } = form
  const trendLike = isTrendLike(strategy)
  const portfolio = strategy === 'trend_portfolio'
  const activePair = PAIR_PRESETS.find((p) => p.y === form.pairY && p.x === form.pairX)
  const portfolioList = parseTickerList(form.portfolioTickers)
  const activePortfolio = PORTFOLIO_PRESETS.find((p) => p.tickers.join(' ') === portfolioList.join(' '))

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
        <h1 className="text-sm font-semibold tracking-widest text-emerald-400 uppercase">Quant Engine</h1>
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
            value={strategy}
            options={STRATEGIES}
            onChange={(s) => setForm((f) => withStrategy(f, s))}
          />

          <Section title="Tickers">
            {strategy === 'pairs' ? (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <Field label="Y (tickers[0])">
                    <TickerInput value={form.pairY} onChange={set('pairY')} placeholder="KO" />
                  </Field>
                  <Field label="X (tickers[1])">
                    <TickerInput value={form.pairX} onChange={set('pairX')} placeholder="PEP" />
                  </Field>
                </div>
                <div className="flex flex-wrap gap-1">
                  {PAIR_PRESETS.map((p) => (
                    <button
                      key={`${p.y}/${p.x}`}
                      type="button"
                      title={p.note}
                      className={chipClass(p === activePair)}
                      onClick={() => setForm((f) => ({ ...f, pairY: p.y, pairX: p.x }))}
                    >
                      {p.y}/{p.x}
                    </button>
                  ))}
                </div>
                {activePair && <p className="text-[11px] leading-snug text-neutral-500">{activePair.note}</p>}
              </>
            ) : portfolio ? (
              <>
                <Field
                  label={`Tickers (${portfolioList.length} of 2–12)`}
                  hint="Separated by spaces or commas. The run's calendar is the history they share, so the youngest sets the start."
                >
                  <textarea
                    className={INPUT + ' h-16 resize-none uppercase'}
                    value={form.portfolioTickers}
                    spellCheck={false}
                    autoComplete="off"
                    placeholder="SPY QQQ GLD TLT"
                    onChange={(e) => set('portfolioTickers')(e.target.value.toUpperCase())}
                  />
                </Field>
                <div className="flex flex-wrap gap-1">
                  {PORTFOLIO_PRESETS.map((p) => (
                    <button
                      key={p.label}
                      type="button"
                      title={`${p.tickers.join(', ')}: ${p.note}`}
                      className={chipClass(p === activePortfolio)}
                      onClick={() => set('portfolioTickers')(p.tickers.join(' '))}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
                {activePortfolio && <p className="text-[11px] leading-snug text-neutral-500">{activePortfolio.note}</p>}
              </>
            ) : (
              <>
                <Field label="Ticker">
                  <TickerInput value={form.ticker} onChange={set('ticker')} placeholder="SPY" />
                </Field>
                {strategy === 'trend' && (
                  <div className="flex flex-wrap gap-1">
                    {TREND_PRESETS.map((t) => (
                      <button
                        key={t}
                        type="button"
                        className={chipClass(form.ticker === t)}
                        onClick={() => set('ticker')(t)}
                      >
                        {t}
                      </button>
                    ))}
                  </div>
                )}
              </>
            )}
          </Section>

          <Section title="Period">
            <div className="grid grid-cols-2 gap-2">
              <Field label="Start">
                <input type="date" className={INPUT} value={form.start} onChange={(e) => set('start')(e.target.value)} />
              </Field>
              <Field label="End">
                <input type="date" className={INPUT} value={form.end} onChange={(e) => set('end')(e.target.value)} />
              </Field>
            </div>
            <Field
              label="Split date (optional)"
              hint={form.split === '' ? 'Blank: 70% of evaluated days in-sample.' : 'Out-of-sample starts on this date.'}
            >
              <div className="flex gap-1">
                <input type="date" className={INPUT} value={form.split} onChange={(e) => set('split')(e.target.value)} />
                {form.split !== '' && (
                  <button
                    type="button"
                    className={CHIP + 'border-neutral-700 text-neutral-400 hover:text-neutral-100'}
                    onClick={() => set('split')('')}
                  >
                    clear
                  </button>
                )}
              </div>
            </Field>
          </Section>

          <Section title="Signal">
            <Field
              label={
                strategy === 'pairs' && form.pairsModel === 'kalman'
                  ? 'Start-up window L (trading days)'
                  : 'Lookback L (trading days)'
              }
              hint={
                strategy === 'pairs' && form.pairsModel === 'kalman'
                  ? 'The filter starts from an OLS fit on the first L days.'
                  : undefined
              }
            >
              <NumberInput value={form.L} onChange={set('L')} min={trendLike ? 21 : 10} max={trendLike ? 504 : 252} step={1} />
            </Field>
            {trendLike && (
              <div className="flex flex-wrap gap-1">
                {TREND_MONTHS.map((m) => {
                  const days = String(m * TRADING_DAYS_PER_MONTH)
                  return (
                    <button key={m} type="button" className={chipClass(form.L === days)} onClick={() => set('L')(days)}>
                      {m}m
                    </button>
                  )
                })}
              </div>
            )}
            {!trendLike && (
              <div className="grid grid-cols-2 gap-2">
                <Field label="z entry">
                  <NumberInput value={form.zEntry} onChange={set('zEntry')} min={0} max={5} step={0.1} />
                </Field>
                <Field label="z exit">
                  <NumberInput value={form.zExit} onChange={set('zExit')} min={0} max={5} step={0.1} />
                </Field>
              </div>
            )}
            {trendLike && (
              <Field label="Signal checked">
                <Segmented
                  label="Signal frequency"
                  value={form.signalFrequency}
                  options={[
                    { value: 'monthly', label: 'Monthly', title: 'Decide on the first trading day of each month' },
                    { value: 'daily', label: 'Daily', title: 'Re-check the sign every day (more whipsaw and cost)' },
                  ]}
                  onChange={set('signalFrequency')}
                />
              </Field>
            )}
            {strategy === 'pairs' && (
              <>
                <Field label="Model">
                  <Segmented
                    label="Pairs model"
                    value={form.pairsModel}
                    options={[
                      { value: 'ols', label: 'Rolling OLS', title: 'β and z from an OLS fit on the trailing L days' },
                      { value: 'kalman', label: 'Kalman', title: 'β from a Kalman filter; z is its standardized innovation' },
                    ]}
                    onChange={set('pairsModel')}
                  />
                </Field>
                {form.pairsModel === 'kalman' && (
                  <Field
                    label="State noise δ"
                    hint={`How fast the hedge ratio may drift: q = δ/(1−δ) a day. Default ${KALMAN_DELTA_DEFAULT}; range 1e-7 to 1e-2.`}
                  >
                    <input
                      type="text"
                      inputMode="decimal"
                      className={INPUT}
                      value={form.kalmanDelta}
                      spellCheck={false}
                      onChange={(e) => set('kalmanDelta')(e.target.value.trim())}
                    />
                  </Field>
                )}
                <Field label="Hedge ratio">
                  <Segmented
                    label="Hedge mode"
                    value={form.hedgeMode}
                    options={[
                      { value: 'fixed_at_entry', label: 'Fixed at entry', title: 'β frozen for the whole trade' },
                      {
                        value: 'rolling',
                        label: form.pairsModel === 'kalman' ? 'Filtered' : 'Rolling',
                        title:
                          form.pairsModel === 'kalman'
                            ? 'Re-hedged daily with the filtered β'
                            : 'Re-hedged daily with the rolling β',
                      },
                    ]}
                    onChange={set('hedgeMode')}
                  />
                </Field>
              </>
            )}
          </Section>

          <Section title="Execution">
            <div className="grid grid-cols-2 gap-2">
              <Field label="Cost (bps, one-way)">
                <NumberInput value={form.costBps} onChange={set('costBps')} min={0} max={100} step={1} />
              </Field>
              <Field label="Execution lag">
                <select className={INPUT} value={form.lag} onChange={(e) => set('lag')(e.target.value)}>
                  {[0, 1, 2, 3, 4, 5].map((k) => (
                    <option key={k} value={String(k)}>
                      {k === 1 ? '1 day' : `${k} days`}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <p className="-mt-2 text-[11px] text-neutral-500">
              {form.lag === '0'
                ? 'Fills at the signal’s own close: optimistic.'
                : form.lag === '1'
                  ? 'Fills at the next close (conservative).'
                  : `Fills ${form.lag} closes after the signal.`}
            </p>
            <Field
              label="Borrow fee (bps a year)"
              hint={`Charged daily on short notional${strategy === 'pairs' ? '; a pair always holds a short leg' : ''}.`}
            >
              <NumberInput value={form.borrowBps} onChange={set('borrowBps')} min={0} max={2000} step={25} />
            </Field>
            {strategy !== 'pairs' && (
              <label className="flex items-center gap-1.5 text-xs text-neutral-400">
                <input
                  type="checkbox"
                  className="accent-emerald-500"
                  checked={form.longOnly}
                  onChange={(e) => set('longOnly')(e.target.checked)}
                />
                Long only: short signals are held flat
              </label>
            )}
            <Field label="Risk-free rate">
              <div className="flex items-center gap-2">
                <Segmented
                  label="Risk-free source"
                  value={form.rfMode}
                  options={[
                    { value: 'tbill', label: 'T-bill', title: '13-week T-bill yield (^IRX)' },
                    { value: 'constant', label: 'Constant' },
                  ]}
                  onChange={set('rfMode')}
                />
                {form.rfMode === 'constant' && (
                  <div className="flex items-center gap-1">
                    <NumberInput value={form.rfPct} onChange={set('rfPct')} min={-1} max={25} step={0.25} />
                    <span className="text-xs text-neutral-500">%/yr</span>
                  </div>
                )}
              </div>
            </Field>
          </Section>

          <Section
            title="Vol targeting"
            aside={
              portfolio ? (
                <span className="text-xs text-neutral-500" title="The trend portfolio is always vol-scaled per asset (§5.8)">
                  always on
                </span>
              ) : (
                <label className="flex items-center gap-1.5 text-xs text-neutral-400">
                  <input
                    type="checkbox"
                    className="accent-emerald-500"
                    checked={form.volOn}
                    onChange={(e) => set('volOn')(e.target.checked)}
                  />
                  {form.volOn ? 'On' : 'Off'}
                </label>
              )
            }
          >
            {portfolio || form.volOn ? (
              <>
                <div className="grid grid-cols-2 gap-2">
                  {portfolio ? (
                    <Field label="Target per asset (%/yr)">
                      <NumberInput value={form.portfolioVolPct} onChange={set('portfolioVolPct')} min={5} max={60} step={5} />
                    </Field>
                  ) : (
                    <Field label="Target (%/yr)">
                      <NumberInput value={form.volTargetPct} onChange={set('volTargetPct')} min={1} max={50} step={1} />
                    </Field>
                  )}
                  <Field label="Half-life (days)">
                    <NumberInput value={form.volHalflife} onChange={set('volHalflife')} min={5} max={126} step={1} />
                  </Field>
                </div>
                <Field label="Resize schedule">
                  <Segmented
                    label="Vol refresh"
                    value={form.volRefresh}
                    options={[
                      { value: 'daily', label: 'Daily' },
                      { value: 'weekly', label: 'Weekly' },
                      { value: 'monthly', label: 'Monthly' },
                    ]}
                    onChange={set('volRefresh')}
                  />
                </Field>
                <Field
                  label="Max leverage (×)"
                  hint={
                    portfolio
                      ? 'Each of the N legs is sized to the target and capped at max leverage / N, so the gross exposure stays within max leverage.'
                      : undefined
                  }
                >
                  <NumberInput value={form.maxLeverage} onChange={set('maxLeverage')} min={1} max={4} step={0.25} />
                </Field>
              </>
            ) : (
              <p className="text-[11px] text-neutral-500">Off: unit gross exposure while invested.</p>
            )}
          </Section>

          <Section title="Walk-forward">
            <div className="grid grid-cols-2 gap-2">
              <Field label="Training (days)">
                <NumberInput value={form.wfTrainDays} onChange={set('wfTrainDays')} min={252} max={2520} step={21} />
              </Field>
              <Field label="Test (days)">
                <NumberInput value={form.wfTestDays} onChange={set('wfTestDays')} min={21} max={252} step={21} />
              </Field>
            </div>
            <Field label="Training windows">
              <Segmented
                label="Walk-forward training windows"
                value={form.wfAnchored ? 'anchored' : 'rolling'}
                options={[
                  { value: 'rolling', label: 'Rolling', title: 'Each fold trains on the last T trading days' },
                  {
                    value: 'anchored',
                    label: 'Anchored',
                    title: 'Each fold trains on everything from the grid’s first common day',
                  },
                ]}
                onChange={(v) => set('wfAnchored')(v === 'anchored')}
              />
            </Field>
            <p className="text-[11px] leading-snug text-neutral-500">
              Sent by <span className="text-neutral-400">Run walk-forward</span> in its panel, not by Run: it reruns the
              whole parameter grid.
            </p>
          </Section>

          <Section title="Monte Carlo">
            <div className="grid grid-cols-2 gap-2">
              <Field label="Paths">
                <NumberInput value={form.mcPaths} onChange={set('mcPaths')} min={1000} max={50000} step={1000} />
              </Field>
              <Field label="Horizon (days)">
                <NumberInput value={form.mcHorizon} onChange={set('mcHorizon')} min={5} max={504} step={1} />
              </Field>
              <Field label="Block length">
                {form.mcBlockAuto ? (
                  <input
                    type="text"
                    className={INPUT + ' text-neutral-500'}
                    value="auto"
                    disabled
                    title="Estimated from the history (Politis & White)"
                  />
                ) : (
                  <NumberInput value={form.mcBlockLen} onChange={set('mcBlockLen')} min={1} max={63} step={1} />
                )}
              </Field>
              <Field label="Seed">
                <input
                  type="text"
                  inputMode="numeric"
                  className={INPUT}
                  value={form.mcSeed}
                  placeholder="random"
                  onChange={(e) => set('mcSeed')(e.target.value.trim())}
                />
              </Field>
            </div>
            <label className="flex items-center gap-1.5 text-xs text-neutral-400" title="Politis & White (2004), with the 2009 correction">
              <input
                type="checkbox"
                className="accent-emerald-500"
                checked={form.mcBlockAuto}
                onChange={(e) => set('mcBlockAuto')(e.target.checked)}
              />
              Automatic block length, estimated from the history
            </label>
            <label className="flex items-center gap-1.5 text-xs text-neutral-400">
              <input
                type="checkbox"
                className="accent-emerald-500"
                checked={form.mcZeroDrift}
                onChange={(e) => set('mcZeroDrift')(e.target.checked)}
              />
              Zero drift (μ = 0)
            </label>
          </Section>
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
