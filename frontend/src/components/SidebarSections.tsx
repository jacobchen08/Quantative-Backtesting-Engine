// The sidebar's sections (§13). Each shows only the controls that apply to the chosen strategy, and
// the request builder (form.ts) sends only those.
import type { Dispatch, SetStateAction } from 'react'
import {
  isTrendLike,
  KALMAN_DELTA_DEFAULT,
  PAIR_PRESETS,
  parseTickerList,
  PORTFOLIO_PRESETS,
  TRADING_DAYS_PER_MONTH,
  TREND_MONTHS,
  TREND_PRESETS,
  type FormState,
} from '../form'
import { Chip, Field, INPUT, NumberInput, Section, TickerInput } from './fields'
import { Segmented } from './ui'

/** Sets one sidebar field. */
export type SetField = <K extends keyof FormState>(key: K) => (value: FormState[K]) => void

export type SectionProps = {
  form: FormState
  set: SetField
  setForm: Dispatch<SetStateAction<FormState>>
}

/** The tickers, with the presets for the chosen strategy. */
export function TickersSection({ form, set, setForm }: SectionProps) {
  const { strategy } = form
  const portfolio = strategy === 'trend_portfolio'
  const activePair = PAIR_PRESETS.find((p) => p.y === form.pairY && p.x === form.pairX)
  const portfolioList = parseTickerList(form.portfolioTickers)
  const activePortfolio = PORTFOLIO_PRESETS.find((p) => p.tickers.join(' ') === portfolioList.join(' '))
  return (
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
              <Chip
                key={`${p.y}/${p.x}`}
                title={p.note}
                active={p === activePair}
                onClick={() => setForm((f) => ({ ...f, pairY: p.y, pairX: p.x }))}
              >
                {p.y}/{p.x}
              </Chip>
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
              <Chip
                key={p.label}
                title={`${p.tickers.join(', ')}: ${p.note}`}
                active={p === activePortfolio}
                onClick={() => set('portfolioTickers')(p.tickers.join(' '))}
              >
                {p.label}
              </Chip>
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
                <Chip key={t} active={form.ticker === t} onClick={() => set('ticker')(t)}>
                  {t}
                </Chip>
              ))}
            </div>
          )}
        </>
      )}
    </Section>
  )
}

/** Start, end and the optional split date. */
export function PeriodSection({ form, set }: SectionProps) {
  return (
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
            <Chip onClick={() => set('split')('')}>clear</Chip>
          )}
        </div>
      </Field>
    </Section>
  )
}

/** Lookback, entry and exit thresholds or signal frequency, and the pairs model. */
export function SignalSection({ form, set }: SectionProps) {
  const { strategy } = form
  const trendLike = isTrendLike(strategy)
  return (
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
              <Chip key={m} active={form.L === days} onClick={() => set('L')(days)}>
                {m}m
              </Chip>
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
  )
}

/** Costs, execution lag, borrow fee, the long-only switch and the risk-free rate. */
export function ExecutionSection({ form, set }: SectionProps) {
  const { strategy } = form
  return (
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
  )
}

/** The volatility-targeting overlay; always on for the trend portfolio. */
export function VolTargetingSection({ form, set }: SectionProps) {
  const portfolio = form.strategy === 'trend_portfolio'
  return (
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
  )
}

/** The walk-forward's windows, sent by its own button. */
export function WalkForwardSection({ form, set }: SectionProps) {
  return (
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
  )
}

/** Paths, horizon, block length, seed and drift. */
export function MonteCarloSection({ form, set }: SectionProps) {
  return (
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
  )
}
