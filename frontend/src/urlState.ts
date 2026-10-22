// URL state (§13). Run writes the sidebar settings that differ from the defaults to the
// query string; loading a URL fills the sidebar from it and sends nothing. The URL holds settings
// only, never results. Parameters are FormState's own keys, so a URL reads like the form.
//
// Each parameter is checked only for its shape (a number, a date, a ticker, one of the allowed
// values), as the form itself is: ranges are left to the API, whose 422 detail names the rule.
import { STRATEGIES, withStrategy, type FormState } from './form'

type Kind =
  | { type: 'enum'; values: readonly string[] }
  | { type: 'number' }
  | { type: 'bool' }
  | { type: 'date'; optional?: boolean }
  | { type: 'ticker' }
  | { type: 'tickers' }
  | { type: 'seed' }

const num = { type: 'number' } as const
const bool = { type: 'bool' } as const
const ticker = { type: 'ticker' } as const

/** Every FormState key, in the order they are written to the URL. */
export const URL_FIELDS: { readonly [K in keyof FormState]: Kind } = {
  strategy: { type: 'enum', values: STRATEGIES.map((s) => s.value) },
  ticker,
  pairY: ticker,
  pairX: ticker,
  portfolioTickers: { type: 'tickers' },
  pairsModel: { type: 'enum', values: ['ols', 'kalman'] },
  kalmanDelta: num,
  portfolioVolPct: num,
  start: { type: 'date' },
  end: { type: 'date' },
  split: { type: 'date', optional: true },
  L: num,
  zEntry: num,
  zExit: num,
  signalFrequency: { type: 'enum', values: ['monthly', 'daily'] },
  hedgeMode: { type: 'enum', values: ['fixed_at_entry', 'rolling'] },
  costBps: num,
  borrowBps: num,
  longOnly: bool,
  lag: num,
  rfMode: { type: 'enum', values: ['tbill', 'constant'] },
  rfPct: num,
  volOn: bool,
  volTargetPct: num,
  volHalflife: num,
  volRefresh: { type: 'enum', values: ['daily', 'weekly', 'monthly'] },
  maxLeverage: num,
  mcPaths: num,
  mcHorizon: num,
  mcBlockLen: num,
  mcBlockAuto: bool,
  mcSeed: { type: 'seed' },
  mcZeroDrift: bool,
  wfTrainDays: num,
  wfTestDays: num,
  wfAnchored: bool,
}

const KEYS = Object.keys(URL_FIELDS) as (keyof FormState)[]
const TICKER = /^[A-Za-z0-9.^=-]{1,15}$/ // the API's pattern (§11.2), before uppercasing

function isDate(s: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) return false
  const d = new Date(`${s}T00:00:00Z`)
  return !Number.isNaN(d.getTime()) && d.toISOString().slice(0, 10) === s // rejects 2024-02-30
}

/** The form value a URL string stands for, or the reason it is rejected. */
function parseValue(kind: Kind, raw: string): { ok: true; value: string | boolean } | { ok: false; why: string } {
  switch (kind.type) {
    case 'enum':
      return kind.values.includes(raw) ? { ok: true, value: raw } : { ok: false, why: `expected one of ${kind.values.join(', ')}` }
    case 'number':
      return raw.trim() !== '' && Number.isFinite(Number(raw)) ? { ok: true, value: raw } : { ok: false, why: 'not a number' }
    case 'bool':
      return raw === 'true' || raw === 'false' ? { ok: true, value: raw === 'true' } : { ok: false, why: 'expected true or false' }
    case 'date':
      if (kind.optional && raw === '') return { ok: true, value: '' }
      return isDate(raw) ? { ok: true, value: raw } : { ok: false, why: 'not a date (YYYY-MM-DD)' }
    case 'ticker':
      return TICKER.test(raw) ? { ok: true, value: raw } : { ok: false, why: 'not a ticker' }
    case 'tickers': {
      const tickers = raw.split(/[\s,;]+/).filter((t) => t !== '')
      if (tickers.length === 0 || !tickers.every((t) => TICKER.test(t))) return { ok: false, why: 'not a ticker list' }
      return { ok: true, value: tickers.join(' ') }
    }
    case 'seed':
      return /^\d+$/.test(raw) && Number.isSafeInteger(Number(raw))
        ? { ok: true, value: raw }
        : { ok: false, why: 'not a whole number below 2^53' }
  }
}

/** Defaults for comparison: L's default depends on the strategy (§13), so it follows form.strategy. */
function baseline(form: FormState, defaults: FormState): FormState {
  return form.strategy === defaults.strategy ? defaults : withStrategy(defaults, form.strategy)
}

/** "?strategy=trend&ticker=SYNT…" with every setting that differs from the defaults, or "" if none does. */
export function formToSearch(form: FormState, defaults: FormState): string {
  const base = baseline(form, defaults)
  const params = new URLSearchParams()
  for (const key of KEYS) {
    const reference = key === 'strategy' ? defaults : base // base already has form's strategy
    if (form[key] !== reference[key]) params.set(key, String(form[key]))
  }
  const query = params.toString()
  return query === '' ? '' : `?${query}`
}

export type UrlForm = { form: FormState; notices: string[] }

/**
 * The sidebar a query string describes: the defaults, with each valid parameter applied. An unknown
 * or invalid parameter is ignored with a notice naming it. The strategy is applied first, so L
 * starts from that strategy's default when the URL leaves it out.
 */
export function searchToForm(search: string, defaults: FormState): UrlForm {
  const params = new URLSearchParams(search)
  const notices: string[] = []
  const values = new Map<keyof FormState, string | boolean>()
  for (const [key, raw] of params) {
    if (!(key in URL_FIELDS)) {
      notices.push(`Ignored unknown URL parameter "${key}".`)
      continue
    }
    const k = key as keyof FormState
    const parsed = parseValue(URL_FIELDS[k], raw)
    if (parsed.ok) values.set(k, parsed.value)
    else notices.push(`Ignored URL parameter ${key}=${JSON.stringify(raw)}: ${parsed.why}.`)
  }
  const strategy = values.get('strategy') as FormState['strategy'] | undefined
  let form = strategy === undefined ? defaults : withStrategy(defaults, strategy)
  for (const [key, value] of values) {
    if (key !== 'strategy') form = { ...form, [key]: value }
  }
  return { form, notices }
}
