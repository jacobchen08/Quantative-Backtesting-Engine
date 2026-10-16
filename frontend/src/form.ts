// Sidebar state and the requests built from it (§13). Inputs are kept as the strings the user
// typed and parsed only when Run is pressed. A blank or non-numeric field is reported here, so it
// can't reach the API as null (JSON.stringify turns NaN into null, and a null L means "default").
// Ranges are left to the API, whose 422 detail names the rule.
import type {
  BacktestRequest,
  HedgeMode,
  MonteCarloRequest,
  SensitivityRequest,
  SignalFrequency,
  Strategy,
  VolRefresh,
} from './types'

export const STRATEGIES: readonly { value: Strategy; label: string }[] = [
  { value: 'mean_reversion', label: 'Mean reversion' },
  { value: 'pairs', label: 'Pairs' },
  { value: 'trend', label: 'Trend' },
]

export const STRATEGY_LABEL: Record<Strategy, string> = {
  mean_reversion: 'Mean reversion',
  pairs: 'Pairs',
  trend: 'Trend',
}

/** Lookback defaults in trading days (§11.2). */
export const L_DEFAULT: Record<Strategy, number> = { mean_reversion: 60, pairs: 60, trend: 252 }

/** Trend lookback presets in months; 21 trading days a month, as in the §9 grid. */
export const TREND_MONTHS = [1, 3, 6, 12, 18] as const
export const TRADING_DAYS_PER_MONTH = 21

export const PAIR_PRESETS: readonly { y: string; x: string; note: string }[] = [
  { y: 'KO', x: 'PEP', note: 'Consumer staples rivals' },
  { y: 'XOM', x: 'CVX', note: 'Oil majors' },
  { y: 'GOOGL', x: 'GOOG', note: 'Near-collinear share classes: exercises the degenerate-window guards' },
  { y: 'NVDA', x: 'AMD', note: 'Likely not cointegrated: demonstrates the warnings' },
]

/** Diversified ETFs, where trend following is usually applied. */
export const TREND_PRESETS = ['SPY', 'QQQ', 'GLD', 'TLT'] as const

export type FormState = {
  strategy: Strategy
  ticker: string // mean reversion and trend
  pairY: string // pairs: Y = tickers[0]
  pairX: string // pairs: X = tickers[1]
  start: string // YYYY-MM-DD
  end: string
  split: string // '' = the default 70% split
  L: string // trading days
  zEntry: string
  zExit: string
  signalFrequency: SignalFrequency
  hedgeMode: HedgeMode
  costBps: string // one-way, per leg
  lag: string // trading days
  rfMode: 'tbill' | 'constant'
  rfPct: string // annual %, used when rfMode is 'constant'
  volOn: boolean
  volTargetPct: string // annual %
  volHalflife: string // trading days
  volRefresh: VolRefresh
  maxLeverage: string
  mcPaths: string
  mcHorizon: string // trading days
  mcBlockLen: string // trading days
  mcSeed: string // '' = the API draws a seed and returns it
  mcZeroDrift: boolean
}

export function defaultForm(today: string): FormState {
  return {
    strategy: 'pairs',
    ticker: 'SPY',
    pairY: 'KO',
    pairX: 'PEP',
    start: '2015-01-01',
    end: today,
    split: '',
    L: String(L_DEFAULT.pairs),
    zEntry: '2.0',
    zExit: '0.5',
    signalFrequency: 'monthly',
    hedgeMode: 'fixed_at_entry',
    costBps: '10',
    lag: '1',
    rfMode: 'tbill',
    rfPct: '3',
    volOn: false,
    volTargetPct: '10',
    volHalflife: '42',
    volRefresh: 'weekly',
    maxLeverage: '2',
    mcPaths: '10000',
    mcHorizon: '252',
    mcBlockLen: '10',
    mcSeed: '',
    mcZeroDrift: false,
  }
}

/** Switching strategy resets L to the new strategy's default (§13). */
export function withStrategy(form: FormState, strategy: Strategy): FormState {
  return { ...form, strategy, L: String(L_DEFAULT[strategy]) }
}

/** The local calendar date as YYYY-MM-DD. The API clamps `end` to yesterday in New York anyway. */
export function isoDate(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

class FormError extends Error {}

function required(label: string, raw: string): string {
  const s = raw.trim()
  if (s === '') throw new FormError(`${label} is required.`)
  return s
}

function num(label: string, raw: string): number {
  const x = Number(required(label, raw))
  if (!Number.isFinite(x)) throw new FormError(`${label} must be a number.`)
  return x
}

/** A percent field converted to a decimal: "10" -> 0.1. */
function pct(label: string, raw: string): number {
  return num(label, raw) / 100
}

function ticker(label: string, raw: string): string {
  return required(label, raw).toUpperCase()
}

/** The seed must survive JSON: JavaScript numbers are exact only below 2^53 (§10.6). */
function seed(raw: string): number | null {
  const s = raw.trim()
  if (s === '') return null
  const x = Number(s)
  if (!/^\d+$/.test(s) || !Number.isSafeInteger(x)) {
    throw new FormError('Seed must be a whole number from 0 to 2^53 − 1, or blank for a random seed.')
  }
  return x
}

export type BuiltRequests = {
  backtest: BacktestRequest
  sensitivity: SensitivityRequest // the default grid (§9): no overrides in the dashboard
  monteCarlo: MonteCarloRequest // the strategy's own returns over the full evaluation window
}

export type BuildResult = ({ ok: true } & BuiltRequests) | { ok: false; error: string }

/** The three requests a Run sends. Only fields that apply to the chosen strategy are included. */
export function buildRequests(form: FormState): BuildResult {
  try {
    const backtest = buildBacktest(form)
    const mcSeed = seed(form.mcSeed)
    const monteCarlo: MonteCarloRequest = {
      source: 'strategy',
      backtest,
      n_paths: num('Paths', form.mcPaths),
      horizon: num('Horizon', form.mcHorizon),
      block_len: num('Block length', form.mcBlockLen),
      zero_drift: form.mcZeroDrift,
      ...(mcSeed === null ? {} : { seed: mcSeed }),
    }
    return { ok: true, backtest, sensitivity: { ...backtest }, monteCarlo }
  } catch (error) {
    if (error instanceof FormError) return { ok: false, error: error.message }
    throw error
  }
}

function buildBacktest(form: FormState): BacktestRequest {
  const { strategy } = form
  const request: BacktestRequest = {
    strategy,
    tickers:
      strategy === 'pairs'
        ? [ticker('Ticker Y', form.pairY), ticker('Ticker X', form.pairX)]
        : [ticker('Ticker', form.ticker)],
    start: required('Start date', form.start),
    end: required('End date', form.end),
    L: num('Lookback L', form.L),
  }
  if (strategy === 'trend') {
    request.signal_frequency = form.signalFrequency
  } else {
    request.z_entry = num('z entry', form.zEntry)
    request.z_exit = num('z exit', form.zExit)
  }
  if (strategy === 'pairs') request.hedge_mode = form.hedgeMode
  request.cost_bps = num('Cost', form.costBps)
  request.execution_lag = num('Execution lag', form.lag)
  request.risk_free = form.rfMode === 'tbill' ? 'tbill' : pct('Constant risk-free rate', form.rfPct)
  if (form.volOn) {
    request.vol_target = pct('Vol target', form.volTargetPct)
    request.vol_halflife = num('Vol half-life', form.volHalflife)
    request.vol_refresh = form.volRefresh
    request.max_leverage = num('Max leverage', form.maxLeverage)
  }
  if (form.split.trim() !== '') request.split_date = form.split.trim()
  return request
}
