import { describe, expect, it } from 'vitest'
import { buildRequests, defaultForm, isoDate, L_DEFAULT, withStrategy, type FormState } from './form'
import type { BacktestRequest } from './types'

const TODAY = '2026-10-16'

function form(overrides: Partial<FormState> = {}): FormState {
  return { ...defaultForm(TODAY), ...overrides }
}

function built(f: FormState) {
  const result = buildRequests(f)
  if (!result.ok) throw new Error(`expected a valid form, got: ${result.error}`)
  return result
}

function errorOf(f: FormState): string {
  const result = buildRequests(f)
  if (result.ok) throw new Error('expected the form to be rejected')
  return result.error
}

const keys = (request: BacktestRequest) => Object.keys(request).sort()

describe('defaults', () => {
  it('starts on a KO/PEP pairs run ending today, with the API defaults', () => {
    const f = defaultForm(TODAY)
    expect(f.end).toBe(TODAY)
    expect(f.strategy).toBe('pairs')
    expect([f.pairY, f.pairX]).toEqual(['KO', 'PEP'])
    expect(f.L).toBe('60')
    expect([f.zEntry, f.zExit, f.costBps, f.lag]).toEqual(['2.0', '0.5', '10', '1'])
    expect([f.borrowBps, f.longOnly]).toEqual(['0', false])
    expect(f.volOn).toBe(false)
    expect(f.mcSeed).toBe('')
  })

  it('switches L to the new strategy’s default (§13)', () => {
    const trend = withStrategy(form({ L: '90' }), 'trend')
    expect(trend.strategy).toBe('trend')
    expect(trend.L).toBe(String(L_DEFAULT.trend))
    expect(trend.L).toBe('252')
    const back = withStrategy(trend, 'mean_reversion')
    expect(back.L).toBe('60')
    expect(back.pairY).toBe('KO') // everything else is kept
  })

  it('formats local dates as YYYY-MM-DD', () => {
    expect(isoDate(new Date(2026, 0, 5))).toBe('2026-01-05')
    expect(isoDate(new Date(2026, 11, 31))).toBe('2026-12-31')
  })
})

describe('only the controls that apply are sent (§13)', () => {
  it('pairs: z thresholds and hedge mode, no signal frequency', () => {
    const { backtest } = built(form())
    expect(keys(backtest)).toEqual(
      ['L', 'borrow_bps', 'cost_bps', 'end', 'execution_lag', 'hedge_mode', 'risk_free', 'start', 'strategy', 'tickers', 'z_entry', 'z_exit'].sort(),
    )
    expect(backtest).toEqual({
      strategy: 'pairs',
      tickers: ['KO', 'PEP'],
      start: '2015-01-01',
      end: TODAY,
      L: 60,
      z_entry: 2,
      z_exit: 0.5,
      hedge_mode: 'fixed_at_entry',
      cost_bps: 10,
      borrow_bps: 0,
      execution_lag: 1,
      risk_free: 'tbill',
    })
  })

  it('mean reversion: one ticker, z thresholds, no hedge mode or signal frequency', () => {
    const { backtest } = built(form({ strategy: 'mean_reversion', ticker: 'XOM' }))
    expect(backtest.tickers).toEqual(['XOM'])
    expect(backtest.z_entry).toBe(2)
    expect(backtest).not.toHaveProperty('hedge_mode')
    expect(backtest).not.toHaveProperty('signal_frequency')
    expect(backtest.allow_short).toBe(true)
  })

  it('trend: signal frequency, no z thresholds or hedge mode', () => {
    const { backtest } = built(withStrategy(form({ ticker: 'GLD', signalFrequency: 'daily' }), 'trend'))
    expect(backtest.tickers).toEqual(['GLD'])
    expect(backtest.L).toBe(252)
    expect(backtest.signal_frequency).toBe('daily')
    for (const hidden of ['z_entry', 'z_exit', 'hedge_mode']) expect(backtest).not.toHaveProperty(hidden)
    expect(backtest.allow_short).toBe(true)
  })

  it('the long-only switch is sent for mean reversion and trend, never for pairs (§5.9)', () => {
    const longOnly = form({ longOnly: true })
    expect(built(longOnly).backtest).not.toHaveProperty('allow_short') // pairs always holds a short leg
    expect(built(withStrategy(longOnly, 'mean_reversion')).backtest.allow_short).toBe(false)
    expect(built(withStrategy(longOnly, 'trend')).backtest.allow_short).toBe(false)
  })

  it('the borrow fee is sent for every strategy, in bps a year', () => {
    for (const strategy of ['pairs', 'mean_reversion', 'trend'] as const) {
      expect(built(withStrategy(form({ borrowBps: '150' }), strategy)).backtest.borrow_bps).toBe(150)
    }
  })

  it('does not validate hidden controls either', () => {
    const trend = withStrategy(form({ zEntry: '', zExit: 'oops' }), 'trend')
    expect(buildRequests(trend).ok).toBe(true)
    expect(buildRequests(form({ volOn: false, volTargetPct: '', maxLeverage: '' })).ok).toBe(true)
    expect(buildRequests(form({ rfMode: 'tbill', rfPct: '' })).ok).toBe(true)
  })

  it('vol targeting off sends no overlay field at all', () => {
    const { backtest } = built(form())
    for (const field of ['vol_target', 'vol_halflife', 'vol_refresh', 'max_leverage']) {
      expect(backtest).not.toHaveProperty(field)
    }
  })

  it('vol targeting on sends the target as a decimal', () => {
    const { backtest } = built(
      form({ volOn: true, volTargetPct: '12.5', volHalflife: '21', volRefresh: 'monthly', maxLeverage: '3' }),
    )
    expect(backtest).toMatchObject({ vol_target: 0.125, vol_halflife: 21, vol_refresh: 'monthly', max_leverage: 3 })
  })

  it('a constant risk-free rate is sent as an annual decimal', () => {
    expect(built(form({ rfMode: 'constant', rfPct: '3' })).backtest.risk_free).toBe(0.03)
    expect(built(form({ rfMode: 'constant', rfPct: '-0.5' })).backtest.risk_free).toBe(-0.005)
  })

  it('sends the split date only when one is set', () => {
    expect(built(form()).backtest).not.toHaveProperty('split_date')
    expect(built(form({ split: ' 2022-01-03 ' })).backtest.split_date).toBe('2022-01-03')
  })

  it('trims and uppercases tickers', () => {
    expect(built(form({ pairY: ' xom ', pairX: 'cvx' })).backtest.tickers).toEqual(['XOM', 'CVX'])
  })
})

describe('malformed input is caught before it can become null', () => {
  it('reports a blank number by its label', () => {
    expect(errorOf(form({ L: '' }))).toBe('Lookback L is required.')
    expect(errorOf(form({ costBps: '  ' }))).toBe('Cost is required.')
    expect(errorOf(form({ borrowBps: '' }))).toBe('Borrow fee is required.')
    expect(errorOf(form({ mcPaths: '' }))).toBe('Paths is required.')
  })

  it('reports a non-numeric value', () => {
    expect(errorOf(form({ zEntry: 'abc' }))).toBe('z entry must be a number.')
    expect(errorOf(form({ volOn: true, volTargetPct: 'ten' }))).toBe('Vol target must be a number.')
  })

  it('reports a missing ticker or date', () => {
    expect(errorOf(form({ pairX: ' ' }))).toBe('Ticker X is required.')
    expect(errorOf(form({ start: '' }))).toBe('Start date is required.')
  })

  it('leaves range checks to the API, which names the rule in its 422', () => {
    expect(built(form({ L: '9999', zEntry: '99' })).backtest).toMatchObject({ L: 9999, z_entry: 99 })
  })

  it('never puts NaN in a request: the JSON round trip is lossless', () => {
    const { backtest, monteCarlo } = built(form({ volOn: true }))
    expect(JSON.parse(JSON.stringify(backtest))).toEqual(backtest)
    expect(JSON.parse(JSON.stringify(monteCarlo))).toEqual(monteCarlo)
  })
})

describe('sensitivity and Monte Carlo requests', () => {
  it('the sensitivity request is the backtest request: the default grid, no overrides', () => {
    const { backtest, sensitivity } = built(form())
    expect(sensitivity).toEqual(backtest)
    expect(sensitivity).not.toHaveProperty('L_values')
    expect(sensitivity).not.toHaveProperty('z_entry_values')
  })

  it('Monte Carlo projects the same backtest’s returns', () => {
    const { backtest, monteCarlo } = built(form())
    expect(monteCarlo).toEqual({
      source: 'strategy',
      backtest,
      n_paths: 10000,
      horizon: 252,
      block_len: 10,
      zero_drift: false,
    })
  })

  it('omits a blank seed so the API draws one', () => {
    expect(built(form()).monteCarlo).not.toHaveProperty('seed')
    expect(built(form({ mcSeed: '42' })).monteCarlo.seed).toBe(42)
    expect(built(form({ mcSeed: '0', mcZeroDrift: true })).monteCarlo).toMatchObject({ seed: 0, zero_drift: true })
  })

  it('accepts seeds up to 2^53 - 1 and rejects anything JSON would round (§10.6)', () => {
    expect(built(form({ mcSeed: '9007199254740991' })).monteCarlo.seed).toBe(2 ** 53 - 1)
    for (const bad of ['9007199254740992', '-1', '1.5', '1e3', 'abc']) {
      expect(errorOf(form({ mcSeed: bad }))).toMatch(/^Seed must be a whole number/)
    }
  })
})
