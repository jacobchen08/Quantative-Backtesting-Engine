import { describe, expect, it } from 'vitest'
import {
  buildRequests,
  buildWalkForward,
  defaultForm,
  isoDate,
  isTrendLike,
  L_DEFAULT,
  parseTickerList,
  PORTFOLIO_PRESETS,
  withStrategy,
  type FormState,
} from './form'
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
    expect(withStrategy(back, 'trend_portfolio').L).toBe('252')
  })

  it('defaults the Phase 7 settings: rolling OLS, δ = 1e-6, the four trend ETFs at 40% each, a fixed block length', () => {
    const f = defaultForm(TODAY)
    expect([f.pairsModel, f.kalmanDelta]).toEqual(['ols', '1e-6'])
    expect(parseTickerList(f.portfolioTickers)).toEqual(['SPY', 'QQQ', 'GLD', 'TLT'])
    expect(f.portfolioVolPct).toBe('40')
    expect([f.mcBlockLen, f.mcBlockAuto]).toEqual(['10', false])
  })

  it('splits ticker lists on commas and spaces, in the order typed', () => {
    expect(parseTickerList('spy, qqq  gld;tlt\n IEF ')).toEqual(['SPY', 'QQQ', 'GLD', 'TLT', 'IEF'])
    expect(parseTickerList(' , ')).toEqual([])
    expect(PORTFOLIO_PRESETS.map((p) => p.tickers.length)).toEqual([4, 10])
    expect(PORTFOLIO_PRESETS[1].tickers).toEqual(['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'DBC', 'VNQ'])
    expect(['trend', 'trend_portfolio', 'pairs', 'mean_reversion'].map((s) => isTrendLike(s as never))).toEqual([
      true,
      true,
      false,
      false,
    ])
  })

  it('formats local dates as YYYY-MM-DD', () => {
    expect(isoDate(new Date(2026, 0, 5))).toBe('2026-01-05')
    expect(isoDate(new Date(2026, 11, 31))).toBe('2026-12-31')
  })
})

describe('only the controls that apply are sent (§13)', () => {
  it('pairs: z thresholds, hedge mode and model, no signal frequency', () => {
    const { backtest } = built(form())
    expect(keys(backtest)).toEqual(
      ['L', 'borrow_bps', 'cost_bps', 'end', 'execution_lag', 'hedge_mode', 'pairs_model', 'risk_free', 'start', 'strategy', 'tickers', 'z_entry', 'z_exit'].sort(),
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
      pairs_model: 'ols',
      cost_bps: 10,
      borrow_bps: 0,
      execution_lag: 1,
      risk_free: 'tbill',
    })
  })

  it('pairs: δ is sent only with the Kalman model (§5.7)', () => {
    expect(built(form({ kalmanDelta: '' })).backtest).not.toHaveProperty('kalman_delta') // ignored, not checked
    const kalman = built(form({ pairsModel: 'kalman', kalmanDelta: ' 2.5e-6 ' })).backtest
    expect(kalman).toMatchObject({ pairs_model: 'kalman', kalman_delta: 2.5e-6 })
    expect(built(form({ pairsModel: 'kalman' })).backtest.kalman_delta).toBe(1e-6) // the default
    expect(errorOf(form({ pairsModel: 'kalman', kalmanDelta: '' }))).toBe('Kalman δ is required.')
    for (const strategy of ['mean_reversion', 'trend', 'trend_portfolio'] as const) {
      const other = built(withStrategy(form({ pairsModel: 'kalman' }), strategy)).backtest
      expect(other).not.toHaveProperty('pairs_model')
      expect(other).not.toHaveProperty('kalman_delta')
    }
  })

  it('trend portfolio: a ticker list, signal frequency and the per-asset vol settings (§5.8)', () => {
    const f = withStrategy(form({ portfolioTickers: ' spy, qqq\nGLD  tlt ', signalFrequency: 'daily' }), 'trend_portfolio')
    const { backtest } = built(f)
    expect(backtest).toEqual({
      strategy: 'trend_portfolio',
      tickers: ['SPY', 'QQQ', 'GLD', 'TLT'],
      start: '2015-01-01',
      end: TODAY,
      L: 252,
      signal_frequency: 'daily',
      cost_bps: 10,
      borrow_bps: 0,
      allow_short: true,
      execution_lag: 1,
      risk_free: 'tbill',
      vol_target: 0.4, // always sent: the overlay can't be off (§5.8)
      vol_halflife: 42,
      vol_refresh: 'weekly',
      max_leverage: 2,
    })
    expect(f.volOn).toBe(false) // the single-asset overlay switch plays no part
    const tuned = built({ ...f, portfolioVolPct: '25', volTargetPct: '12', maxLeverage: '3', longOnly: true }).backtest
    expect(tuned).toMatchObject({ vol_target: 0.25, max_leverage: 3, allow_short: false })
    expect(errorOf({ ...f, portfolioTickers: ' , ' })).toBe('Tickers is required.')
    expect(errorOf({ ...f, portfolioVolPct: '' })).toBe('Per-asset vol target is required.')
    // The count (2-12) and uniqueness are the API's to check, like every range.
    expect(built({ ...f, portfolioTickers: 'SPY' }).backtest.tickers).toEqual(['SPY'])
  })

  it('the overlay settings of one strategy are not checked for the other', () => {
    const portfolio = withStrategy(form({ volOn: true, volTargetPct: '' }), 'trend_portfolio')
    expect(buildRequests(portfolio).ok).toBe(true)
    expect(buildRequests(form({ portfolioVolPct: '', portfolioTickers: '' })).ok).toBe(true)
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

  it('sends block_len "auto" instead of a number when automatic is on (§10.8)', () => {
    expect(built(form({ mcBlockAuto: true, mcBlockLen: '' })).monteCarlo.block_len).toBe('auto') // not checked
    expect(built(form({ mcBlockLen: '21' })).monteCarlo.block_len).toBe(21)
    expect(errorOf(form({ mcBlockLen: '' }))).toBe('Block length is required.')
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

describe('the walk-forward request (§9.2, §13)', () => {
  function walkForward(f: FormState) {
    const result = buildWalkForward(f)
    if (!result.ok) throw new Error(`expected a valid form, got: ${result.error}`)
    return result.request
  }

  it('defaults to 756 training and 63 test days, rolling', () => {
    const f = defaultForm(TODAY)
    expect([f.wfTrainDays, f.wfTestDays, f.wfAnchored]).toEqual(['756', '63', false])
  })

  it('is the backtest request plus the fold windows, on the default grid', () => {
    const f = form({ wfTrainDays: '504', wfTestDays: '21', wfAnchored: true })
    expect(walkForward(f)).toEqual({ ...built(f).backtest, train_days: 504, test_days: 21, anchored: true })
    expect(walkForward(f)).not.toHaveProperty('L_values')
    expect(walkForward(f)).not.toHaveProperty('z_entry_values')
  })

  it('sends only the controls that apply, like the backtest', () => {
    const trend = walkForward(withStrategy(form({ ticker: 'TLT' }), 'trend'))
    expect(trend).toMatchObject({ strategy: 'trend', tickers: ['TLT'], L: 252, signal_frequency: 'monthly' })
    for (const hidden of ['z_entry', 'z_exit', 'hedge_mode', 'vol_target']) expect(trend).not.toHaveProperty(hidden)
    const portfolio = walkForward(withStrategy(form(), 'trend_portfolio'))
    expect(portfolio).toMatchObject({ strategy: 'trend_portfolio', tickers: ['SPY', 'QQQ', 'GLD', 'TLT'], vol_target: 0.4 })
  })

  it('reports a blank or non-numeric window by its label', () => {
    expect(buildWalkForward(form({ wfTrainDays: '' }))).toEqual({ ok: false, error: 'Training days is required.' })
    expect(buildWalkForward(form({ wfTestDays: 'x' }))).toEqual({ ok: false, error: 'Test days must be a number.' })
    expect(buildWalkForward(form({ L: '' }))).toEqual({ ok: false, error: 'Lookback L is required.' })
  })

  it('never blocks a Run, which does not send it', () => {
    expect(buildRequests(form({ wfTrainDays: '', wfTestDays: 'x' })).ok).toBe(true)
    expect(built(form())).not.toHaveProperty('walkForward')
  })
})
