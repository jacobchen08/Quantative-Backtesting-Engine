import { describe, expect, it } from 'vitest'
import { buildRequests, defaultForm, withStrategy, type FormState } from './form'
import { formToSearch, searchToForm, URL_FIELDS } from './urlState'

const TODAY = '2026-10-19'
const DEFAULTS = defaultForm(TODAY)

const roundTrip = (form: FormState) => searchToForm(formToSearch(form, DEFAULTS), DEFAULTS)

describe('formToSearch', () => {
  it('is empty when every setting is the default', () => {
    expect(formToSearch(DEFAULTS, DEFAULTS)).toBe('')
  })

  it('writes only the settings that differ from the defaults', () => {
    const search = formToSearch({ ...DEFAULTS, pairY: 'SYNY', pairX: 'SYNX', zEntry: '2.5' }, DEFAULTS)
    expect(Object.fromEntries(new URLSearchParams(search))).toEqual({ pairY: 'SYNY', pairX: 'SYNX', zEntry: '2.5' })
    expect(search.startsWith('?')).toBe(true)
  })

  it("compares L with the chosen strategy's default", () => {
    const trend = withStrategy(DEFAULTS, 'trend')
    expect(formToSearch(trend, DEFAULTS)).toBe('?strategy=trend') // L = 252 is trend's default
    expect(formToSearch({ ...trend, L: '126' }, DEFAULTS)).toBe('?strategy=trend&L=126')
  })

  it('writes booleans as true/false and encodes special characters', () => {
    const search = formToSearch({ ...DEFAULTS, volOn: true, portfolioTickers: 'SPY QQQ', ticker: 'BRK.B' }, DEFAULTS)
    const params = new URLSearchParams(search)
    expect(params.get('volOn')).toBe('true')
    expect(params.get('portfolioTickers')).toBe('SPY QQQ')
    expect(params.get('ticker')).toBe('BRK.B')
  })

  it('never writes results, only FormState keys', () => {
    const search = formToSearch({ ...DEFAULTS, mcSeed: '42', split: '2020-01-02' }, DEFAULTS)
    for (const key of new URLSearchParams(search).keys()) expect(key in URL_FIELDS).toBe(true)
  })
})

describe('searchToForm', () => {
  it('fills the sidebar from the query string', () => {
    const { form, notices } = searchToForm('?strategy=trend&ticker=SYNT&start=2012-01-03&end=2019-12-31&volOn=true', DEFAULTS)
    expect(notices).toEqual([])
    expect(form).toEqual({
      ...withStrategy(DEFAULTS, 'trend'),
      ticker: 'SYNT',
      start: '2012-01-03',
      end: '2019-12-31',
      volOn: true,
    })
  })

  it("starts L from the URL strategy's default, whatever the parameter order", () => {
    expect(searchToForm('?strategy=trend', DEFAULTS).form.L).toBe('252')
    expect(searchToForm('?L=126&strategy=trend', DEFAULTS).form.L).toBe('126')
  })

  it('ignores an unknown parameter with a notice naming it', () => {
    const { form, notices } = searchToForm('?foo=1&zEntry=2.5', DEFAULTS)
    expect(form).toEqual({ ...DEFAULTS, zEntry: '2.5' })
    expect(notices).toEqual(['Ignored unknown URL parameter "foo".'])
  })

  it.each([
    ['strategy', 'momentum', 'expected one of mean_reversion, pairs, trend, trend_portfolio'],
    ['L', 'abc', 'not a number'],
    ['L', '', 'not a number'],
    ['costBps', 'Infinity', 'not a number'],
    ['start', '2024-02-30', 'not a date (YYYY-MM-DD)'],
    ['end', '2024-1-5', 'not a date (YYYY-MM-DD)'],
    ['volOn', 'yes', 'expected true or false'],
    ['ticker', 'KO;DROP', 'not a ticker'],
    ['pairY', 'ABCDEFGHIJKLMNOP', 'not a ticker'],
    ['portfolioTickers', 'SPY,<x>', 'not a ticker list'],
    ['mcSeed', '-1', 'not a whole number below 2^53'],
    ['mcSeed', '9007199254740992', 'not a whole number below 2^53'],
    ['hedgeMode', 'static', 'expected one of fixed_at_entry, rolling'],
  ])('ignores %s=%j with a notice: %s', (key, value, why) => {
    const { form, notices } = searchToForm(new URLSearchParams({ [key]: value }).toString(), DEFAULTS)
    expect(form).toEqual(DEFAULTS)
    expect(notices).toEqual([`Ignored URL parameter ${key}=${JSON.stringify(value)}: ${why}.`])
  })

  it('accepts an empty split (the default 70% split) and a valid seed', () => {
    const { form, notices } = searchToForm('?split=&mcSeed=9007199254740991', DEFAULTS)
    expect(notices).toEqual([])
    expect(form.split).toBe('')
    expect(form.mcSeed).toBe('9007199254740991')
  })

  it('normalizes a ticker list to space-separated', () => {
    expect(searchToForm('?portfolioTickers=SPY,QQQ;GLD', DEFAULTS).form.portfolioTickers).toBe('SPY QQQ GLD')
  })

  it('keeps every valid parameter when another is invalid', () => {
    const { form, notices } = searchToForm('?zEntry=x&zExit=0.25', DEFAULTS)
    expect(form).toEqual({ ...DEFAULTS, zExit: '0.25' })
    expect(notices).toHaveLength(1)
  })
})

describe('URL round trips', () => {
  it.each<[string, Partial<FormState>]>([
    ['pairs on synthetic tickers', { pairY: 'SYNY', pairX: 'SYNX', start: '2012-01-03', end: '2019-12-31', split: '2017-06-01' }],
    ['the Kalman model', { pairsModel: 'kalman', kalmanDelta: '1e-5', hedgeMode: 'rolling' }],
    ['every Monte Carlo and walk-forward setting', {
      mcPaths: '2000', mcHorizon: '126', mcBlockLen: '5', mcBlockAuto: true, mcSeed: '42', mcZeroDrift: true,
      wfTrainDays: '504', wfTestDays: '42', wfAnchored: true,
    }],
    ['costs, rates and vol targeting', {
      costBps: '2.5', borrowBps: '50', lag: '0', rfMode: 'constant', rfPct: '3.5',
      volOn: true, volTargetPct: '12', volHalflife: '21', volRefresh: 'monthly', maxLeverage: '3',
    }],
  ])('%s', (_name, overrides) => {
    const form = { ...DEFAULTS, ...overrides }
    expect(roundTrip(form)).toEqual({ form, notices: [] })
  })

  it.each(['mean_reversion', 'trend', 'trend_portfolio'] as const)('the %s strategy with a changed L', (strategy) => {
    const form = {
      ...withStrategy(DEFAULTS, strategy),
      L: '42',
      ticker: 'SYNT',
      portfolioTickers: 'SPY GLD',
      longOnly: true,
      signalFrequency: 'daily' as const,
    }
    expect(roundTrip(form)).toEqual({ form, notices: [] })
  })

  it('builds the same requests after the round trip', () => {
    const form = { ...withStrategy(DEFAULTS, 'trend'), ticker: 'SYNT', start: '2012-01-03', end: '2019-12-31', mcSeed: '7' }
    expect(buildRequests(roundTrip(form).form)).toEqual(buildRequests(form))
  })
})
