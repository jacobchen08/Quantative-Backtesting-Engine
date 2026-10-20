import { describe, expect, it } from 'vitest'
import { buildRequests, defaultForm, withStrategy, type FormState } from './form'
import { comparisonRows, isOtherEngine, percentField, resolvedToForm } from './history'
import { DASH } from './format'
import type { BacktestRequest, Metrics, StoredRun } from './types'

const CURRENT = { ...defaultForm('2026-10-19'), mcSeed: '42', wfTrainDays: '504' }

type Resolved = StoredRun['resolved']

/** What the API stores for a request with every default (§11.2), here for pairs KO/PEP. */
function resolved(overrides: Partial<Resolved> = {}): Resolved {
  return {
    strategy: 'pairs',
    tickers: ['KO', 'PEP'],
    start: '2015-01-02',
    end: '2025-12-31',
    L: 60,
    z_entry: 2,
    z_exit: 0.5,
    signal_frequency: null,
    cost_bps: 10,
    execution_lag: 1,
    hedge_mode: 'fixed_at_entry',
    pairs_model: 'ols',
    kalman_delta: null,
    borrow_bps: 0,
    allow_short: true,
    vol_target: null,
    vol_halflife: null,
    vol_refresh: null,
    max_leverage: null,
    split_date: '2022-09-14',
    risk_free: 'tbill',
    initial_capital: 100000,
    first_evaluated_date: '2015-03-31',
    rf_source: 'tbill (^IRX)',
    ...overrides,
  }
}

/** The backtest request the sidebar builds after loading a run. */
function rebuilt(r: Resolved, current: FormState = CURRENT): BacktestRequest {
  const built = buildRequests(resolvedToForm(r, current))
  if (!built.ok) throw new Error(built.error)
  return built.backtest
}

describe('percentField', () => {
  it('turns a decimal rate into the percent string the sidebar holds, without float noise', () => {
    expect(percentField(0.1)).toBe('10')
    expect(percentField(0.07)).toBe('7')
    expect(percentField(0.035)).toBe('3.5')
    expect(percentField(0.4)).toBe('40')
    expect(percentField(-0.005)).toBe('-0.5')
  })
})

describe('resolvedToForm: "Load settings" (§13 view 12)', () => {
  it("rebuilds the run's request for pairs", () => {
    expect(rebuilt(resolved())).toEqual({
      strategy: 'pairs',
      tickers: ['KO', 'PEP'],
      start: '2015-01-02',
      end: '2025-12-31',
      L: 60,
      z_entry: 2,
      z_exit: 0.5,
      hedge_mode: 'fixed_at_entry',
      pairs_model: 'ols',
      cost_bps: 10,
      borrow_bps: 0,
      execution_lag: 1,
      risk_free: 'tbill',
      split_date: '2022-09-14',
    })
  })

  it('keeps the Monte Carlo and walk-forward settings, which a run does not store', () => {
    const form = resolvedToForm(resolved(), CURRENT)
    expect(form.mcSeed).toBe('42')
    expect(form.wfTrainDays).toBe('504')
  })

  it('loads the Kalman model and a constant risk-free rate', () => {
    const r = resolved({ pairs_model: 'kalman', kalman_delta: 1e-5, hedge_mode: 'rolling', risk_free: 0.035, cost_bps: 2.5 })
    expect(rebuilt(r)).toMatchObject({ pairs_model: 'kalman', kalman_delta: 1e-5, hedge_mode: 'rolling', risk_free: 0.035, cost_bps: 2.5 })
  })

  it('loads trend with the overlay on and long-only', () => {
    const r = resolved({
      strategy: 'trend',
      tickers: ['SPY'],
      L: 126,
      z_entry: null,
      z_exit: null,
      signal_frequency: 'daily',
      hedge_mode: null,
      pairs_model: null,
      allow_short: false,
      borrow_bps: 25,
      vol_target: 0.12,
      vol_halflife: 21,
      vol_refresh: 'monthly',
      max_leverage: 3,
      execution_lag: 2,
    })
    expect(rebuilt(r)).toEqual({
      strategy: 'trend',
      tickers: ['SPY'],
      start: '2015-01-02',
      end: '2025-12-31',
      L: 126,
      signal_frequency: 'daily',
      cost_bps: 10,
      borrow_bps: 25,
      allow_short: false,
      execution_lag: 2,
      risk_free: 'tbill',
      vol_target: 0.12,
      vol_halflife: 21,
      vol_refresh: 'monthly',
      max_leverage: 3,
      split_date: '2022-09-14',
    })
  })

  it('turns the overlay off when the run had none, and keeps the hidden target value', () => {
    const on = { ...withStrategy(CURRENT, 'trend'), volOn: true, volTargetPct: '15' }
    const form = resolvedToForm(resolved({ strategy: 'mean_reversion', tickers: ['KO'], hedge_mode: null, pairs_model: null }), on)
    expect(form.volOn).toBe(false)
    expect(form.volTargetPct).toBe('15')
    expect(form.ticker).toBe('KO')
  })

  it('loads the trend portfolio with its per-asset target', () => {
    const r = resolved({
      strategy: 'trend_portfolio',
      tickers: ['SPY', 'QQQ', 'GLD'],
      L: 252,
      z_entry: null,
      z_exit: null,
      signal_frequency: 'monthly',
      hedge_mode: null,
      pairs_model: null,
      vol_target: 0.4,
      vol_halflife: 42,
      vol_refresh: 'weekly',
      max_leverage: 2,
    })
    const form = resolvedToForm(r, CURRENT)
    expect(form.portfolioTickers).toBe('SPY QQQ GLD')
    expect(form.portfolioVolPct).toBe('40')
    expect(rebuilt(r)).toMatchObject({ strategy: 'trend_portfolio', tickers: ['SPY', 'QQQ', 'GLD'], vol_target: 0.4 })
  })

  it('loads a run stored by an older engine, whose later fields are null, as their defaults', () => {
    const old = resolved({ borrow_bps: null, allow_short: null, pairs_model: null })
    const form = resolvedToForm(old, { ...CURRENT, borrowBps: '100', pairsModel: 'kalman' })
    expect(form.borrowBps).toBe('0')
    expect(form.pairsModel).toBe('ols')
    expect(rebuilt(old)).toEqual(rebuilt(resolved()))
    const oldTrend = resolved({ strategy: 'trend', tickers: ['SPY'], allow_short: null, hedge_mode: null, pairs_model: null })
    expect(resolvedToForm(oldTrend, { ...CURRENT, longOnly: true }).longOnly).toBe(false)
  })

  it('sets L from the run, not from the strategy default', () => {
    expect(resolvedToForm(resolved({ strategy: 'trend', tickers: ['SPY'], L: 378 }), CURRENT).L).toBe('378')
  })
})

describe('isOtherEngine', () => {
  it('flags a run whose engine version differs from the API', () => {
    expect(isOtherEngine('0.0.9', '0.1.0')).toBe(true)
    expect(isOtherEngine('0.1.0', '0.1.0')).toBe(false)
  })

  it('flags nothing while the current version is unknown', () => {
    expect(isOtherEngine('0.0.9', null)).toBe(false)
  })
})

function metrics(overrides: Partial<Metrics> = {}): Metrics {
  return {
    n_days: 2500,
    total_return: 0.25,
    cagr: 0.0228,
    ann_vol: 0.1,
    sharpe: 0.5,
    sharpe_ci95: [0.1, 0.9],
    sortino: 0.7,
    max_drawdown: -0.2,
    beta_spy: 0.05,
    corr_spy: 0.1,
    pct_days_in_market: 0.4,
    n_closed_trades: 30,
    n_open_trades: 1,
    win_rate: 0.6,
    avg_holding_days: 12,
    ann_turnover: 8,
    ann_cost_drag: 0.008,
    ann_borrow_drag: 0,
    avg_gross_exposure: 0.4,
    ...overrides,
  }
}

function run(id: number, overrides: Partial<StoredRun> = {}): StoredRun {
  return {
    id,
    created_at: '2026-10-19T12:00:00+00:00',
    engine_version: '0.1.0',
    resolved: resolved(),
    metrics: { is: metrics({ sharpe: 0.8 }), oos: metrics({ sharpe: -0.2 }), full: metrics() },
    benchmark_metrics: { is: metrics({ sharpe: 0.9 }), oos: metrics({ sharpe: 1.1 }), full: metrics({ sharpe: 0.95 }) },
    ...overrides,
  }
}

const row = (rows: ReturnType<typeof comparisonRows>, group: string, label: string) =>
  rows.find((r) => r.group === group && r.label === label)?.values

describe('comparisonRows', () => {
  it('has one column per run and follows the segment', () => {
    const runs = [run(1), run(2, { resolved: resolved({ L: 90 }) })]
    expect(row(comparisonRows(runs, 'is'), 'strategy', 'Sharpe')).toEqual(['0.80', '0.80'])
    expect(row(comparisonRows(runs, 'oos'), 'strategy', 'Sharpe')).toEqual(['-0.20', '-0.20'])
    expect(row(comparisonRows(runs, 'full'), 'benchmark', 'Sharpe')).toEqual(['0.95', '0.95'])
    expect(row(comparisonRows(runs, 'full'), 'settings', 'L')).toEqual(['60', '90'])
    expect(row(comparisonRows(runs, 'full'), 'settings', 'Tickers')).toEqual(['KO PEP', 'KO PEP'])
  })

  it('shows dashes for a segment the run has no metrics for (ruin in IS)', () => {
    const ruined = run(3, { metrics: { is: metrics(), oos: null, full: metrics() } })
    const rows = comparisonRows([ruined], 'oos')
    expect(row(rows, 'strategy', 'Sharpe')).toEqual([DASH])
    expect(row(rows, 'strategy', 'Max drawdown')).toEqual([DASH])
    expect(row(rows, 'benchmark', 'Sharpe')).toEqual(['1.10'])
  })

  it('shows a field an older engine did not store as a dash', () => {
    const old = run(4, { metrics: { is: metrics(), oos: metrics(), full: { ...metrics(), ann_borrow_drag: null } } })
    expect(row(comparisonRows([old], 'full'), 'strategy', 'Borrow drag')).toEqual([DASH])
    expect(row(comparisonRows([old], 'full'), 'strategy', 'Cost drag')).toEqual(['0.80%/yr'])
  })

  it('names the pairs model and the vol target', () => {
    const kalman = run(6, { resolved: resolved({ pairs_model: 'kalman', kalman_delta: 1e-6 }) })
    const old = run(7, { resolved: resolved({ pairs_model: null }) })
    const trend = run(8, { resolved: resolved({ strategy: 'trend', tickers: ['SPY'], pairs_model: null, vol_target: 0.1 }) })
    const rows = comparisonRows([run(1), kalman, old, trend], 'full')
    expect(row(rows, 'settings', 'Pairs model')).toEqual(['Rolling OLS', 'Kalman δ 1e-6', 'Rolling OLS', DASH])
    expect(row(rows, 'settings', 'Vol target')).toEqual(['off', 'off', 'off', '10%'])
  })

  it('shows z thresholds only where they apply', () => {
    const trend = run(5, { resolved: resolved({ strategy: 'trend', tickers: ['SPY'], z_entry: null, z_exit: null }) })
    expect(row(comparisonRows([run(1), trend], 'full'), 'settings', 'z entry / exit')).toEqual(['2 / 0.5', DASH])
    expect(row(comparisonRows([trend], 'full'), 'settings', 'Strategy')).toEqual(['Trend'])
  })
})
