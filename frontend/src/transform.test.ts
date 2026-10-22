import { describe, expect, it } from 'vitest'
import {
  assumptionLines,
  benchmarkLabels,
  blockLengthLabel,
  calendarGrid,
  configLabel,
  costSweepView,
  divergingColor,
  equityLogAxis,
  fanRows,
  fmtDelta,
  firstOosIndex,
  foldBands,
  heatmapBound,
  heatmapLayout,
  histogramRows,
  legRows,
  logAxis,
  lookbackLabel,
  niceTicks,
  NULL_CELL,
  segmentRange,
  selectionChanges,
  seriesRows,
  sortTrades,
  walkForwardRows,
} from './transform'
import type {
  CalendarReturns,
  CostSensitivity,
  Fold,
  LegSeries,
  ModelResult,
  MonteCarloResponse,
  ResolvedBacktestRun,
  SensitivityResponse,
  Series,
  Trade,
  WalkForwardSeries,
} from './types'

// Six days from s0: two warm-up days with no equity yet, the baseline on day 2, then three
// evaluated days; the split falls on day 4.
function makeSeries(overrides: Partial<Series> = {}): Series {
  return {
    dates: ['2020-01-02', '2020-01-03', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09'],
    equity: [null, null, 100000, 101000, 100500, 102000],
    benchmark_equity: [null, null, 100000, 100200, 100100, 100400],
    drawdown: [null, null, 0, 0, -0.00495, 0],
    daily_return: [null, null, null, 0.01, -0.00495, 0.01493],
    position: [0, 0, 1, 1, 0, -1],
    gross_exposure: [0, 0, 1, 1, 0, 1],
    score: [-2.5, -2.1, -0.4, null, 2.3, 2.6],
    beta: null,
    beta_used: null,
    vol_forecast: null,
    is_oos: [false, false, false, false, true, true],
    rolling_sharpe: [null, null, null, null, 0.8, 0.75],
    rolling_beta: [null, null, null, null, 0.1, 0.12],
    legs: null,
    ...overrides,
  }
}

function leg(ticker: string, overrides: Partial<LegSeries> = {}): LegSeries {
  return {
    ticker,
    score: [null, null, 0.05, 0.04, -0.01, -0.02],
    position: [0, 0, 0, 1, 1, -1],
    weight: [0, 0, 0.25, 0.25, -0.2, -0.2],
    vol_forecast: [null, 0.2, 0.21, 0.19, 0.18, 0.2],
    ...overrides,
  }
}

/** A trend-portfolio series: no top-level score, one entry per leg in `legs`. */
function portfolioSeries(legs: LegSeries[]): Series {
  return makeSeries({ score: null, legs, position: [0, 0, 1, 1, 1, 1] })
}

function trade(overrides: Partial<Trade>): Trade {
  return {
    id: 1,
    side: 'long',
    segment: 'is',
    open: false,
    signal_date: '2020-01-02',
    entry_date: '2020-01-03',
    exit_signal_date: '2020-01-06',
    exit_date: '2020-01-07',
    holding_days: 2,
    pnl: 0.01,
    beta_at_entry: null,
    ticker: null,
    ...overrides,
  }
}

describe('seriesRows: the shared rows of every synchronized chart (§13)', () => {
  it('keeps one row per date, nulls included, so chart indexes line up', () => {
    const series = makeSeries()
    const { rows } = seriesRows(series, [])
    expect(rows).toHaveLength(series.dates.length)
    expect(rows.map((r) => r.date)).toEqual(series.dates)
    expect(rows.map((r) => r.equity)).toEqual(series.equity) // null before the baseline renders as a gap
    expect(rows.map((r) => r.drawdown)).toEqual(series.drawdown)
    expect(rows.map((r) => r.score)).toEqual(series.score)
    expect(rows.map((r) => r.oos)).toEqual(series.is_oos)
    expect(rows.map((r) => r.exposure)).toEqual(series.gross_exposure)
    expect(rows.map((r) => r.rollingSharpe)).toEqual(series.rolling_sharpe)
    expect(rows.map((r) => r.rollingBeta)).toEqual(series.rolling_beta)
  })

  it('fills pairs and overlay series when present, null otherwise', () => {
    const plain = seriesRows(makeSeries(), []).rows
    expect(plain.every((r) => r.beta === null && r.betaUsed === null && r.volForecast === null)).toBe(true)
    const beta = [null, null, 0.9, 0.95, 1.0, 1.05]
    const betaUsed = [null, null, 0.9, 0.9, 0.9, 1.05]
    const vol = [null, 0.2, 0.21, 0.19, 0.18, 0.2]
    const { rows } = seriesRows(makeSeries({ beta, beta_used: betaUsed, vol_forecast: vol }), [])
    expect(rows.map((r) => r.beta)).toEqual(beta)
    expect(rows.map((r) => r.betaUsed)).toEqual(betaUsed)
    expect(rows.map((r) => r.volForecast)).toEqual(vol)
  })

  it('puts entry and exit markers on signal dates, at the score', () => {
    const trades = [
      trade({ id: 1, side: 'long', signal_date: '2020-01-02', exit_signal_date: '2020-01-06' }),
      trade({ id: 2, side: 'short', signal_date: '2020-01-08', exit_signal_date: null, exit_date: null, open: true }),
    ]
    const { rows, unplacedMarkers } = seriesRows(makeSeries(), trades)
    expect(unplacedMarkers).toBe(0)
    expect(rows[0].entryLong).toBe(-2.5)
    expect(rows[2].exit).toBe(-0.4)
    expect(rows[4].entryShort).toBe(2.3)
    // nothing anywhere else, and no exit marker for the open trade
    const marked = rows.filter((r) => r.entryLong !== null || r.entryShort !== null || r.exit !== null)
    expect(marked.map((r) => r.date)).toEqual(['2020-01-02', '2020-01-06', '2020-01-08'])
  })

  it('draws an exit on a null score (a degenerate window means flat) at zero', () => {
    const { rows } = seriesRows(makeSeries(), [trade({ exit_signal_date: '2020-01-07' })])
    expect(rows[3].score).toBeNull()
    expect(rows[3].exit).toBe(0)
  })

  it('a direct flip puts an exit and an entry on the same date', () => {
    const trades = [
      trade({ id: 1, side: 'long', signal_date: '2020-01-02', exit_signal_date: '2020-01-08' }),
      trade({ id: 2, side: 'short', signal_date: '2020-01-08', exit_signal_date: null, open: true }),
    ]
    const { rows } = seriesRows(makeSeries(), trades)
    expect(rows[4].exit).toBe(2.3)
    expect(rows[4].entryShort).toBe(2.3)
  })

  it('counts markers whose date is outside the series instead of dropping them silently', () => {
    const { unplacedMarkers } = seriesRows(makeSeries(), [trade({ signal_date: '2019-12-31' })])
    expect(unplacedMarkers).toBe(1)
  })

  it('rejects arrays of different lengths', () => {
    expect(() => seriesRows(makeSeries({ score: [1, 2] }), [])).toThrow(/different lengths/)
    expect(() => seriesRows(makeSeries({ beta: [1] }), [])).toThrow(/different lengths/)
  })

  it('finds the first OOS row, or -1 when ruin ended the series in-sample', () => {
    expect(firstOosIndex(makeSeries().is_oos)).toBe(4)
    expect(firstOosIndex([false, false, false])).toBe(-1)
  })

  it('a trend portfolio has no top-level score: null scores, and its markers go on the legs', () => {
    const series = portfolioSeries([leg('SPY'), leg('GLD')])
    const { rows, unplacedMarkers } = seriesRows(series, [trade({ ticker: 'SPY', signal_date: '2020-01-06' })])
    expect(rows.every((r) => r.score === null)).toBe(true)
    expect(rows.every((r) => r.entryLong === null && r.entryShort === null && r.exit === null)).toBe(true)
    expect(unplacedMarkers).toBe(0)
    expect(() => seriesRows(portfolioSeries([leg('SPY', { weight: [0, 1] })]), [])).toThrow(/different lengths/)
  })
})

describe('legRows: the trend portfolio’s per-leg charts (§13, view 3)', () => {
  it('keeps one row per date for every leg, in the request’s order', () => {
    const spy = leg('SPY')
    const gld = leg('GLD', { score: [null, null, -0.1, -0.12, -0.08, 0.02], weight: [0, 0, -0.5, -0.5, -0.5, 0.4] })
    const { legs } = legRows(portfolioSeries([spy, gld]), [])
    expect(legs.map((l) => l.ticker)).toEqual(['SPY', 'GLD'])
    expect(legs[1].rows.map((r) => r.date)).toEqual(makeSeries().dates)
    expect(legs[1].rows.map((r) => r.score)).toEqual(gld.score)
    expect(legs[1].weight).toEqual(gld.weight)
    expect(legs[0].volForecast).toEqual(spy.vol_forecast)
  })

  it('puts each trade’s markers on its own leg only', () => {
    const trades = [
      trade({ id: 1, ticker: 'SPY', side: 'long', signal_date: '2020-01-03', exit_signal_date: '2020-01-07' }),
      trade({ id: 2, ticker: 'GLD', side: 'short', signal_date: '2020-01-06', exit_signal_date: null, open: true }),
    ]
    const { legs, unplacedMarkers } = legRows(portfolioSeries([leg('SPY'), leg('GLD')]), trades)
    const marked = (i: number) =>
      legs[i].rows.filter((r) => r.entryLong !== null || r.entryShort !== null || r.exit !== null).map((r) => r.date)
    expect(marked(0)).toEqual(['2020-01-03', '2020-01-07'])
    expect(marked(1)).toEqual(['2020-01-06'])
    expect(legs[0].rows[1].entryLong).toBe(0) // a null score (warm-up) puts the marker at 0
    expect(legs[0].rows[3].exit).toBe(0.04)
    expect(legs[1].rows[2].entryShort).toBe(0.05)
    expect(unplacedMarkers).toBe(0)
    const outside = legRows(portfolioSeries([leg('SPY')]), [trade({ ticker: 'SPY', signal_date: '2019-12-31' })])
    expect(outside.unplacedMarkers).toBe(1)
  })

  it('is empty for the other strategies, and rejects a leg of the wrong length', () => {
    expect(legRows(makeSeries(), [trade({})])).toEqual({ legs: [], unplacedMarkers: 0 })
    expect(() => legRows(portfolioSeries([leg('SPY', { vol_forecast: [0.1] })]), [])).toThrow(/SPY leg/)
  })
})

describe('labels', () => {
  it('benchmarks A and C on the traded ticker, pairs on SPY and the portfolio on its basket (§6.4)', () => {
    expect(benchmarkLabels('mean_reversion', ['KO'])).toEqual({
      name: 'Buy & hold KO',
      phrase: 'buy & hold KO', // the ticker keeps its case inside a sentence
      short: 'B&H KO',
      tag: 'B&H',
    })
    expect(benchmarkLabels('trend', ['GLD']).name).toBe('Buy & hold GLD')
    expect(benchmarkLabels('pairs', ['KO', 'PEP']).short).toBe('B&H SPY')
    expect(benchmarkLabels('trend_portfolio', ['SPY', 'GLD'])).toEqual({
      name: 'Equal-weight basket',
      phrase: 'equal-weight basket',
      short: 'EW basket',
      tag: 'EW',
    })
  })

  const dates = { first_evaluated_date: '2015-04-01', split_date: '2022-08-05', end: '2025-12-31' }

  it('describes each segment’s dates', () => {
    expect(segmentRange(dates, null, 'is')).toBe('2015-04-01 → before 2022-08-05')
    expect(segmentRange(dates, null, 'oos')).toBe('2022-08-05 → 2025-12-31')
    expect(segmentRange(dates, null, 'full')).toBe('2015-04-01 → 2025-12-31')
  })

  it('ends the segments at a ruin day (§6.2)', () => {
    expect(segmentRange(dates, '2024-03-01', 'is')).toBe('2015-04-01 → before 2022-08-05')
    expect(segmentRange(dates, '2024-03-01', 'oos')).toBe('2022-08-05 → 2024-03-01 (ruin)')
    expect(segmentRange(dates, '2018-03-01', 'is')).toBe('2015-04-01 → 2018-03-01 (ruin)')
    expect(segmentRange(dates, '2018-03-01', 'oos')).toBe('none: ruined before the split')
    expect(segmentRange(dates, '2018-03-01', 'full')).toBe('2015-04-01 → 2018-03-01 (ruin)')
  })

  it('lists the assumptions the API actually used (§13)', () => {
    const resolved = {
      strategy: 'mean_reversion',
      cost_bps: 10,
      borrow_bps: 0,
      allow_short: true,
      execution_lag: 1,
      rf_source: '13-week T-bill (^IRX)',
      vol_target: null,
      vol_halflife: null,
      vol_refresh: null,
      max_leverage: null,
    } as ResolvedBacktestRun
    expect(assumptionLines(resolved)).toEqual([
      'Costs: 10 bps one-way, per leg, on traded notional',
      'Borrow fee: none; short-sale proceeds earn the risk-free rate',
      'Shorting: allowed',
      'Execution lag: 1 day (fills at the next close)',
      'Risk-free rate: 13-week T-bill (^IRX); idle cash and short-sale proceeds earn it',
      'Rebalancing: daily, to target weights; turnover from price drift is ignored',
      'Vol targeting: off',
    ])
    const sized = { ...resolved, vol_target: 0.1, vol_halflife: 42, vol_refresh: 'weekly', max_leverage: 2 } as const
    expect(assumptionLines(sized).at(-1)).toBe(
      'Vol targeting: 10% annual target, 42-day EWMA half-life, weekly refresh, 2.0× max leverage',
    )
    const longOnly = assumptionLines({ ...resolved, borrow_bps: 150, allow_short: false })
    expect(longOnly.slice(1, 3)).toEqual([
      'Borrow fee: 150 bps a year on short notional, charged daily',
      'Long only: short signals are held flat',
    ])
    const pairs = assumptionLines({ ...resolved, strategy: 'pairs' })
    expect(pairs[2]).toBe('Shorting: required (a pair always holds a short leg)')
  })

  it('states the pairs model, and the trend portfolio’s per-asset sizing', () => {
    const base = {
      strategy: 'pairs',
      tickers: ['KO', 'PEP'],
      L: 60,
      cost_bps: 10,
      borrow_bps: 0,
      allow_short: true,
      execution_lag: 1,
      rf_source: '13-week T-bill (^IRX)',
      pairs_model: 'ols',
      kalman_delta: null,
      vol_target: null,
      vol_halflife: null,
      vol_refresh: null,
      max_leverage: null,
    } as ResolvedBacktestRun
    expect(assumptionLines(base).at(-2)).toBe('Pairs model: rolling OLS over 60 days')
    expect(assumptionLines({ ...base, pairs_model: 'kalman', kalman_delta: 1e-6 }).at(-2)).toBe(
      'Pairs model: Kalman filter, δ = 1e-6, started from an OLS fit on the first 60 days',
    )
    expect([fmtDelta(1e-6), fmtDelta(0.00025), fmtDelta(1e-2), fmtDelta(null)]).toEqual(['1e-6', '2.5e-4', '1e-2', '—'])
    const portfolio = assumptionLines({
      ...base,
      strategy: 'trend_portfolio',
      tickers: ['SPY', 'QQQ', 'GLD', 'TLT'],
      L: 252,
      pairs_model: null,
      vol_target: 0.4,
      vol_halflife: 42,
      vol_refresh: 'weekly',
      max_leverage: 2,
    })
    expect(portfolio.some((line) => line.startsWith('Pairs model'))).toBe(false)
    expect(portfolio.at(-1)).toBe(
      'Vol targeting: always on, 40% a year per asset, 42-day EWMA half-life, weekly refresh, each leg at most 2.0× / 4 (gross at most 2.0×)',
    )
  })

  it('names the block lengths a Monte Carlo run used (§10.8)', () => {
    const resolved = (block_len: number | 'auto', circular: number, stationary: number) =>
      ({ block_len, block_len_used: { circular, stationary } }) as MonteCarloResponse['resolved']
    expect(blockLengthLabel(resolved(10, 10, 10))).toBe("10 days (the stationary bootstrap's mean)")
    expect(blockLengthLabel(resolved(1, 1, 1))).toBe("1 day (the stationary bootstrap's mean)")
    expect(blockLengthLabel(resolved('auto', 9, 8))).toBe('auto: 9 days circular, mean 8 days stationary')
  })
})

describe('sortTrades', () => {
  const trades = [
    trade({ id: 1, pnl: 0.02, exit_date: '2020-03-02', side: 'long', open: false }),
    trade({ id: 2, pnl: -0.01, exit_date: null, side: 'short', open: true }),
    trade({ id: 3, pnl: 0.005, exit_date: '2020-02-03', side: 'long', open: false }),
    trade({ id: 4, pnl: 0.02, exit_date: '2020-01-15', side: 'short', open: false }),
  ]
  const ids = (ts: Trade[]) => ts.map((t) => t.id)

  it('sorts numbers both ways and breaks ties by id', () => {
    expect(ids(sortTrades(trades, 'pnl', 'asc'))).toEqual([2, 3, 1, 4])
    expect(ids(sortTrades(trades, 'pnl', 'desc'))).toEqual([1, 4, 3, 2])
  })

  it('sorts ISO dates chronologically and keeps nulls last in both directions', () => {
    expect(ids(sortTrades(trades, 'exit_date', 'asc'))).toEqual([4, 3, 1, 2])
    expect(ids(sortTrades(trades, 'exit_date', 'desc'))).toEqual([1, 3, 4, 2])
  })

  it('sorts strings and booleans', () => {
    expect(ids(sortTrades(trades, 'side', 'asc'))).toEqual([1, 3, 2, 4])
    expect(ids(sortTrades(trades, 'open', 'desc'))).toEqual([2, 1, 3, 4])
  })

  it('returns a copy', () => {
    const before = ids(trades)
    sortTrades(trades, 'pnl', 'desc')
    expect(ids(trades)).toEqual(before)
  })
})

describe('sensitivity heatmap (§9, §13)', () => {
  const pairs: SensitivityResponse = {
    L_values: [20, 60],
    z_entry_values: [1.5, 2.0],
    sharpe: [
      [0.4, null],
      [-0.8, 1.2],
    ],
    n_configs_tested: 4,
    window: { start: '2015-06-01', end: '2022-08-05' },
    selected: { L: 60, z_entry: 2.0 },
    deflated: { n_trials: 3, sr0_star: 0.9, selected_dsr: 0.6, best: { L: 60, z_entry: 2.0, sharpe: 1.2, dsr: 0.6 } },
  }

  it('lays out rows = L and columns = z entry, outlining the request’s own cell', () => {
    const layout = heatmapLayout(pairs)
    expect(layout.rowLabels).toEqual(['L 20', 'L 60'])
    expect(layout.colLabels).toEqual(['z 1.5', 'z 2.0'])
    expect(layout.cells.map((row) => row.map((c) => c.sharpe))).toEqual(pairs.sharpe)
    expect(layout.cells[1][1]).toEqual({ sharpe: 1.2, L: 60, zEntry: 2.0, selected: true })
    expect(layout.cells.flat().filter((c) => c.selected)).toHaveLength(1)
  })

  it('shows trend as a single row over L', () => {
    const trend: SensitivityResponse = {
      L_values: [21, 63, 126, 252, 378],
      z_entry_values: null,
      sharpe: [[0.1], [0.2], [null], [0.35], [0.3]],
      n_configs_tested: 5,
      window: { start: '2010-02-01', end: '2020-06-01' },
      selected: { L: 252, z_entry: null },
      deflated: { n_trials: 4, sr0_star: 0.2, selected_dsr: 0.7, best: { L: 252, z_entry: null, sharpe: 0.35, dsr: 0.7 } },
    }
    const layout = heatmapLayout(trend)
    expect(layout.cells).toHaveLength(1)
    expect(layout.cells[0].map((c) => c.sharpe)).toEqual([0.1, 0.2, null, 0.35, 0.3])
    expect(layout.colLabels).toEqual(['21 (1m)', '63 (3m)', '126 (6m)', '252 (12m)', '378 (18m)'])
    expect(layout.cells[0].map((c) => c.selected)).toEqual([false, false, false, true, false])
  })

  it('labels z entries without rounding them', () => {
    const custom = heatmapLayout({ ...pairs, z_entry_values: [2, 2.25], selected: null })
    expect(custom.colLabels).toEqual(['z 2.0', 'z 2.25'])
  })

  it('outlines nothing when the request is not in the grid', () => {
    const layout = heatmapLayout({ ...pairs, selected: null })
    expect(layout.cells.flat().some((c) => c.selected)).toBe(false)
  })

  it('labels lookbacks in months only when they are whole months', () => {
    expect(lookbackLabel(252)).toBe('252 (12m)')
    expect(lookbackLabel(60)).toBe('60')
  })

  it('scales colours to the largest |Sharpe|, at least 0.5, ignoring nulls', () => {
    expect(heatmapBound(heatmapLayout(pairs).cells)).toBe(1.2)
    const small = heatmapLayout({ ...pairs, sharpe: [[0.1, null], [-0.2, 0.05]] })
    expect(heatmapBound(small.cells)).toBe(0.5)
    const empty = heatmapLayout({ ...pairs, sharpe: [[null, null], [null, null]] })
    expect(heatmapBound(empty.cells)).toBe(0.5)
  })

  it('uses a diverging scale centred on a Sharpe of 0', () => {
    expect(divergingColor(0, 1)).toBe('rgb(38, 38, 38)')
    expect(divergingColor(1, 1)).toBe('rgb(16, 185, 129)')
    expect(divergingColor(-1, 1)).toBe('rgb(239, 68, 68)')
    expect(divergingColor(5, 1)).toBe(divergingColor(1, 1)) // saturates beyond the bound
    expect(divergingColor(-0.5, 1)).toBe('rgb(139, 53, 53)')
    expect(divergingColor(0.5, 2)).toBe(divergingColor(0.25, 1))
    expect(divergingColor(null, 1)).toBe(NULL_CELL)
  })
})

describe('Monte Carlo rows', () => {
  const model: ModelResult = {
    bands: { p5: [1, 0.9, 0.85], p25: [1, 0.97, 0.95], p50: [1, 1.01, 1.02], p75: [1, 1.04, 1.07], p95: [1, 1.1, 1.15] },
    sample_paths: [
      [1, 1.02, 1.05],
      [1, 0.98, 0.96],
    ],
    hist_counts: [3, 5, 2],
    var95: 0.12,
    var99: 0.2,
    cvar95: 0.16,
    cvar99: 0.25,
    elapsed_ms: 12,
  }

  it('builds one fan row per horizon day with the sample paths transposed', () => {
    const rows = fanRows(model)
    expect(rows).toHaveLength(3)
    expect(rows[2]).toEqual({ t: 2, p5: 0.85, p25: 0.95, p50: 1.02, p75: 1.07, p95: 1.15, paths: [1.05, 0.96] })
  })

  it('pairs each count with its bin and flags the tail bins (§10.4)', () => {
    const rows = histogramRows(model.hist_counts, [-0.3, -0.1, 0.1, 0.3])
    expect(rows.map((r) => r.mid)).toEqual([-0.2, 0, 0.2])
    expect(rows.map((r) => r.tail)).toEqual(['low', null, 'high'])
    expect(rows.map((r) => r.share)).toEqual([0.3, 0.5, 0.2])
    expect(rows[1]).toMatchObject({ lo: -0.1, hi: 0.1, count: 5 })
  })

  it('rejects edges that do not match the counts', () => {
    expect(() => histogramRows([1, 2], [0, 1])).toThrow(/one more edge/)
  })

  it('places round axis ticks inside the exact bin range', () => {
    expect(niceTicks(-0.1934, 0.2271)).toEqual([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2])
    expect(niceTicks(0, 252)).toEqual([0, 50, 100, 150, 200, 250])
    expect(niceTicks(0, 21)).toEqual([0, 5, 10, 15, 20])
    expect(niceTicks(1, 1)).toEqual([1])
    for (const t of niceTicks(-0.0071, 0.0089)) expect(Object.is(t, -0)).toBe(false)
  })
})

describe('cost sweep (§8.1)', () => {
  const sweep = (breakEven: number | null): CostSensitivity => ({
    cost_bps: [0, 2, 5, 10, 20, 50, 100],
    sharpe: {
      is: [1.1, 1.0, 0.9, 0.7, 0.4, -0.5, null],
      oos: null,
      full: [0.8, 0.75, 0.65, 0.5, 0.2, -0.6, -1.9],
    },
    break_even_bps: { is: 30, oos: null, full: breakEven },
  })

  it('pairs each cost with the selected segment’s Sharpe, keeping nulls', () => {
    const view = costSweepView(sweep(28), 'is', 2.5)
    expect(view.rows).toEqual([
      { bps: 0, sharpe: 1.1 },
      { bps: 2, sharpe: 1.0 },
      { bps: 5, sharpe: 0.9 },
      { bps: 10, sharpe: 0.7 },
      { bps: 20, sharpe: 0.4 },
      { bps: 50, sharpe: -0.5 },
      { bps: 100, sharpe: null }, // ruined at that cost
    ])
    expect(view.breakEven).toBe(30)
    expect(costSweepView(sweep(28), 'full', 2.5).rows.map((r) => r.sharpe)).toEqual(sweep(28).sharpe.full)
    expect(costSweepView(sweep(28), 'oos', 2.5).rows.every((r) => r.sharpe === null)).toBe(true)
  })

  it('keeps the grid’s axis when the break-even is inside it', () => {
    expect(costSweepView(sweep(28), 'full', 2.5)).toMatchObject({ breakEven: 28, domainMax: 100, offChart: false })
  })

  it('stretches the axis to a break-even up to 4× beyond the grid, and reports one further out in words', () => {
    expect(costSweepView(sweep(250), 'full', 2.5)).toMatchObject({ domainMax: 280, offChart: false }) // ceil(27.5) * 10
    expect(costSweepView(sweep(400), 'full', 2.5)).toMatchObject({ domainMax: 440, offChart: false })
    expect(costSweepView(sweep(401), 'full', 2.5)).toMatchObject({ domainMax: 100, offChart: true })
  })

  it('says why a break-even is null: no trades, or losing before costs', () => {
    expect(costSweepView(sweep(null), 'full', 0)).toMatchObject({ breakEven: null, nullReason: 'no-trades' })
    expect(costSweepView(sweep(null), 'full', 3.1)).toMatchObject({ breakEven: null, nullReason: 'losing' })
    expect(costSweepView(sweep(28), 'full', 0).nullReason).toBeNull()
  })
})

describe('calendar grid (§8.2)', () => {
  const calendar: CalendarReturns = {
    monthly: [
      { year: 2020, month: 11, strategy: 0.004, benchmark: 0.1, partial: true },
      { year: 2020, month: 12, strategy: -0.031, benchmark: 0.03, partial: false },
      { year: 2021, month: 1, strategy: 0.012, benchmark: -0.01, partial: false },
      { year: 2021, month: 2, strategy: 0.001, benchmark: 0.02, partial: true },
    ],
    annual: [
      { year: 2020, strategy: -0.027, benchmark: 0.133, partial: true },
      { year: 2021, strategy: 0.013, benchmark: 0.01, partial: true },
    ],
  }

  it('places each month in its year’s row and leaves the rest empty', () => {
    const { years } = calendarGrid(calendar)
    expect(years.map((y) => y.year)).toEqual([2020, 2021])
    expect(years[0].months).toHaveLength(12)
    expect(years[0].months.slice(0, 10).every((c) => c === null)).toBe(true)
    expect(years[0].months[10]).toEqual({ value: 0.004, partial: true })
    expect(years[0].months[11]).toEqual({ value: -0.031, partial: false })
    expect(years[1].months.slice(0, 2)).toEqual([
      { value: 0.012, partial: false },
      { value: 0.001, partial: true },
    ])
    expect(years[1].months.slice(2).every((c) => c === null)).toBe(true)
    expect(years[0]).toMatchObject({ strategy: -0.027, benchmark: 0.133, partial: true })
  })

  it('scales colours to the largest |monthly return| of the strategy, at least 2%', () => {
    expect(calendarGrid(calendar).bound).toBe(0.031) // the benchmark's 10% month doesn't count
    const calm = { ...calendar, monthly: calendar.monthly.map((m) => ({ ...m, strategy: m.strategy / 10 })) }
    expect(calendarGrid(calm).bound).toBe(0.02)
  })

  it('rejects a month whose year has no annual row', () => {
    expect(() => calendarGrid({ ...calendar, annual: calendar.annual.slice(1) })).toThrow(/2020-11/)
  })
})

// ---------------------------------------------------------------- walk-forward (§9.2)

// The baseline, fold 0's entry day and two test days, then folds 1 and 2 of two days and a one-day fold 3.
function makeWalkForwardSeries(overrides: Partial<WalkForwardSeries> = {}): WalkForwardSeries {
  return {
    dates: ['2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08', '2021-01-11', '2021-01-12', '2021-01-13', '2021-01-14'],
    equity: [100000, 99990, 100200, 100100, 100500, 100400, 100900, 101000, 100800],
    fixed_equity: [100000, 100100, 100300, 100250, 0, null, null, null, null], // ruined on 2021-01-08
    benchmark_equity: [100000, 100050, 100120, 100300, 100200, 100600, 100700, 100650, 100900],
    fold: [null, 0, 0, 0, 1, 1, 2, 2, 3],
    ...overrides,
  }
}

function fold(selected: Fold['selected'], overrides: Partial<Fold> = {}): Fold {
  return {
    train_start: '2018-01-02',
    train_end: '2020-12-31',
    test_start: '2021-01-04',
    test_end: '2021-03-31',
    selected,
    train_sharpe: selected === null ? null : 0.8,
    test_sharpe: null,
    ...overrides,
  }
}

describe('walkForwardRows', () => {
  it('keeps one row per date with every series, nulls included', () => {
    const series = makeWalkForwardSeries()
    const rows = walkForwardRows(series)
    expect(rows.map((r) => r.date)).toEqual(series.dates)
    expect(rows.map((r) => r.equity)).toEqual(series.equity)
    expect(rows.map((r) => r.fixed)).toEqual(series.fixed_equity) // a gap after the fixed run's ruin
    expect(rows.map((r) => r.benchmark)).toEqual(series.benchmark_equity)
    expect(rows.map((r) => r.fold)).toEqual(series.fold)
  })

  it('rejects arrays of different lengths', () => {
    expect(() => walkForwardRows(makeWalkForwardSeries({ fold: [null, 0] }))).toThrow(/different lengths/)
    expect(() => walkForwardRows(makeWalkForwardSeries({ fixed_equity: [] }))).toThrow(/different lengths/)
  })
})

describe('foldBands: alternate test folds are shaded (§13, view 11)', () => {
  it('spans each odd fold from its first to its last date', () => {
    expect(foldBands(makeWalkForwardSeries())).toEqual([
      { fold: 1, x1: '2021-01-08', x2: '2021-01-11' },
      { fold: 3, x1: '2021-01-14', x2: '2021-01-14' }, // a one-day fold
    ])
  })

  it('shades nothing with a single fold, and skips the baseline', () => {
    const one = makeWalkForwardSeries({ fold: [null, 0, 0, 0, 0, 0, 0, 0, 0] })
    expect(foldBands(one)).toEqual([])
    const two = makeWalkForwardSeries({ fold: [null, 0, 0, 0, 0, 1, 1, 1, 1] })
    expect(foldBands(two)).toEqual([{ fold: 1, x1: '2021-01-11', x2: '2021-01-14' }])
  })
})

describe('the fold table', () => {
  it('labels a configuration like the heatmap, and a flat fold as flat', () => {
    expect(configLabel({ L: 60, z_entry: 2 })).toBe('L 60 · z 2.0')
    expect(configLabel({ L: 40, z_entry: 2.25 })).toBe('L 40 · z 2.25')
    expect(configLabel({ L: 252, z_entry: null })).toBe('L 252 (12m)') // trend
    expect(configLabel({ L: 250, z_entry: null })).toBe('L 250')
    expect(configLabel(null)).toBe('flat')
  })

  it('marks every fold whose choice differs from the one before, by value', () => {
    const folds = [
      fold({ L: 60, z_entry: 2 }),
      fold({ L: 60, z_entry: 2 }), // equal but a different object: no change
      fold({ L: 60, z_entry: 2.5 }),
      fold(null),
      fold(null),
      fold({ L: 20, z_entry: 2.5 }),
      fold({ L: 20, z_entry: null }),
    ]
    expect(selectionChanges(folds)).toEqual([false, false, true, true, false, true, true])
    expect(selectionChanges([fold({ L: 20, z_entry: 1.5 }), fold(null)])).toEqual([false, true]) // the second fold
    expect(selectionChanges([])).toEqual([])
  })
})

describe('logAxis: the log-scale equity axis (§13 view 2)', () => {
  it('puts ticks at 1, 2 and 5 × 10^k, with the domain widened to the nearest such values', () => {
    expect(logAxis(80_000, 260_000)).toEqual({ domain: [50_000, 500_000], ticks: [50_000, 100_000, 200_000, 500_000] })
    expect(logAxis(100_000, 180_000)).toEqual({ domain: [100_000, 200_000], ticks: [100_000, 200_000] })
    expect(logAxis(0.3, 0.45)).toEqual({ domain: [0.2, 0.5], ticks: [0.2, 0.5] })
  })

  it('has no float noise in its ticks', () => {
    const axis = logAxis(0.0011, 0.07)
    expect(axis?.ticks).toEqual([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    expect(logAxis(3e-6, 8e-6)?.ticks).toEqual([2e-6, 5e-6, 1e-5]) // 5 * 10 ** -6 is 4.9999999999999996e-6
  })

  it('every tick is a 1, 2 or 5 inside the domain, and there are at least two', () => {
    for (const [lo, hi] of [[1, 1], [3, 3], [99_999, 100_001], [12_345, 9_876_543], [5e4, 5e4], [7, 1e9]]) {
      const axis = logAxis(lo, hi)
      expect(axis).not.toBeNull()
      const { domain, ticks } = axis!
      expect(domain[0]).toBeLessThanOrEqual(lo)
      expect(domain[1]).toBeGreaterThanOrEqual(hi)
      expect(ticks.length).toBeGreaterThanOrEqual(2)
      for (const t of ticks) {
        expect(t).toBeGreaterThanOrEqual(domain[0])
        expect(t).toBeLessThanOrEqual(domain[1])
        expect(['1', '2', '5']).toContain(t.toExponential().charAt(0))
        expect(t.toExponential()).toMatch(/^[125]e/)
      }
    }
  })

  it('keeps a flat series on two ticks', () => {
    expect(logAxis(100_000, 100_000)).toEqual({ domain: [100_000, 200_000], ticks: [100_000, 200_000] })
  })

  it('keeps only the powers of ten over many decades', () => {
    expect(logAxis(5, 2e6)).toEqual({ domain: [5, 2e6], ticks: [10, 100, 1000, 10_000, 100_000, 1_000_000] })
    expect(logAxis(1000, 5e5)?.ticks).toHaveLength(9) // 1, 2, 5 over nine values still fits
    expect(logAxis(1000, 1e6)?.ticks).toEqual([1000, 10_000, 100_000, 1_000_000]) // ten would not
  })

  it('is null when the range is not positive: a ruined run reaches 0', () => {
    expect(logAxis(0, 100_000)).toBeNull()
    expect(logAxis(-5, 10)).toBeNull()
    expect(logAxis(10, 5)).toBeNull()
    expect(logAxis(1, Infinity)).toBeNull()
    expect(logAxis(Number.NaN, 5)).toBeNull()
  })
})

describe('equityLogAxis', () => {
  it('spans the equity, and the benchmark only while it is shown', () => {
    const { rows } = seriesRows(
      makeSeries({ equity: [null, null, 100000, 150000, 120000, 130000], benchmark_equity: [null, null, 100000, 99000, 98000, 260000] }),
      [],
    )
    expect(equityLogAxis(rows, false)?.domain).toEqual([100_000, 200_000])
    expect(equityLogAxis(rows, true)?.domain).toEqual([50_000, 500_000])
  })

  it('is null for a ruined run and for one with no equity yet', () => {
    const ruined = seriesRows(makeSeries({ equity: [null, null, 100000, 50000, 10000, 0] }), []).rows
    expect(equityLogAxis(ruined, false)).toBeNull()
    const empty = seriesRows(makeSeries({ equity: [null, null, null, null, null, null] }), []).rows
    expect(equityLogAxis(empty, false)).toBeNull()
  })
})
