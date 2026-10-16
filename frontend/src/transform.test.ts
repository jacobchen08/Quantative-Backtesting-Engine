import { describe, expect, it } from 'vitest'
import {
  assumptionLines,
  benchmarkName,
  divergingColor,
  fanRows,
  firstOosIndex,
  heatmapBound,
  heatmapLayout,
  histogramRows,
  lookbackLabel,
  niceTicks,
  NULL_CELL,
  segmentRange,
  seriesRows,
  sortTrades,
} from './transform'
import type { ModelResult, ResolvedBacktestRun, SensitivityResponse, Series, Trade } from './types'

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
    ...overrides,
  }
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
})

describe('labels', () => {
  it('benchmarks A and C on the traded ticker and pairs on SPY (§6.4)', () => {
    expect(benchmarkName('mean_reversion', ['KO'])).toBe('KO')
    expect(benchmarkName('trend', ['GLD'])).toBe('GLD')
    expect(benchmarkName('pairs', ['KO', 'PEP'])).toBe('SPY')
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
      cost_bps: 10,
      execution_lag: 1,
      rf_source: '13-week T-bill (^IRX)',
      vol_target: null,
      vol_halflife: null,
      vol_refresh: null,
      max_leverage: null,
    } as ResolvedBacktestRun
    expect(assumptionLines(resolved)).toEqual([
      'Costs: 10 bps one-way, per leg, on traded notional',
      'Execution lag: 1 day (fills at the next close)',
      'Risk-free rate: 13-week T-bill (^IRX); idle cash and short-sale proceeds earn it',
      'Rebalancing: daily, to target weights; turnover from price drift is ignored',
      'Vol targeting: off',
    ])
    const sized = { ...resolved, vol_target: 0.1, vol_halflife: 42, vol_refresh: 'weekly', max_leverage: 2 } as const
    expect(assumptionLines(sized).at(-1)).toBe(
      'Vol targeting: 10% annual target, 42-day EWMA half-life, weekly refresh, 2.0× max leverage',
    )
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
