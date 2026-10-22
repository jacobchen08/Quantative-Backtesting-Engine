import { describe, expect, it } from 'vitest'
import { calendarCsv, csvField, csvFileName, seriesCsv, toCsv, TRADE_COLUMNS, tradesCsv } from './csv'
import type { CalendarReturns, LegSeries, Series, Trade } from './types'

/** Records split on CRLF; the text ends with one, so the last element is empty. */
const lines = (csv: string) => csv.split('\r\n')

describe('csvField: RFC 4180 quoting', () => {
  it('leaves plain fields alone', () => {
    expect(csvField('KO')).toBe('KO')
    expect(csvField('2020-01-02')).toBe('2020-01-02')
  })

  it('quotes a comma, a double quote, CR or LF, doubling inner quotes', () => {
    expect(csvField('a,b')).toBe('"a,b"')
    expect(csvField('say "hi"')).toBe('"say ""hi"""')
    expect(csvField('"')).toBe('""""')
    expect(csvField('two\nlines')).toBe('"two\nlines"')
    expect(csvField('cr\rhere')).toBe('"cr\rhere"')
    expect(csvField('crlf\r\n')).toBe('"crlf\r\n"')
  })

  it('writes nulls as empty fields', () => {
    expect(csvField(null)).toBe('')
    expect(csvField(undefined)).toBe('')
  })

  it('writes numbers at full precision with String(x), not formatted', () => {
    expect(csvField(0.1 + 0.2)).toBe('0.30000000000000004')
    expect(csvField(123456.789)).toBe('123456.789')
    expect(csvField(1e-7)).toBe('1e-7')
    expect(csvField(-0.012345678901234)).toBe('-0.012345678901234')
    expect(csvField(0)).toBe('0')
  })

  it('writes booleans as true and false', () => {
    expect(csvField(true)).toBe('true')
    expect(csvField(false)).toBe('false')
  })
})

describe('toCsv', () => {
  it('ends every record, the last included, with CRLF', () => {
    expect(toCsv(['a', 'b'], [[1, null], ['x,y', true]])).toBe('a,b\r\n1,\r\n"x,y",true\r\n')
  })

  it('has no bare LF line endings outside quoted fields', () => {
    const csv = toCsv(['a'], [['one'], ['two']])
    expect(csv.replaceAll('\r\n', '')).not.toMatch(/[\r\n]/)
  })

  it('writes just the header for no rows', () => {
    expect(toCsv(['a', 'b'], [])).toBe('a,b\r\n')
  })

  it('rejects a row whose length differs from the header', () => {
    expect(() => toCsv(['a', 'b'], [[1]])).toThrow(/1 fields, header has 2/)
  })
})

describe('csvFileName', () => {
  it('is {strategy}-{tickers}-{start}-{end}-{table}.csv', () => {
    const resolved = { strategy: 'pairs', tickers: ['KO', 'PEP'], start: '2015-01-02', end: '2025-12-31' } as const
    expect(csvFileName({ ...resolved, tickers: [...resolved.tickers] }, 'trades')).toBe('pairs-KO_PEP-2015-01-02-2025-12-31-trades.csv')
    expect(csvFileName({ ...resolved, strategy: 'trend', tickers: ['BRK-B'] }, 'series')).toBe(
      'trend-BRK-B-2015-01-02-2025-12-31-series.csv',
    )
  })
})

function trade(overrides: Partial<Trade>): Trade {
  return {
    id: 1,
    side: 'long',
    segment: 'is',
    open: false,
    signal_date: '2020-01-02',
    entry_date: '2020-01-03',
    exit_signal_date: '2020-01-07',
    exit_date: '2020-01-08',
    holding_days: 3,
    pnl: 0.0123456789,
    beta_at_entry: 1.05,
    ticker: null,
    ...overrides,
  }
}

describe('tradesCsv', () => {
  it('has the API field names as its header, in a fixed order', () => {
    expect(lines(tradesCsv([]))[0]).toBe(
      'id,ticker,side,segment,open,signal_date,entry_date,exit_signal_date,exit_date,holding_days,pnl,beta_at_entry',
    )
    expect(TRADE_COLUMNS).toHaveLength(12)
  })

  it('writes one row per trade in id order, with nulls empty and P&L unrounded', () => {
    const csv = tradesCsv([
      trade({ id: 2, side: 'short', segment: 'oos', open: true, exit_signal_date: null, exit_date: null, pnl: -0.001, beta_at_entry: null }),
      trade({ id: 1 }),
    ])
    expect(lines(csv)).toEqual([
      'id,ticker,side,segment,open,signal_date,entry_date,exit_signal_date,exit_date,holding_days,pnl,beta_at_entry',
      '1,,long,is,false,2020-01-02,2020-01-03,2020-01-07,2020-01-08,3,0.0123456789,1.05',
      '2,,short,oos,true,2020-01-02,2020-01-03,,,3,-0.001,',
      '',
    ])
  })

  it('fills the ticker column for trend-portfolio legs', () => {
    expect(lines(tradesCsv([trade({ ticker: 'GLD', beta_at_entry: null })]))[1]).toMatch(/^1,GLD,long,/)
  })
})

function makeSeries(overrides: Partial<Series> = {}): Series {
  return {
    dates: ['2020-01-02', '2020-01-03', '2020-01-06'],
    equity: [null, 100000, 100500.25],
    benchmark_equity: [null, 100000, 99900],
    drawdown: [null, 0, 0],
    daily_return: [null, null, 0.0050025],
    position: [0, 1, 1],
    gross_exposure: [0, 1, 1],
    score: [-2.5, -2.1, -0.4],
    beta: [0.9, 0.91, 0.92],
    beta_used: [null, 0.91, 0.91],
    vol_forecast: null,
    is_oos: [false, false, true],
    rolling_sharpe: [null, null, null],
    rolling_beta: [null, null, null],
    legs: null,
    ...overrides,
  }
}

describe('seriesCsv', () => {
  it('writes one row per date and leaves out the arrays the API sent as null', () => {
    const rows = lines(seriesCsv(makeSeries()))
    expect(rows[0]).toBe(
      'date,equity,benchmark_equity,drawdown,daily_return,position,gross_exposure,score,beta,beta_used,is_oos,rolling_sharpe,rolling_beta',
    )
    expect(rows.slice(1)).toEqual([
      '2020-01-02,,,,,0,0,-2.5,0.9,,false,,',
      '2020-01-03,100000,100000,0,,1,1,-2.1,0.91,0.91,false,,',
      '2020-01-06,100500.25,99900,0,0.0050025,1,1,-0.4,0.92,0.91,true,,',
      '',
    ])
  })

  it('adds four columns per trend-portfolio leg', () => {
    const leg = (ticker: string): LegSeries => ({
      ticker,
      score: [null, 0.1, 0.2],
      position: [0, 1, 1],
      weight: [0, 0.25, 0.3],
      vol_forecast: [0.2, 0.21, 0.19],
    })
    const header = lines(seriesCsv(makeSeries({ score: null, beta: null, beta_used: null, legs: [leg('SPY'), leg('GLD')] })))[0]
    expect(header.split(',').slice(-8)).toEqual([
      'SPY_score',
      'SPY_position',
      'SPY_weight',
      'SPY_vol_forecast',
      'GLD_score',
      'GLD_position',
      'GLD_weight',
      'GLD_vol_forecast',
    ])
    expect(header).not.toMatch(/(^|,)score(,|$)/)
  })

  it('includes vol_forecast when the overlay is on', () => {
    expect(lines(seriesCsv(makeSeries({ vol_forecast: [null, 0.1, 0.11] })))[0]).toContain(',vol_forecast,')
  })

  it('rejects an array whose length differs from the dates', () => {
    expect(() => seriesCsv(makeSeries({ position: [0, 1] }))).toThrow(/series.position has 2 values for 3 dates/)
  })
})

describe('calendarCsv', () => {
  it('writes the monthly rows, then the annual rows with an empty month', () => {
    const calendar: CalendarReturns = {
      monthly: [
        { year: 2020, month: 11, strategy: 0.0125, benchmark: -0.002, partial: true },
        { year: 2020, month: 12, strategy: -0.01, benchmark: 0.03, partial: false },
      ],
      annual: [{ year: 2020, strategy: 0.0023750000000000004, benchmark: 0.02794, partial: true }],
    }
    expect(lines(calendarCsv(calendar))).toEqual([
      'period,year,month,strategy,benchmark,partial',
      'month,2020,11,0.0125,-0.002,true',
      'month,2020,12,-0.01,0.03,false',
      'year,2020,,0.0023750000000000004,0.02794,true',
      '',
    ])
  })
})
