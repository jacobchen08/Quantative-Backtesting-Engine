import { describe, expect, it } from 'vitest'
import {
  DASH,
  fmtCI,
  fmtDays,
  fmtInt,
  fmtLag,
  fmtMoney,
  fmtMonth,
  fmtMs,
  fmtMultiple,
  fmtNum,
  fmtPct,
  fmtPValue,
  fmtRate,
  fmtSignedPct,
  fmtTurnover,
  isNum,
} from './format'

describe('null handling (§2.4: undefined means null; §13: null shows as a dash)', () => {
  const formatters: ((x: number | null | undefined) => string)[] = [
    fmtPct,
    fmtSignedPct,
    fmtRate,
    fmtNum,
    fmtMultiple,
    fmtPValue,
    fmtDays,
    fmtInt,
    fmtMoney,
    fmtMs,
    fmtTurnover,
  ]

  it.each([null, undefined, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    'every number formatter shows %s as a dash',
    (x) => {
      for (const f of formatters) expect(f(x)).toBe(DASH)
    },
  )

  it('shows a missing or non-finite confidence interval as a dash', () => {
    expect(fmtCI(null)).toBe(DASH)
    expect(fmtCI(undefined)).toBe(DASH)
    expect(fmtCI([Number.NaN, 1])).toBe(DASH)
    expect(fmtCI([0, Number.POSITIVE_INFINITY])).toBe(DASH)
  })

  it('isNum accepts only finite numbers', () => {
    expect(isNum(0)).toBe(true)
    expect(isNum(-1.5)).toBe(true)
    expect(isNum(null)).toBe(false)
    expect(isNum(undefined)).toBe(false)
    expect(isNum(Number.NaN)).toBe(false)
    expect(isNum(Number.POSITIVE_INFINITY)).toBe(false)
  })

  it('keeps real zeros, which are not null', () => {
    expect(fmtPct(0)).toBe('0.0%')
    expect(fmtNum(0)).toBe('0.00')
    expect(fmtInt(0)).toBe('0')
    expect(fmtMultiple(0)).toBe('0.00×')
  })
})

describe('percent formatting (the API sends decimals, §2.1)', () => {
  it('converts decimals to percent', () => {
    expect(fmtPct(0.1234)).toBe('12.3%')
    expect(fmtPct(0.1234, 2)).toBe('12.34%')
    expect(fmtPct(-0.25, 0)).toBe('-25%')
  })

  it('drops the minus sign from values that round to zero', () => {
    expect(fmtPct(-0.00004, 2)).toBe('0.00%')
    expect(fmtNum(-0.001)).toBe('0.00')
    expect(fmtSignedPct(-0.00001)).toBe('0.0%')
  })

  it('signs non-zero returns', () => {
    expect(fmtSignedPct(0.05)).toBe('+5.0%')
    expect(fmtSignedPct(-0.05)).toBe('-5.0%')
    expect(fmtSignedPct(0.00001)).toBe('0.0%')
  })

  it('echoes settings without padding', () => {
    expect(fmtRate(0.1)).toBe('10%')
    expect(fmtRate(0.125)).toBe('12.5%')
    expect(fmtRate(0.07)).toBe('7%')
    expect(fmtRate(-0.001)).toBe('-0.1%')
    expect(fmtRate(-0.0000001)).toBe('0%')
  })
})

describe('other formatters', () => {
  it('formats ratios, multiples and intervals', () => {
    expect(fmtNum(1.2345)).toBe('1.23')
    expect(fmtNum(1.2346, 3)).toBe('1.235')
    expect(fmtMultiple(1.5)).toBe('1.50×')
    expect(fmtMultiple(2, 1)).toBe('2.0×')
    expect(fmtCI([-0.123, 0.876])).toBe('[-0.12, 0.88]')
    expect(fmtTurnover(2.04)).toBe('2.0×/yr')
  })

  it('formats p-values with a floor', () => {
    expect(fmtPValue(0.0234)).toBe('0.023')
    expect(fmtPValue(0.0004)).toBe('<0.001')
    expect(fmtPValue(0)).toBe('<0.001')
    expect(fmtPValue(1)).toBe('1.000')
  })

  it('formats days, counts, money and elapsed time', () => {
    expect(fmtDays(12.34)).toBe('12.3 d')
    expect(fmtInt(12345.6)).toBe('12,346')
    expect(fmtMoney(123456.7)).toBe('$123,457')
    expect(fmtMoney(-5000)).toBe('-$5,000')
    expect(fmtMoney(123456.7, true)).toBe('$123.5K')
    expect(fmtMs(12.3)).toBe('12 ms')
    expect(fmtMs(1234)).toBe('1.23 s')
  })

  it('explains the execution lag (§2.2)', () => {
    expect(fmtLag(0)).toMatch(/signal close; optimistic/)
    expect(fmtLag(1)).toBe('1 day (fills at the next close)')
    expect(fmtLag(3)).toBe('3 days (fills 3 closes after the signal)')
  })

  it('shortens dates to months for axis ticks', () => {
    expect(fmtMonth('2016-01-04')).toBe('2016-01')
  })
})
