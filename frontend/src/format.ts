// Display formatting. The API sends decimals (0.01 = 1%); percent formatting happens only here
// (§2.1). Every formatter shows null, undefined, NaN and ±Infinity as an em dash (§13).

export const DASH = '—'

export function isNum(x: number | null | undefined): x is number {
  return typeof x === 'number' && Number.isFinite(x)
}

/** toFixed, without a minus sign on a value that rounds to zero ("-0.00" becomes "0.00"). */
function fixed(x: number, digits: number): string {
  const s = x.toFixed(digits)
  return Number(s) === 0 ? (0).toFixed(digits) : s
}

/** 0.1234 -> "12.3%". */
export function fmtPct(x: number | null | undefined, digits = 1): string {
  return isNum(x) ? `${fixed(x * 100, digits)}%` : DASH
}

/** 0.1234 -> "+12.3%", -0.05 -> "-5.0%", and no sign on zero. */
export function fmtSignedPct(x: number | null | undefined, digits = 1): string {
  if (!isNum(x)) return DASH
  const s = fixed(x * 100, digits)
  return `${Number(s) > 0 ? '+' : ''}${s}%`
}

/** A setting echoed back, without padding: 0.1 -> "10%", 0.125 -> "12.5%". At most 2 decimals. */
export function fmtRate(x: number | null | undefined): string {
  return isNum(x) ? `${Number((x * 100).toFixed(2))}%` : DASH
}

/** 1.2345 -> "1.23". */
export function fmtNum(x: number | null | undefined, digits = 2): string {
  return isNum(x) ? fixed(x, digits) : DASH
}

/** An equity multiple or leverage: 1.5 -> "1.50×". */
export function fmtMultiple(x: number | null | undefined, digits = 2): string {
  return isNum(x) ? `${fixed(x, digits)}×` : DASH
}

/** A 95% confidence interval: [-0.12, 0.88] -> "[-0.12, 0.88]". */
export function fmtCI(ci: readonly [number, number] | null | undefined, digits = 2): string {
  return ci && isNum(ci[0]) && isNum(ci[1]) ? `[${fixed(ci[0], digits)}, ${fixed(ci[1], digits)}]` : DASH
}

/** A p-value: 3 decimals, or "<0.001". */
export function fmtPValue(p: number | null | undefined): string {
  if (!isNum(p)) return DASH
  return p < 0.001 ? '<0.001' : p.toFixed(3)
}

/** Trading days: 12.34 -> "12.3 d". */
export function fmtDays(x: number | null | undefined, digits = 1): string {
  return isNum(x) ? `${fixed(x, digits)} d` : DASH
}

/** A count with thousands separators: 12345 -> "12,345". */
export function fmtInt(x: number | null | undefined): string {
  return isNum(x) ? Math.round(x).toLocaleString('en-US') : DASH
}

/** Dollars: 123456.7 -> "$123,457"; compact for chart axes: "$123.5K". */
export function fmtMoney(x: number | null | undefined, compact = false): string {
  if (!isNum(x)) return DASH
  return x.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: compact ? 1 : 0,
    notation: compact ? 'compact' : 'standard',
  })
}

/** Elapsed time: 12.3 -> "12 ms", 1234 -> "1.23 s". */
export function fmtMs(ms: number | null | undefined): string {
  if (!isNum(ms)) return DASH
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

/** An annual drag as a decimal: 0.00106 -> "0.11%/yr". */
export function fmtPerYear(x: number | null | undefined): string {
  return isNum(x) ? `${fmtPct(x, 2)}/yr` : DASH
}

/** Annual turnover in multiples of equity: 2.04 -> "2.0×/yr". */
export function fmtTurnover(x: number | null | undefined): string {
  return isNum(x) ? `${fixed(x, 1)}×/yr` : DASH
}

/** Execution lag in trading days, with what it means for the fill (§2.2). */
export function fmtLag(k: number): string {
  if (k === 0) return '0 days (fills at the signal close; optimistic)'
  if (k === 1) return '1 day (fills at the next close)'
  return `${k} days (fills ${k} closes after the signal)`
}

/** "2016-01-04" -> "2016-01", for date-axis ticks. */
export function fmtMonth(date: string): string {
  return date.slice(0, 7)
}

/** Basis points: 12.34 -> "12.3 bps". */
export function fmtBps(x: number | null | undefined, digits = 1): string {
  return isNum(x) ? `${fixed(x, digits)} bps` : DASH
}
