// Parts shared by the dashboard's Recharts charts (§13). Each is a function that returns the
// Recharts element itself, to be placed directly among a chart's children (as the date axis
// always was), not a component that wraps one.
import { Line, ReferenceArea, ReferenceLine, Tooltip, XAxis, YAxis, type DotItemDotProps } from 'recharts'
import type { ReactElement } from 'react'
import { fmtMoney, fmtMonth, fmtMultiple, fmtNum, fmtPct } from '../format'
import { AXIS_TICK, COLORS, TOOLTIP_STYLE } from '../theme'

/** The value axis' width. The same on every synchronized chart, so their x positions line up. */
export const Y_WIDTH = 64

export const asNumber = (value: unknown) => (typeof value === 'number' ? value : null)
export const moneyTip = (value: unknown) => fmtMoney(asNumber(value))
export const pctTip = (value: unknown) => fmtPct(asNumber(value), 2)
export const numTip = (value: unknown) => fmtNum(asNumber(value), 2)
export const exposureTip = (value: unknown) => fmtMultiple(asNumber(value))

/** The x axis of a chart over the backtest's dates: month labels, thinned to fit. */
export function dateAxis() {
  return <XAxis dataKey="date" tick={AXIS_TICK} tickFormatter={fmtMonth} minTickGap={56} stroke={COLORS.axis} tickLine={false} />
}

type Domain = [number | 'auto', number | 'auto']

/** The y axis: `format` labels the ticks; the domain is automatic unless given. */
export function valueAxis(
  format: (v: number) => string,
  options: { domain?: Domain; ticks?: number[]; width?: number; log?: boolean } = {},
) {
  return (
    <YAxis
      width={options.width ?? Y_WIDTH}
      tick={AXIS_TICK}
      stroke={COLORS.axis}
      tickLine={false}
      domain={options.domain ?? ['auto', 'auto']}
      ticks={options.ticks}
      {...(options.log ? { scale: 'log' as const, allowDataOverflow: true } : {})}
      tickFormatter={format}
    />
  )
}

/** The dark tooltip, with no animation (§13: about 4,000 points per series). */
export function tooltip(formatter: (value: unknown) => string, labelFormatter?: (label: unknown, payload: readonly { payload?: unknown }[] | undefined) => string) {
  return <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={formatter} labelFormatter={labelFormatter} />
}

/** A dashed line at y = 0. */
export function zeroLine() {
  return <ReferenceLine y={0} stroke={COLORS.exitBand} strokeDasharray="2 3" />
}

/** OOS shading from the split date to the end, plus the split line (§13). Nothing without an OOS day. */
export function oosMarks(rows: readonly { date: string }[], firstOos: number, label: boolean): ReactElement[] | null {
  if (firstOos < 0) return null
  const split = rows[firstOos].date
  return [
    <ReferenceArea
      key="oos"
      x1={split}
      x2={rows[rows.length - 1].date}
      fill={COLORS.oos}
      fillOpacity={0.07}
      stroke="none"
      ifOverflow="hidden"
    />,
    <ReferenceLine
      key="split"
      x={split}
      stroke={COLORS.split}
      strokeDasharray="4 3"
      label={label ? { value: `split ${split} · OOS →`, position: 'insideTopLeft', fill: COLORS.tick, fontSize: 10 } : undefined}
    />,
  ]
}

// Trade markers: one Line per marker kind with no stroke, drawing a glyph only where the row has a
// value (the score on that signal date). Recharts passes no cy on null rows.
function longMarker({ cx, cy }: DotItemDotProps) {
  if (cx == null || cy == null) return null
  return <path d={`M${cx},${cy - 6}L${cx + 5},${cy + 3}L${cx - 5},${cy + 3}Z`} fill={COLORS.long} />
}

function shortMarker({ cx, cy }: DotItemDotProps) {
  if (cx == null || cy == null) return null
  return <path d={`M${cx},${cy + 6}L${cx + 5},${cy - 3}L${cx - 5},${cy - 3}Z`} fill={COLORS.short} />
}

function exitMarker({ cx, cy }: DotItemDotProps) {
  if (cx == null || cy == null) return null
  return <circle cx={cx} cy={cy} r={3.5} fill="none" stroke={COLORS.exit} strokeWidth={1.25} />
}

/** The long-entry, short-entry and exit markers, from the rows' entryLong, entryShort and exit. */
export function tradeMarkers(): ReactElement[] {
  const marker = { stroke: 'none', activeDot: false, tooltipType: 'none', isAnimationActive: false } as const
  return [
    <Line key="entryLong" dataKey="entryLong" name="Long entry" dot={longMarker} {...marker} />,
    <Line key="entryShort" dataKey="entryShort" name="Short entry" dot={shortMarker} {...marker} />,
    <Line key="exit" dataKey="exit" name="Exit" dot={exitMarker} {...marker} />,
  ]
}
