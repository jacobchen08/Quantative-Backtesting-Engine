// Equity, drawdown, rolling-statistics and signal charts (§13). They all draw the same rows (one per
// `series` date) and share one syncId, so hovering a day highlights that day in every chart.
import { memo, useMemo, type ReactNode } from 'react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
  type DotItemDotProps,
} from 'recharts'
import { fmtMoney, fmtMonth, fmtMultiple, fmtNum, fmtPct, fmtRate } from '../format'
import { equityLogAxis, fmtDelta, niceTicks, type BenchmarkLabels, type LegView, type SeriesRow } from '../transform'
import { AXIS_TICK, BACKTEST_SYNC_ID, CHART_MARGIN, COLORS, LEG_COLORS, TOOLTIP_STYLE } from '../theme'
import type { BacktestResponse } from '../types'
import { Panel, Segmented, Toggle } from './ui'

const Y_WIDTH = 64 // the same on every synchronized chart, so their x positions line up

const asNumber = (value: unknown) => (typeof value === 'number' ? value : null)
const moneyTip = (value: unknown) => fmtMoney(asNumber(value))
const pctTip = (value: unknown) => fmtPct(asNumber(value), 2)
const numTip = (value: unknown) => fmtNum(asNumber(value), 2)
const multipleTip = (value: unknown) => fmtMultiple(asNumber(value))

function dateAxis() {
  return (
    <XAxis
      dataKey="date"
      tick={AXIS_TICK}
      tickFormatter={fmtMonth}
      minTickGap={56}
      stroke={COLORS.axis}
      tickLine={false}
    />
  )
}

/** OOS shading from the split date to the end, plus the split line (§13). */
function oosMarks(rows: readonly { date: string }[], firstOos: number, label: boolean) {
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

export function ChartTitle(props: { children: ReactNode; legend?: ReactNode }) {
  return (
    <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-4 text-[11px] text-neutral-500">
      <span className="tracking-widest uppercase">{props.children}</span>
      {props.legend && <span className="flex flex-wrap gap-x-3">{props.legend}</span>}
    </div>
  )
}

export function Key(props: { color: string; children: ReactNode; dashed?: boolean; glyph?: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      {props.glyph ? (
        <span style={{ color: props.color }}>{props.glyph}</span>
      ) : (
        <span
          className="inline-block w-4"
          style={{ borderTop: `${props.dashed ? '1px dashed' : '2px solid'} ${props.color}` }}
        />
      )}
      {props.children}
    </span>
  )
}

// ---------------------------------------------------------------- equity and drawdown

const SCALES = [
  { value: 'linear', label: 'Linear' },
  { value: 'log', label: 'Log', title: 'Log scale: ticks at 1, 2 and 5 × 10^k dollars' },
] as const

/** Linear | Log for the equity axis (§13), disabled with a reason when the equity reaches 0. */
function ScaleSwitch(props: { log: boolean; disabled: boolean; onChange: (log: boolean) => void }) {
  const segmented = (
    <Segmented
      label="Equity scale"
      value={props.log ? 'log' : 'linear'}
      options={SCALES}
      onChange={(v) => props.onChange(v === 'log')}
    />
  )
  if (!props.disabled) return segmented
  return (
    <span className="inline-block" title="Log scale is unavailable: this run was ruined, so its equity reaches $0, which a log axis can't show.">
      <fieldset disabled className="pointer-events-none opacity-40">
        {segmented}
      </fieldset>
    </span>
  )
}

export const EquityPanel = memo(function EquityPanel(props: {
  rows: SeriesRow[]
  firstOos: number
  benchmark: BenchmarkLabels
  showBenchmark: boolean
  onShowBenchmark: (show: boolean) => void
  logScale: boolean
  onLogScale: (log: boolean) => void
  ruined: boolean
}) {
  const { rows, firstOos, showBenchmark } = props
  // Round ticks down to the deepest drawdown (at least -1%, so a run that never lost still has an axis).
  const ddFloor = useMemo(() => Math.min(-0.01, ...rows.map((r) => r.drawdown ?? 0)), [rows])
  // A log axis needs positive equity: a ruined run reaches 0, so the toggle is disabled there (§13).
  const logAxis = useMemo(() => (props.ruined ? null : equityLogAxis(rows, showBenchmark)), [rows, showBenchmark, props.ruined])
  const log = props.logScale && logAxis !== null
  return (
    <Panel
      title="Equity and drawdown"
      subtitle="Gaps before the baseline day are intentional: the arrays start before trading does."
      actions={
        <div className="flex items-center gap-2">
          <ScaleSwitch log={log} disabled={logAxis === null} onChange={props.onLogScale} />
          <Toggle pressed={showBenchmark} onChange={props.onShowBenchmark} title="Show or hide the benchmark in the chart and the KPI tiles">
            {props.benchmark.name}
          </Toggle>
        </div>
      }
    >
      <ChartTitle
        legend={
          <>
            <Key color={COLORS.strategy}>Strategy</Key>
            {showBenchmark && <Key color={COLORS.benchmark}>{props.benchmark.name}</Key>}
            <Key color={COLORS.oos} glyph="▮">OOS</Key>
          </>
        }
      >
        Equity
      </ChartTitle>
      <LineChart responsive width="100%" height={280} data={rows} syncId={BACKTEST_SYNC_ID} margin={CHART_MARGIN}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        {dateAxis()}
        {log && logAxis ? (
          <YAxis
            width={Y_WIDTH}
            tick={AXIS_TICK}
            stroke={COLORS.axis}
            tickLine={false}
            scale="log"
            domain={logAxis.domain}
            ticks={logAxis.ticks}
            allowDataOverflow
            tickFormatter={(v: number) => fmtMoney(v, true)}
          />
        ) : (
          <YAxis
            width={Y_WIDTH}
            tick={AXIS_TICK}
            stroke={COLORS.axis}
            tickLine={false}
            domain={['auto', 'auto']}
            tickFormatter={(v: number) => fmtMoney(v, true)}
          />
        )}
        <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={moneyTip} />
        {oosMarks(rows, firstOos, true)}
        <Line dataKey="equity" name="Strategy" stroke={COLORS.strategy} strokeWidth={1.5} dot={false} isAnimationActive={false} />
        {showBenchmark && (
          <Line
            dataKey="benchmark"
            name={props.benchmark.name}
            stroke={COLORS.benchmark}
            strokeWidth={1}
            dot={false}
            isAnimationActive={false}
          />
        )}
      </LineChart>

      <ChartTitle>Drawdown</ChartTitle>
      <AreaChart responsive width="100%" height={130} data={rows} syncId={BACKTEST_SYNC_ID} margin={CHART_MARGIN}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        {dateAxis()}
        <YAxis
          width={Y_WIDTH}
          tick={AXIS_TICK}
          stroke={COLORS.axis}
          tickLine={false}
          domain={[ddFloor, 0]}
          ticks={niceTicks(ddFloor, 0, 4)}
          tickFormatter={(v: number) => fmtPct(v, 0)}
        />
        <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={pctTip} />
        {oosMarks(rows, firstOos, false)}
        <Area
          dataKey="drawdown"
          name="Drawdown"
          stroke={COLORS.drawdown}
          fill={COLORS.drawdown}
          fillOpacity={0.25}
          strokeWidth={1}
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </Panel>
  )
})

// ---------------------------------------------------------------- signal panel

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

export const SignalPanel = memo(function SignalPanel(props: {
  rows: SeriesRow[]
  firstOos: number
  backtest: BacktestResponse
  unplacedMarkers: number
}) {
  const { rows, firstOos, backtest } = props
  const { resolved, sizing } = backtest
  const isTrend = resolved.strategy === 'trend'
  const isPairs = resolved.strategy === 'pairs'
  const kalman = isPairs && resolved.pairs_model === 'kalman'
  const scoreTip = isTrend ? pctTip : numTip
  const zEntry = resolved.z_entry ?? 0
  const zExit = resolved.z_exit ?? 0
  const fills = fillsText(resolved.execution_lag)
  const score = isTrend
    ? `Trailing ${resolved.L}-day excess log return, checked ${resolved.signal_frequency}`
    : kalman
      ? `Kalman innovation z-score (δ = ${fmtDelta(resolved.kalman_delta)}, started from an OLS fit on the first ${resolved.L} days)`
      : `Rolling ${resolved.L}-day ${isPairs ? 'residual ' : ''}z-score`

  return (
    <Panel title="Signal" subtitle={`${score}. Markers sit on signal dates; the order ${fills}.`}>
      <ChartTitle
        legend={
          <>
            <Key color={COLORS.score}>{isTrend ? 'excess return' : 'z'}</Key>
            {isTrend ? (
              <Key color={COLORS.exitBand} dashed>
                zero
              </Key>
            ) : (
              <>
                <Key color={COLORS.entryBand} dashed>
                  ±{zEntry} entry
                </Key>
                <Key color={COLORS.exitBand} dashed>
                  ±{zExit} exit
                </Key>
              </>
            )}
            <Key color={COLORS.long} glyph="▲">
              long entry
            </Key>
            <Key color={COLORS.short} glyph="▼">
              short entry
            </Key>
            <Key color={COLORS.exit} glyph="○">
              exit
            </Key>
          </>
        }
      >
        Score
      </ChartTitle>
      <ComposedChart responsive width="100%" height={230} data={rows} syncId={BACKTEST_SYNC_ID} margin={CHART_MARGIN}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        {dateAxis()}
        <YAxis
          width={Y_WIDTH}
          tick={AXIS_TICK}
          stroke={COLORS.axis}
          tickLine={false}
          domain={['auto', 'auto']}
          tickFormatter={(v: number) => (isTrend ? fmtPct(v, 0) : fmtNum(v, 1))}
        />
        <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={scoreTip} />
        {oosMarks(rows, firstOos, false)}
        {isTrend ? (
          <ReferenceLine y={0} stroke={COLORS.exitBand} strokeDasharray="2 3" />
        ) : (
          [
            <ReferenceLine key="e+" y={zEntry} stroke={COLORS.entryBand} strokeDasharray="4 3" />,
            <ReferenceLine key="e-" y={-zEntry} stroke={COLORS.entryBand} strokeDasharray="4 3" />,
            <ReferenceLine key="x+" y={zExit} stroke={COLORS.exitBand} strokeDasharray="2 3" />,
            <ReferenceLine key="x-" y={-zExit} stroke={COLORS.exitBand} strokeDasharray="2 3" />,
          ]
        )}
        <Line
          dataKey="score"
          name={isTrend ? 'Excess return' : 'z'}
          type={isTrend ? 'stepAfter' : 'linear'}
          stroke={COLORS.score}
          strokeWidth={1}
          dot={false}
          isAnimationActive={false}
        />
        <Line dataKey="entryLong" name="Long entry" stroke="none" dot={longMarker} activeDot={false} tooltipType="none" isAnimationActive={false} />
        <Line dataKey="entryShort" name="Short entry" stroke="none" dot={shortMarker} activeDot={false} tooltipType="none" isAnimationActive={false} />
        <Line dataKey="exit" name="Exit" stroke="none" dot={exitMarker} activeDot={false} tooltipType="none" isAnimationActive={false} />
      </ComposedChart>
      {props.unplacedMarkers > 0 && (
        <p className="text-xs text-amber-300">{props.unplacedMarkers} trade marker(s) fall outside the series and are not drawn.</p>
      )}

      {isPairs && (
        <>
          <ChartTitle
            legend={
              <>
                <Key color={COLORS.beta}>{kalman ? 'filtered β' : 'rolling β'}</Key>
                <Key color={COLORS.betaUsed}>
                  β used ({resolved.hedge_mode === 'rolling' ? (kalman ? 'filtered' : 'rolling') : 'fixed at entry'})
                </Key>
              </>
            }
          >
            Hedge ratio
          </ChartTitle>
          <LineChart responsive width="100%" height={120} data={rows} syncId={BACKTEST_SYNC_ID} margin={CHART_MARGIN}>
            <CartesianGrid stroke={COLORS.grid} vertical={false} />
            {dateAxis()}
            <YAxis width={Y_WIDTH} tick={AXIS_TICK} stroke={COLORS.axis} tickLine={false} domain={['auto', 'auto']} tickFormatter={(v: number) => fmtNum(v, 2)} />
            <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={(v: unknown) => fmtNum(asNumber(v), 3)} />
            {oosMarks(rows, firstOos, false)}
            <Line dataKey="beta" name={kalman ? 'Filtered β' : 'Rolling β'} stroke={COLORS.beta} strokeWidth={1} dot={false} isAnimationActive={false} />
            <Line dataKey="betaUsed" name="β used" type="stepAfter" stroke={COLORS.betaUsed} strokeWidth={1.25} dot={false} isAnimationActive={false} />
          </LineChart>
        </>
      )}

      <GrossExposureChart rows={rows} firstOos={firstOos} cap={sizing?.max_leverage ?? null} />

      {sizing && (
        <>
          <ChartTitle
            legend={
              <Key color={COLORS.vol} dashed>
                {fmtRate(sizing.vol_target)} target
              </Key>
            }
          >
            Vol forecast (annualized EWMA σ)
          </ChartTitle>
          <LineChart responsive width="100%" height={110} data={rows} syncId={BACKTEST_SYNC_ID} margin={CHART_MARGIN}>
            <CartesianGrid stroke={COLORS.grid} vertical={false} />
            {dateAxis()}
            <YAxis width={Y_WIDTH} tick={AXIS_TICK} stroke={COLORS.axis} tickLine={false} domain={[0, 'auto']} tickFormatter={(v: number) => fmtPct(v, 0)} />
            <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={pctTip} />
            {oosMarks(rows, firstOos, false)}
            <ReferenceLine y={sizing.vol_target} stroke={COLORS.vol} strokeDasharray="4 3" />
            <Line dataKey="volForecast" name="Vol forecast" stroke={COLORS.vol} strokeWidth={1} dot={false} isAnimationActive={false} />
          </LineChart>
        </>
      )}
    </Panel>
  )
})

function fillsText(k: number): string {
  return k === 0 ? 'fills at the same close' : `fills ${k} trading day${k === 1 ? '' : 's'} later`
}

function GrossExposureChart(props: { rows: SeriesRow[]; firstOos: number; cap: number | null }) {
  const { rows, firstOos, cap } = props
  return (
    <>
      <ChartTitle
        legend={
          cap !== null && (
            <Key color={COLORS.split} dashed>
              {fmtMultiple(cap, 1)} cap
            </Key>
          )
        }
      >
        Gross exposure (Σ|w| at each close)
      </ChartTitle>
      <AreaChart responsive width="100%" height={110} data={rows} syncId={BACKTEST_SYNC_ID} margin={CHART_MARGIN}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        {dateAxis()}
        <YAxis width={Y_WIDTH} tick={AXIS_TICK} stroke={COLORS.axis} tickLine={false} domain={[0, 'auto']} tickFormatter={(v: number) => fmtMultiple(v, 1)} />
        <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={multipleTip} />
        {oosMarks(rows, firstOos, false)}
        {cap !== null && <ReferenceLine y={cap} stroke={COLORS.split} strokeDasharray="4 3" />}
        <Area
          dataKey="exposure"
          name="Gross exposure"
          type="stepAfter"
          stroke={COLORS.exposure}
          fill={COLORS.exposure}
          fillOpacity={0.2}
          strokeWidth={1}
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </>
  )
}

// ---------------------------------------------------------------- trend-portfolio signal panel (§13, view 3)

type LegChartRow = { date: string; weights: number[]; vols: (number | null)[] }

/** One small score chart for a leg: its trailing excess return, a zero line, and its own trades. */
function LegScoreChart(props: { leg: LegView; color: string; firstOos: number }) {
  const { leg, firstOos } = props
  return (
    <div>
      <ChartTitle>
        <span style={{ color: props.color }}>{leg.ticker}</span>
      </ChartTitle>
      <ComposedChart responsive width="100%" height={110} data={leg.rows} syncId={BACKTEST_SYNC_ID} margin={CHART_MARGIN}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        {dateAxis()}
        <YAxis width={Y_WIDTH} tick={AXIS_TICK} stroke={COLORS.axis} tickLine={false} domain={['auto', 'auto']} tickFormatter={(v: number) => fmtPct(v, 0)} />
        <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={pctTip} />
        {oosMarks(leg.rows, firstOos, false)}
        <ReferenceLine y={0} stroke={COLORS.exitBand} strokeDasharray="2 3" />
        <Line dataKey="score" name={`${leg.ticker} excess return`} type="stepAfter" stroke={props.color} strokeWidth={1} dot={false} isAnimationActive={false} />
        <Line dataKey="entryLong" name="Long entry" stroke="none" dot={longMarker} activeDot={false} tooltipType="none" isAnimationActive={false} />
        <Line dataKey="entryShort" name="Short entry" stroke="none" dot={shortMarker} activeDot={false} tooltipType="none" isAnimationActive={false} />
        <Line dataKey="exit" name="Exit" stroke="none" dot={exitMarker} activeDot={false} tooltipType="none" isAnimationActive={false} />
      </ComposedChart>
    </div>
  )
}

export const PortfolioSignalPanel = memo(function PortfolioSignalPanel(props: {
  rows: SeriesRow[]
  legs: LegView[]
  firstOos: number
  backtest: BacktestResponse
  unplacedMarkers: number
}) {
  const { rows, legs, firstOos, backtest } = props
  const { resolved, sizing } = backtest
  const n = legs.length
  const color = (j: number) => LEG_COLORS[j % LEG_COLORS.length]
  const legRows = useMemo<LegChartRow[]>(
    () => rows.map((r, i) => ({ date: r.date, weights: legs.map((l) => l.weight[i]), vols: legs.map((l) => l.volForecast[i]) })),
    [rows, legs],
  )
  const keys = useMemo(
    () => legs.map((_, j) => ({ weight: (r: LegChartRow) => r.weights[j], vol: (r: LegChartRow) => r.vols[j] })),
    [legs],
  )
  const legend = (
    <>
      {legs.map((l, j) => (
        <Key key={l.ticker} color={color(j)}>
          {l.ticker}
        </Key>
      ))}
    </>
  )
  return (
    <Panel
      title="Signal"
      subtitle={
        `Each leg's trailing ${resolved.L}-day excess log return, checked ${resolved.signal_frequency}, with its own trades. ` +
        `Markers sit on signal dates; the order ${fillsText(resolved.execution_lag)}.`
      }
    >
      <ChartTitle
        legend={
          <>
            <Key color={COLORS.long} glyph="▲">
              long entry
            </Key>
            <Key color={COLORS.short} glyph="▼">
              short entry
            </Key>
            <Key color={COLORS.exit} glyph="○">
              exit
            </Key>
          </>
        }
      >
        Scores by leg
      </ChartTitle>
      <div className="grid grid-cols-2 gap-x-4">
        {legs.map((leg, j) => (
          <LegScoreChart key={leg.ticker} leg={leg} color={color(j)} firstOos={firstOos} />
        ))}
      </div>
      {props.unplacedMarkers > 0 && (
        <p className="text-xs text-amber-300">{props.unplacedMarkers} trade marker(s) fall outside the series and are not drawn.</p>
      )}

      <ChartTitle legend={legend}>Executed weights (each leg at most {sizing ? fmtMultiple(sizing.max_leverage / n, 2) : '—'})</ChartTitle>
      <LineChart responsive width="100%" height={160} data={legRows} syncId={BACKTEST_SYNC_ID} margin={CHART_MARGIN}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        {dateAxis()}
        <YAxis width={Y_WIDTH} tick={AXIS_TICK} stroke={COLORS.axis} tickLine={false} domain={['auto', 'auto']} tickFormatter={(v: number) => fmtMultiple(v, 2)} />
        <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={(v: unknown) => fmtMultiple(asNumber(v), 3)} />
        {oosMarks(legRows, firstOos, false)}
        <ReferenceLine y={0} stroke={COLORS.exitBand} strokeDasharray="2 3" />
        {legs.map((l, j) => (
          <Line key={l.ticker} dataKey={keys[j].weight} name={l.ticker} type="stepAfter" stroke={color(j)} strokeWidth={1} dot={false} isAnimationActive={false} />
        ))}
      </LineChart>

      <GrossExposureChart rows={rows} firstOos={firstOos} cap={sizing?.max_leverage ?? null} />

      <ChartTitle
        legend={
          <>
            {legend}
            {sizing && (
              <Key color={COLORS.vol} dashed>
                {fmtRate(sizing.vol_target)} target per asset
              </Key>
            )}
          </>
        }
      >
        Vol forecasts (annualized EWMA σ of each leg)
      </ChartTitle>
      <LineChart responsive width="100%" height={150} data={legRows} syncId={BACKTEST_SYNC_ID} margin={CHART_MARGIN}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        {dateAxis()}
        <YAxis width={Y_WIDTH} tick={AXIS_TICK} stroke={COLORS.axis} tickLine={false} domain={[0, 'auto']} tickFormatter={(v: number) => fmtPct(v, 0)} />
        <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={pctTip} />
        {oosMarks(legRows, firstOos, false)}
        {sizing && <ReferenceLine y={sizing.vol_target} stroke={COLORS.vol} strokeDasharray="4 3" />}
        {legs.map((l, j) => (
          <Line key={l.ticker} dataKey={keys[j].vol} name={l.ticker} stroke={color(j)} strokeWidth={1} dot={false} isAnimationActive={false} />
        ))}
      </LineChart>
    </Panel>
  )
})

// ---------------------------------------------------------------- rolling statistics (§8.2)

export const RollingPanel = memo(function RollingPanel(props: { rows: SeriesRow[]; firstOos: number }) {
  const { rows, firstOos } = props
  const zero = (
    <Key color={COLORS.exitBand} dashed>
      zero
    </Key>
  )
  return (
    <Panel
      title="Rolling statistics"
      subtitle="Trailing 252 evaluated days (one year). Empty before the first full year, and in windows without a position."
    >
      <ChartTitle legend={zero}>Rolling 1-year Sharpe</ChartTitle>
      <LineChart responsive width="100%" height={150} data={rows} syncId={BACKTEST_SYNC_ID} margin={CHART_MARGIN}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        {dateAxis()}
        <YAxis width={Y_WIDTH} tick={AXIS_TICK} stroke={COLORS.axis} tickLine={false} domain={['auto', 'auto']} tickFormatter={(v: number) => fmtNum(v, 1)} />
        <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={numTip} />
        {oosMarks(rows, firstOos, false)}
        <ReferenceLine y={0} stroke={COLORS.exitBand} strokeDasharray="2 3" />
        <Line dataKey="rollingSharpe" name="Rolling Sharpe" stroke={COLORS.strategy} strokeWidth={1} dot={false} isAnimationActive={false} />
      </LineChart>
      <ChartTitle legend={zero}>Rolling 1-year beta to SPY</ChartTitle>
      <LineChart responsive width="100%" height={130} data={rows} syncId={BACKTEST_SYNC_ID} margin={CHART_MARGIN}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        {dateAxis()}
        <YAxis width={Y_WIDTH} tick={AXIS_TICK} stroke={COLORS.axis} tickLine={false} domain={['auto', 'auto']} tickFormatter={(v: number) => fmtNum(v, 2)} />
        <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={numTip} />
        {oosMarks(rows, firstOos, false)}
        <ReferenceLine y={0} stroke={COLORS.exitBand} strokeDasharray="2 3" />
        <Line dataKey="rollingBeta" name="Rolling beta" stroke={COLORS.score} strokeWidth={1} dot={false} isAnimationActive={false} />
      </LineChart>
    </Panel>
  )
})
