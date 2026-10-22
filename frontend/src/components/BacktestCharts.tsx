// Equity, drawdown, rolling-statistics and signal charts (§13). They all draw the same rows (one per
// `series` date) and share one syncId, so hovering a day highlights that day in every chart.
import { memo, useMemo, type ReactNode } from 'react'
import { Area, AreaChart, CartesianGrid, ComposedChart, Line, LineChart, ReferenceLine } from 'recharts'
import { fmtMoney, fmtMultiple, fmtNum, fmtPct, fmtRate } from '../format'
import { equityLogAxis, fmtDelta, niceTicks, type BenchmarkLabels, type LegView, type SeriesRow } from '../transform'
import { BACKTEST_SYNC_ID, CHART_MARGIN, COLORS, LEG_COLORS } from '../theme'
import type { BacktestResponse } from '../types'
import {
  asNumber,
  dateAxis,
  exposureTip,
  moneyTip,
  numTip,
  oosMarks,
  pctTip,
  tooltip,
  tradeMarkers,
  valueAxis,
  zeroLine,
} from './chart'
import { ChartTitle, Key, Panel, Segmented, Toggle } from './ui'

const money = (v: number) => fmtMoney(v, true)

/** A line through the data with no dots or animation (§13: about 4,000 points per series). */
const plain = { dot: false, isAnimationActive: false } as const

/** The legend of the entry and exit markers. */
const MARKER_KEYS = (
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
)

/** One of the backtest's synchronized charts: grid, date axis and the OOS shading included. */
function SyncedChart(props: {
  kind: typeof LineChart | typeof AreaChart | typeof ComposedChart
  rows: readonly { date: string }[]
  firstOos: number
  height: number
  splitLabel?: boolean
  children: ReactNode
}) {
  const Chart = props.kind
  return (
    <Chart responsive width="100%" height={props.height} data={props.rows} syncId={BACKTEST_SYNC_ID} margin={CHART_MARGIN}>
      <CartesianGrid stroke={COLORS.grid} vertical={false} />
      {dateAxis()}
      {oosMarks(props.rows, props.firstOos, props.splitLabel ?? false)}
      {props.children}
    </Chart>
  )
}

function UnplacedMarkers(props: { count: number }) {
  if (props.count === 0) return null
  return <p className="text-xs text-amber-300">{props.count} trade marker(s) fall outside the series and are not drawn.</p>
}

function fillsText(k: number): string {
  return k === 0 ? 'fills at the same close' : `fills ${k} trading day${k === 1 ? '' : 's'} later`
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
            <Key color={COLORS.oos} glyph="▮">
              OOS
            </Key>
          </>
        }
      >
        Equity
      </ChartTitle>
      <SyncedChart kind={LineChart} rows={rows} firstOos={firstOos} height={280} splitLabel>
        {log && logAxis ? valueAxis(money, { log: true, domain: logAxis.domain, ticks: logAxis.ticks }) : valueAxis(money)}
        {tooltip(moneyTip)}
        <Line dataKey="equity" name="Strategy" stroke={COLORS.strategy} strokeWidth={1.5} {...plain} />
        {showBenchmark && <Line dataKey="benchmark" name={props.benchmark.name} stroke={COLORS.benchmark} strokeWidth={1} {...plain} />}
      </SyncedChart>

      <ChartTitle>Drawdown</ChartTitle>
      <SyncedChart kind={AreaChart} rows={rows} firstOos={firstOos} height={130}>
        {valueAxis((v) => fmtPct(v, 0), { domain: [ddFloor, 0], ticks: niceTicks(ddFloor, 0, 4) })}
        {tooltip(pctTip)}
        <Area
          dataKey="drawdown"
          name="Drawdown"
          stroke={COLORS.drawdown}
          fill={COLORS.drawdown}
          fillOpacity={0.25}
          strokeWidth={1}
          {...plain}
        />
      </SyncedChart>
    </Panel>
  )
})

// ---------------------------------------------------------------- signal panel

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
  const zEntry = resolved.z_entry ?? 0
  const zExit = resolved.z_exit ?? 0
  const score = isTrend
    ? `Trailing ${resolved.L}-day excess log return, checked ${resolved.signal_frequency}`
    : kalman
      ? `Kalman innovation z-score (δ = ${fmtDelta(resolved.kalman_delta)}, started from an OLS fit on the first ${resolved.L} days)`
      : `Rolling ${resolved.L}-day ${isPairs ? 'residual ' : ''}z-score`

  return (
    <Panel title="Signal" subtitle={`${score}. Markers sit on signal dates; the order ${fillsText(resolved.execution_lag)}.`}>
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
            {MARKER_KEYS}
          </>
        }
      >
        Score
      </ChartTitle>
      <SyncedChart kind={ComposedChart} rows={rows} firstOos={firstOos} height={230}>
        {valueAxis((v) => (isTrend ? fmtPct(v, 0) : fmtNum(v, 1)))}
        {tooltip(isTrend ? pctTip : numTip)}
        {isTrend
          ? zeroLine()
          : [
              <ReferenceLine key="e+" y={zEntry} stroke={COLORS.entryBand} strokeDasharray="4 3" />,
              <ReferenceLine key="e-" y={-zEntry} stroke={COLORS.entryBand} strokeDasharray="4 3" />,
              <ReferenceLine key="x+" y={zExit} stroke={COLORS.exitBand} strokeDasharray="2 3" />,
              <ReferenceLine key="x-" y={-zExit} stroke={COLORS.exitBand} strokeDasharray="2 3" />,
            ]}
        <Line
          dataKey="score"
          name={isTrend ? 'Excess return' : 'z'}
          type={isTrend ? 'stepAfter' : 'linear'}
          stroke={COLORS.score}
          strokeWidth={1}
          {...plain}
        />
        {tradeMarkers()}
      </SyncedChart>
      <UnplacedMarkers count={props.unplacedMarkers} />

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
          <SyncedChart kind={LineChart} rows={rows} firstOos={firstOos} height={120}>
            {valueAxis((v) => fmtNum(v, 2))}
            {tooltip((v) => fmtNum(asNumber(v), 3))}
            <Line dataKey="beta" name={kalman ? 'Filtered β' : 'Rolling β'} stroke={COLORS.beta} strokeWidth={1} {...plain} />
            <Line dataKey="betaUsed" name="β used" type="stepAfter" stroke={COLORS.betaUsed} strokeWidth={1.25} {...plain} />
          </SyncedChart>
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
          <SyncedChart kind={LineChart} rows={rows} firstOos={firstOos} height={110}>
            {valueAxis((v) => fmtPct(v, 0), { domain: [0, 'auto'] })}
            {tooltip(pctTip)}
            <ReferenceLine y={sizing.vol_target} stroke={COLORS.vol} strokeDasharray="4 3" />
            <Line dataKey="volForecast" name="Vol forecast" stroke={COLORS.vol} strokeWidth={1} {...plain} />
          </SyncedChart>
        </>
      )}
    </Panel>
  )
})

function GrossExposureChart(props: { rows: SeriesRow[]; firstOos: number; cap: number | null }) {
  const { cap } = props
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
      <SyncedChart kind={AreaChart} rows={props.rows} firstOos={props.firstOos} height={110}>
        {valueAxis((v) => fmtMultiple(v, 1), { domain: [0, 'auto'] })}
        {tooltip(exposureTip)}
        {cap !== null && <ReferenceLine y={cap} stroke={COLORS.split} strokeDasharray="4 3" />}
        <Area
          dataKey="exposure"
          name="Gross exposure"
          type="stepAfter"
          stroke={COLORS.exposure}
          fill={COLORS.exposure}
          fillOpacity={0.2}
          strokeWidth={1}
          {...plain}
        />
      </SyncedChart>
    </>
  )
}

// ---------------------------------------------------------------- trend-portfolio signal panel (§13, view 3)

type LegChartRow = { date: string; weights: number[]; vols: (number | null)[] }

/** One small score chart for a leg: its trailing excess return, a zero line, and its own trades. */
function LegScoreChart(props: { leg: LegView; color: string; firstOos: number }) {
  const { leg } = props
  return (
    <div>
      <ChartTitle>
        <span style={{ color: props.color }}>{leg.ticker}</span>
      </ChartTitle>
      <SyncedChart kind={ComposedChart} rows={leg.rows} firstOos={props.firstOos} height={110}>
        {valueAxis((v) => fmtPct(v, 0))}
        {tooltip(pctTip)}
        {zeroLine()}
        <Line dataKey="score" name={`${leg.ticker} excess return`} type="stepAfter" stroke={props.color} strokeWidth={1} {...plain} />
        {tradeMarkers()}
      </SyncedChart>
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
  const color = (j: number) => LEG_COLORS[j % LEG_COLORS.length]
  const legRows = useMemo<LegChartRow[]>(
    () => rows.map((r, i) => ({ date: r.date, weights: legs.map((l) => l.weight[i]), vols: legs.map((l) => l.volForecast[i]) })),
    [rows, legs],
  )
  const keys = useMemo(
    () => legs.map((_, j) => ({ weight: (r: LegChartRow) => r.weights[j], vol: (r: LegChartRow) => r.vols[j] })),
    [legs],
  )
  const legend = legs.map((l, j) => (
    <Key key={l.ticker} color={color(j)}>
      {l.ticker}
    </Key>
  ))
  return (
    <Panel
      title="Signal"
      subtitle={
        `Each leg's trailing ${resolved.L}-day excess log return, checked ${resolved.signal_frequency}, with its own trades. ` +
        `Markers sit on signal dates; the order ${fillsText(resolved.execution_lag)}.`
      }
    >
      <ChartTitle legend={MARKER_KEYS}>Scores by leg</ChartTitle>
      <div className="grid grid-cols-2 gap-x-4">
        {legs.map((leg, j) => (
          <LegScoreChart key={leg.ticker} leg={leg} color={color(j)} firstOos={firstOos} />
        ))}
      </div>
      <UnplacedMarkers count={props.unplacedMarkers} />

      <ChartTitle legend={legend}>
        Executed weights (each leg at most {sizing ? fmtMultiple(sizing.max_leverage / legs.length, 2) : '—'})
      </ChartTitle>
      <SyncedChart kind={LineChart} rows={legRows} firstOos={firstOos} height={160}>
        {valueAxis((v) => fmtMultiple(v, 2))}
        {tooltip((v) => fmtMultiple(asNumber(v), 3))}
        {zeroLine()}
        {legs.map((l, j) => (
          <Line key={l.ticker} dataKey={keys[j].weight} name={l.ticker} type="stepAfter" stroke={color(j)} strokeWidth={1} {...plain} />
        ))}
      </SyncedChart>

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
      <SyncedChart kind={LineChart} rows={legRows} firstOos={firstOos} height={150}>
        {valueAxis((v) => fmtPct(v, 0), { domain: [0, 'auto'] })}
        {tooltip(pctTip)}
        {sizing && <ReferenceLine y={sizing.vol_target} stroke={COLORS.vol} strokeDasharray="4 3" />}
        {legs.map((l, j) => (
          <Line key={l.ticker} dataKey={keys[j].vol} name={l.ticker} stroke={color(j)} strokeWidth={1} {...plain} />
        ))}
      </SyncedChart>
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
      <SyncedChart kind={LineChart} rows={rows} firstOos={firstOos} height={150}>
        {valueAxis((v) => fmtNum(v, 1))}
        {tooltip(numTip)}
        {zeroLine()}
        <Line dataKey="rollingSharpe" name="Rolling Sharpe" stroke={COLORS.strategy} strokeWidth={1} {...plain} />
      </SyncedChart>
      <ChartTitle legend={zero}>Rolling 1-year beta to SPY</ChartTitle>
      <SyncedChart kind={LineChart} rows={rows} firstOos={firstOos} height={130}>
        {valueAxis((v) => fmtNum(v, 2))}
        {tooltip(numTip)}
        {zeroLine()}
        <Line dataKey="rollingBeta" name="Rolling beta" stroke={COLORS.score} strokeWidth={1} {...plain} />
      </SyncedChart>
    </Panel>
  )
})
