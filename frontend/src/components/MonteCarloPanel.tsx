// Monte Carlo views (§13): fan chart, terminal-return histogram, risk table, seed and timing.
import { memo, useMemo, useState } from 'react'
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { fmtDayCount, fmtInt, fmtMs, fmtPct, fmtSignedPct } from '../format'
import type { Async } from '../state'
import { blockLengthLabel, fanRows, histogramRows, niceTicks, type FanRow, type HistRow } from '../transform'
import { AXIS_TICK, CHART_MARGIN, COLORS, TOOLTIP_STYLE } from '../theme'
import type { ModelResult, MonteCarloResponse } from '../types'
import { AsyncNotice, Panel, Segmented } from './ui'

type ModelKey = 'gbm' | 'bootstrap' | 'stationary'

const MODELS = [
  { value: 'gbm', label: 'GBM', title: 'Geometric Brownian motion: iid normal log returns' },
  { value: 'bootstrap', label: 'Block bootstrap', title: 'Circular moving-block bootstrap of historical log returns' },
  {
    value: 'stationary',
    label: 'Stationary',
    title: 'Stationary bootstrap (Politis & Romano): blocks of random, geometric length',
  },
] as const

// Stable accessors: a range Area takes a dataKey that returns [low, high] (§13).
const band90 = (r: FanRow) => [r.p5, r.p95]
const band50 = (r: FanRow) => [r.p25, r.p75]
const PATH_KEYS = Array.from({ length: 20 }, (_, j) => (r: FanRow) => r.paths[j])

/** Equity multiple -> cumulative return, for axis ticks and tooltips. */
function multipleTip(value: unknown): string {
  if (Array.isArray(value)) return `${fmtSignedPct(value[0] - 1)} to ${fmtSignedPct(value[1] - 1)}`
  return typeof value === 'number' ? fmtSignedPct(value - 1) : ''
}

function FanChart(props: { model: ModelResult; horizon: number }) {
  const rows = useMemo(() => fanRows(props.model), [props.model])
  const nPaths = Math.min(props.model.sample_paths.length, PATH_KEYS.length)
  // Round ticks in return space; the domain covers the bands and every sample path.
  const { domain, ticks } = useMemo(() => {
    const lo = Math.min(...rows.map((r) => Math.min(r.p5, ...r.paths)))
    const hi = Math.max(...rows.map((r) => Math.max(r.p95, ...r.paths)))
    return { domain: [lo, hi], ticks: niceTicks(lo - 1, hi - 1, 5).map((t) => 1 + t) }
  }, [rows])
  return (
    <ComposedChart responsive width="100%" height={300} data={rows} margin={CHART_MARGIN}>
      <CartesianGrid stroke={COLORS.grid} vertical={false} />
      <XAxis
        dataKey="t"
        type="number"
        domain={[0, props.horizon]}
        ticks={niceTicks(0, props.horizon)}
        tick={AXIS_TICK}
        stroke={COLORS.axis}
        tickLine={false}
        tickFormatter={(t: number) => `${t}d`}
      />
      <YAxis
        width={56}
        tick={AXIS_TICK}
        stroke={COLORS.axis}
        tickLine={false}
        domain={domain}
        ticks={ticks}
        tickFormatter={(v: number) => fmtSignedPct(v - 1, 0)}
      />
      <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={multipleTip} labelFormatter={(t) => `day ${t}`} />
      <ReferenceLine y={1} stroke={COLORS.axis} />
      <Area dataKey={band90} name="p5–p95" stroke="none" fill={COLORS.fan} fillOpacity={0.12} dot={false} isAnimationActive={false} />
      <Area dataKey={band50} name="p25–p75" stroke="none" fill={COLORS.fan} fillOpacity={0.25} dot={false} isAnimationActive={false} />
      {PATH_KEYS.slice(0, nPaths).map((key, j) => (
        <Line
          key={j}
          dataKey={key}
          stroke={COLORS.path}
          strokeOpacity={0.3}
          strokeWidth={0.75}
          dot={false}
          activeDot={false}
          tooltipType="none"
          isAnimationActive={false}
        />
      ))}
      <Line dataKey="p50" name="Median" stroke={COLORS.fan} strokeWidth={2} dot={false} isAnimationActive={false} />
    </ComposedChart>
  )
}

function binLabel(row: HistRow): string {
  if (row.tail === 'low') return `below ${fmtSignedPct(row.hi)} (lowest bin holds the whole left tail)`
  if (row.tail === 'high') return `above ${fmtSignedPct(row.lo)} (highest bin holds the whole right tail)`
  return `${fmtSignedPct(row.lo)} to ${fmtSignedPct(row.hi)}`
}

function Histogram(props: { model: ModelResult; edges: number[] }) {
  const rows = useMemo(() => histogramRows(props.model.hist_counts, props.edges), [props.model, props.edges])
  const lo = props.edges[0]
  const hi = props.edges[props.edges.length - 1]
  return (
    <BarChart responsive width="100%" height={300} data={rows} margin={CHART_MARGIN} barCategoryGap={1}>
      <CartesianGrid stroke={COLORS.grid} vertical={false} />
      <XAxis
        dataKey="mid"
        type="number"
        domain={[lo, hi]}
        ticks={niceTicks(lo, hi)}
        tick={AXIS_TICK}
        stroke={COLORS.axis}
        tickLine={false}
        tickFormatter={(v: number) => fmtSignedPct(v, 0)}
      />
      <YAxis width={56} tick={AXIS_TICK} stroke={COLORS.axis} tickLine={false} tickFormatter={(v: number) => fmtInt(v)} />
      <Tooltip
        {...TOOLTIP_STYLE}
        isAnimationActive={false}
        cursor={{ fill: '#ffffff10' }}
        labelFormatter={(_, payload) => {
          const row = payload?.[0]?.payload as HistRow | undefined
          return row ? binLabel(row) : ''
        }}
        formatter={(value, _name, item) => {
          const row = item.payload as HistRow
          return [`${fmtInt(typeof value === 'number' ? value : null)} (${fmtPct(row.share, 1)})`, 'Paths']
        }}
      />
      <Bar dataKey="count" name="Paths" fill={COLORS.hist} fillOpacity={0.75} isAnimationActive={false} />
      <ReferenceLine
        x={-props.model.var95}
        stroke={COLORS.var}
        strokeWidth={2}
        label={{ value: `−VaR95 ${fmtSignedPct(-props.model.var95)}`, position: 'insideTopLeft', fill: '#fca5a5', fontSize: 11 }}
      />
    </BarChart>
  )
}

function RiskTable(props: { data: MonteCarloResponse }) {
  const { gbm, bootstrap, stationary, closed_form_gbm: cf } = props.data
  const rows = [
    { label: 'VaR 95%', key: 'var95' },
    { label: 'CVaR 95%', key: 'cvar95' },
    { label: 'VaR 99%', key: 'var99' },
    { label: 'CVaR 99%', key: 'cvar99' },
  ] as const
  return (
    <table className="w-full text-xs tabular-nums">
      <thead className="text-neutral-400">
        <tr>
          <th className="py-1 text-left font-normal">Loss over the horizon</th>
          <th className="py-1 text-right font-normal">GBM simulated</th>
          <th className="py-1 text-right font-normal">GBM closed form</th>
          <th className="py-1 text-right font-normal">Block bootstrap</th>
          <th className="py-1 text-right font-normal">Stationary bootstrap</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.key} className="border-t border-neutral-800">
            <td className="py-1 text-neutral-400">{r.label}</td>
            <td className="py-1 text-right">{fmtPct(gbm[r.key], 2)}</td>
            <td className="py-1 text-right text-neutral-400">{fmtPct(cf[r.key], 2)}</td>
            <td className="py-1 text-right">{fmtPct(bootstrap[r.key], 2)}</td>
            <td className="py-1 text-right">{fmtPct(stationary[r.key], 2)}</td>
          </tr>
        ))}
        <tr className="border-t border-neutral-800 text-neutral-500">
          <td className="py-1">Block length used</td>
          <td className="py-1 text-right">—</td>
          <td className="py-1 text-right">—</td>
          <td className="py-1 text-right">{fmtDayCount(props.data.resolved.block_len_used.circular)}</td>
          <td className="py-1 text-right" title="The mean of the random block lengths">
            mean {fmtDayCount(props.data.resolved.block_len_used.stationary)}
          </td>
        </tr>
      </tbody>
    </table>
  )
}

function Meta(props: { data: MonteCarloResponse; onUseSeed: (seed: number) => void }) {
  const { resolved: r, gbm, bootstrap, stationary } = props.data
  const item = (label: string, value: string, wide = false) => (
    <div className={wide ? 'col-span-2' : undefined}>
      <dt className="text-[11px] text-neutral-500">{label}</dt>
      <dd className="tabular-nums">{value}</dd>
    </div>
  )
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
      <div>
        <dt className="text-[11px] text-neutral-500">Seed</dt>
        <dd className="flex items-center gap-2 tabular-nums">
          <span className="break-all">{r.seed}</span>
          <button
            type="button"
            className="shrink-0 rounded border border-neutral-700 px-1 text-[10px] text-neutral-400 hover:text-neutral-100"
            title="Put this seed in the sidebar so the next run reproduces these paths"
            onClick={() => props.onUseSeed(r.seed)}
          >
            reuse
          </button>
        </dd>
      </div>
      {item('Block length', blockLengthLabel(r))}
      {item('History', `${fmtInt(r.n_hist)} daily returns (${r.window ?? 'ticker'} window)`)}
      {item('Paths × horizon', `${fmtInt(r.n_paths)} × ${r.horizon} days`)}
      {item('Drift μ (log, daily)', `${fmtPct(r.mu_daily, 3)} · ${fmtPct(r.mu_daily * 252, 1)}/yr${r.zero_drift ? ' (zero drift)' : ''}`)}
      {item('Vol σ (log, daily)', `${fmtPct(r.sigma_daily, 2)} · ${fmtPct(r.sigma_daily * Math.sqrt(252), 1)}/yr`)}
      {item(
        'Elapsed',
        `${fmtMs(props.data.elapsed_ms_total)} total: GBM ${fmtMs(gbm.elapsed_ms)}, block bootstrap ` +
          `${fmtMs(bootstrap.elapsed_ms)}, stationary ${fmtMs(stationary.elapsed_ms)}`,
        true,
      )}
    </dl>
  )
}

export const MonteCarloPanel = memo(function MonteCarloPanel(props: {
  state: Async<MonteCarloResponse>
  onUseSeed: (seed: number) => void
}) {
  const [modelKey, setModelKey] = useState<ModelKey>('gbm')
  const { state } = props
  const data = state.status === 'done' ? state.data : null
  const model = data?.[modelKey]
  return (
    <Panel
      title="Monte Carlo"
      subtitle={
        data
          ? `${fmtInt(data.resolved.n_paths)} paths of the strategy's daily returns, ${data.resolved.horizon} trading days ahead. Bands and paths are cumulative returns.`
          : 'Forward projection of the strategy’s daily returns'
      }
      actions={data && <Segmented label="Model" value={modelKey} options={MODELS} onChange={setModelKey} />}
    >
      <AsyncNotice state={state} what="the Monte Carlo simulation" errorTitle="Monte Carlo failed" />
      {data && model && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="mb-1 text-[11px] tracking-widest text-neutral-500 uppercase">
                Fan chart · p5–p95, p25–p75, median, 20 sample paths
              </p>
              <FanChart model={model} horizon={data.resolved.horizon} />
            </div>
            <div>
              <p className="mb-1 text-[11px] tracking-widest text-neutral-500 uppercase">
                Terminal return · 50 bins shared by the three models
              </p>
              <Histogram model={model} edges={data.hist_edges} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-6">
            <RiskTable data={data} />
            <Meta data={data} onUseSeed={props.onUseSeed} />
          </div>
        </div>
      )}
    </Panel>
  )
})
