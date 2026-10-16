// KPI grid with an IS | OOS | Full toggle (§13). Benchmark figures sit under each strategy figure.
import { memo, useState, type ReactNode } from 'react'
import {
  DASH,
  fmtCI,
  fmtDays,
  fmtInt,
  fmtMultiple,
  fmtNum,
  fmtPct,
  fmtPValue,
  fmtRate,
  fmtSignedPct,
  fmtTurnover,
  isNum,
} from '../format'
import type { Async } from '../state'
import { benchmarkName, segmentRange } from '../transform'
import type { BacktestResponse, Metrics, MonteCarloResponse, SegmentKey } from '../types'
import { Panel, Segmented } from './ui'

type Kpi = {
  label: string
  value: (m: Metrics) => string
  sub?: (m: Metrics) => string
  bench?: boolean // false: the field is always null for a benchmark (trade and cost fields)
  title?: string
}

const KPIS: readonly Kpi[] = [
  { label: 'Sharpe', value: (m) => fmtNum(m.sharpe), sub: (m) => `CI ${fmtCI(m.sharpe_ci95)}`, title: 'Excess-return Sharpe, annualized, with its 95% CI (Lo 2002; assumes iid returns)' },
  { label: 'Sortino', value: (m) => fmtNum(m.sortino) },
  { label: 'CAGR', value: (m) => fmtPct(m.cagr) },
  { label: 'Total return', value: (m) => fmtSignedPct(m.total_return) },
  { label: 'Ann. vol', value: (m) => fmtPct(m.ann_vol) },
  { label: 'Max drawdown', value: (m) => fmtPct(m.max_drawdown) },
  { label: 'Beta to SPY', value: (m) => fmtNum(m.beta_spy), sub: (m) => `corr ${fmtNum(m.corr_spy)}` },
  { label: 'Win rate', value: (m) => fmtPct(m.win_rate, 0), bench: false, title: 'Share of closed trades with P&L > 0' },
  {
    label: 'Trades closed/open',
    value: (m) => `${fmtInt(m.n_closed_trades)} / ${fmtInt(m.n_open_trades)}`,
    bench: false,
    title: 'Trades entered in this segment',
  },
  { label: 'Avg holding', value: (m) => fmtDays(m.avg_holding_days), bench: false, title: 'Closed trades, trading days' },
  { label: 'Days in market', value: (m) => fmtPct(m.pct_days_in_market, 0) },
  { label: 'Turnover', value: (m) => fmtTurnover(m.ann_turnover), bench: false },
  { label: 'Cost drag', value: (m) => (isNum(m.ann_cost_drag) ? `${fmtPct(m.ann_cost_drag, 2)}/yr` : DASH), bench: false },
  { label: 'Avg gross exposure', value: (m) => fmtMultiple(m.avg_gross_exposure), title: 'Held into each day; flat days count as 0' },
]

function Tile(props: { label: string; value: string; sub?: string; bench?: string; title?: string; tone?: 'warn' }) {
  return (
    <div className="rounded border border-neutral-800 bg-neutral-950/60 px-3 py-2" title={props.title}>
      <p className="text-[11px] text-neutral-500">{props.label}</p>
      <p className={'text-lg leading-tight tabular-nums ' + (props.tone === 'warn' ? 'text-amber-300' : 'text-neutral-100')}>
        {props.value}
      </p>
      {props.sub && <p className="text-[11px] text-neutral-500 tabular-nums">{props.sub}</p>}
      {props.bench !== undefined && <p className="text-[11px] text-slate-400 tabular-nums">{props.bench}</p>}
    </div>
  )
}

function Group(props: { title: string; note?: ReactNode; children: ReactNode }) {
  return (
    <div>
      <p className="mb-1.5 text-[11px] tracking-widest text-neutral-500 uppercase">
        {props.title}
        {props.note && <span className="ml-2 tracking-normal normal-case">{props.note}</span>}
      </p>
      <div className="grid grid-cols-7 gap-2">{props.children}</div>
    </div>
  )
}

const SEGMENTS = [
  { value: 'is', label: 'IS' },
  { value: 'oos', label: 'OOS' },
  { value: 'full', label: 'Full' },
] as const

const RISK_TILES = [
  { label: 'VaR95 GBM', model: 'gbm', key: 'var95' },
  { label: 'VaR95 bootstrap', model: 'bootstrap', key: 'var95' },
  { label: 'CVaR95 GBM', model: 'gbm', key: 'cvar95' },
  { label: 'CVaR95 bootstrap', model: 'bootstrap', key: 'cvar95' },
] as const

function riskValue(mc: Async<MonteCarloResponse>, tile: (typeof RISK_TILES)[number]): string {
  if (mc.status === 'loading') return '…'
  return mc.status === 'done' ? fmtPct(mc.data[tile.model][tile.key]) : DASH
}

export const KpiGrid = memo(function KpiGrid(props: { backtest: BacktestResponse; monteCarlo: Async<MonteCarloResponse> }) {
  const { backtest: bt, monteCarlo: mc } = props
  const [segment, setSegment] = useState<SegmentKey>('full')
  const m = bt.metrics[segment]
  const b = bt.benchmark_metrics[segment]
  const bench = benchmarkName(bt.resolved.strategy, bt.resolved.tickers)
  const { sizing, diagnostics } = bt
  const segmentNote = segmentRange(bt.resolved, bt.ruin_date, segment)

  return (
    <Panel
      title="Performance"
      subtitle={
        <>
          {segmentNote}
          {m && ` · ${fmtInt(m.n_days)} evaluated days`} · benchmark: buy &amp; hold {bench}
        </>
      }
      actions={<Segmented label="Segment" value={segment} options={SEGMENTS} onChange={setSegment} />}
    >
      <div className="space-y-4">
        {m === null || b === null ? (
          <p className="text-sm text-amber-300">
            No out-of-sample segment: the run was ruined in-sample on {bt.ruin_date ?? DASH}, before the split.
          </p>
        ) : (
          <Group title="Strategy" note={<span className="text-slate-400">grey: buy &amp; hold {bench}</span>}>
            {KPIS.map((k) => (
              <Tile
                key={k.label}
                label={k.label}
                value={k.value(m)}
                sub={k.sub?.(m)}
                bench={k.bench === false ? undefined : `B&H ${k.value(b)}`}
                title={k.title}
              />
            ))}
          </Group>
        )}

        {sizing && (
          <Group title="Vol targeting">
            <Tile label="Vol target" value={fmtRate(sizing.vol_target)} sub={`max ${fmtMultiple(sizing.max_leverage, 1)} leverage`} />
            <Tile
              label="Realized vol"
              value={m ? fmtPct(m.ann_vol) : DASH}
              bench={b ? `B&H ${fmtPct(b.ann_vol)}` : undefined}
              title="ann_vol of the selected segment"
            />
            <Tile
              label="Days at cap"
              value={fmtPct(sizing.pct_days_at_cap)}
              sub="of invested days, full window"
              title="Share of invested days held at max_leverage (warning above 25%)"
              tone={sizing.pct_days_at_cap !== null && sizing.pct_days_at_cap > 0.25 ? 'warn' : undefined}
            />
          </Group>
        )}

        {diagnostics && (
          <Group title="Pairs diagnostics" note="independent of the segment toggle">
            <Tile label="Coint p (IS)" value={fmtPValue(diagnostics.coint_p_is)} sub="Engle–Granger" tone={diagnostics.coint_p_is !== null && diagnostics.coint_p_is >= 0.05 ? 'warn' : undefined} />
            <Tile label="Coint p (OOS)" value={fmtPValue(diagnostics.coint_p_oos)} sub="display only" />
            <Tile label="Half-life (IS)" value={fmtDays(diagnostics.half_life_is)} />
            <Tile label="Static β (IS)" value={fmtNum(diagnostics.static_beta_is, 3)} sub={`α ${fmtNum(diagnostics.static_alpha_is, 3)}`} />
          </Group>
        )}

        <Group
          title="Monte Carlo risk"
          note={
            mc.status === 'done'
              ? `latest run: ${mc.data.resolved.horizon}-day horizon, ${mc.data.resolved.window} window, losses as fractions of equity`
              : mc.status === 'error'
                ? 'the Monte Carlo run failed (see below)'
                : 'latest run'
          }
        >
          {RISK_TILES.map((tile) => (
            <Tile
              key={tile.label}
              label={tile.label}
              value={riskValue(mc, tile)}
              title={tile.key === 'var95' ? 'Loss at the 5% quantile of terminal return' : 'Mean loss at or beyond that quantile'}
            />
          ))}
        </Group>
      </div>
    </Panel>
  )
})
