// Walk-forward analysis (§9.2, §13 view 11): the stitched out-of-sample track record against the
// fixed-parameter run and the benchmark, with alternate test folds shaded, then a metrics table and
// the fold table. It has its own Run button, because it backtests every grid configuration.
import { memo, useMemo } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceArea, Tooltip, XAxis, YAxis } from 'recharts'
import {
  DASH,
  fmtCI,
  fmtInt,
  fmtMoney,
  fmtMonth,
  fmtMultiple,
  fmtNum,
  fmtPct,
  fmtPerYear,
  fmtSignedPct,
  fmtTurnover,
} from '../format'
import { STRATEGY_LABEL } from '../form'
import type { Async } from '../state'
import {
  benchmarkLabels,
  configLabel,
  foldBands,
  selectionChanges,
  walkForwardRows,
  type BenchmarkLabels,
  type FoldBand,
  type WalkForwardRow,
} from '../transform'
import { AXIS_TICK, CHART_MARGIN, COLORS, TOOLTIP_STYLE } from '../theme'
import type { Metrics, WalkForwardResponse } from '../types'
import { ChartTitle, Key } from './BacktestCharts'
import { AsyncNotice, Muted, Panel, Stat } from './ui'

type Row = { label: string; value: (m: Metrics) => string; title?: string }

const METRIC_ROWS: readonly Row[] = [
  { label: 'Sharpe', value: (m) => fmtNum(m.sharpe), title: 'Excess-return Sharpe, annualized' },
  { label: 'Sharpe 95% CI', value: (m) => fmtCI(m.sharpe_ci95), title: 'Lo (2002); assumes iid returns' },
  { label: 'Sortino', value: (m) => fmtNum(m.sortino) },
  { label: 'CAGR', value: (m) => fmtPct(m.cagr) },
  { label: 'Total return', value: (m) => fmtSignedPct(m.total_return) },
  { label: 'Ann. vol', value: (m) => fmtPct(m.ann_vol) },
  { label: 'Max drawdown', value: (m) => fmtPct(m.max_drawdown) },
  { label: 'Beta to SPY', value: (m) => fmtNum(m.beta_spy) },
  { label: 'Days in market', value: (m) => fmtPct(m.pct_days_in_market, 0) },
  { label: 'Avg gross exposure', value: (m) => fmtMultiple(m.avg_gross_exposure) },
  { label: 'Turnover', value: (m) => fmtTurnover(m.ann_turnover) },
  { label: 'Cost drag', value: (m) => fmtPerYear(m.ann_cost_drag) },
  { label: 'Borrow drag', value: (m) => fmtPerYear(m.ann_borrow_drag) },
  {
    label: 'Trades closed / open',
    value: (m) => (m.n_closed_trades === null ? DASH : `${fmtInt(m.n_closed_trades)} / ${fmtInt(m.n_open_trades)}`),
    title: 'Trades entered on these days. The walk-forward has none: its positions span configurations.',
  },
  { label: 'Win rate', value: (m) => fmtPct(m.win_rate, 0) },
]

const asNumber = (value: unknown) => (typeof value === 'number' ? value : null)
const moneyTip = (value: unknown) => fmtMoney(asNumber(value))

/** The tooltip's date line, with the fold that owns the day. */
function foldLabel(date: unknown, items: readonly { payload?: unknown }[] | undefined): string {
  const row = items?.[0]?.payload as WalkForwardRow | undefined
  return row?.fold == null ? String(date) : `${String(date)} · fold ${row.fold}`
}

function EquityChart(props: {
  rows: WalkForwardRow[]
  bands: FoldBand[]
  fixedLabel: string
  benchmark: BenchmarkLabels
  showBenchmark: boolean
}) {
  return (
    <LineChart responsive width="100%" height={300} data={props.rows} margin={CHART_MARGIN}>
      <CartesianGrid stroke={COLORS.grid} vertical={false} />
      <XAxis dataKey="date" tick={AXIS_TICK} tickFormatter={fmtMonth} minTickGap={56} stroke={COLORS.axis} tickLine={false} />
      <YAxis
        width={64}
        tick={AXIS_TICK}
        stroke={COLORS.axis}
        tickLine={false}
        domain={['auto', 'auto']}
        tickFormatter={(v: number) => fmtMoney(v, true)}
      />
      <Tooltip {...TOOLTIP_STYLE} isAnimationActive={false} formatter={moneyTip} labelFormatter={foldLabel} />
      {props.bands.map((band) => (
        <ReferenceArea
          key={band.fold}
          x1={band.x1}
          x2={band.x2}
          fill={COLORS.oos}
          fillOpacity={0.08}
          stroke="none"
          ifOverflow="hidden"
        />
      ))}
      <Line dataKey="equity" name="Walk-forward" stroke={COLORS.strategy} strokeWidth={1.5} dot={false} isAnimationActive={false} />
      <Line
        dataKey="fixed"
        name={`Fixed: ${props.fixedLabel}`}
        stroke={COLORS.fixed}
        strokeWidth={1}
        dot={false}
        isAnimationActive={false}
      />
      {props.showBenchmark && (
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
  )
}

function MetricsTable(props: { data: WalkForwardResponse; fixedLabel: string; benchmark: BenchmarkLabels; showBenchmark: boolean }) {
  const { data } = props
  const columns: { label: string; metrics: Metrics | null; tone: string }[] = [
    { label: 'Walk-forward', metrics: data.metrics, tone: 'text-neutral-100' },
    { label: `Fixed: ${props.fixedLabel}`, metrics: data.fixed_metrics, tone: 'text-neutral-300' },
    ...(props.showBenchmark ? [{ label: props.benchmark.short, metrics: data.benchmark_metrics, tone: 'text-slate-400' }] : []),
  ]
  return (
    <table className="w-full text-xs tabular-nums">
      <thead className="text-[11px] text-neutral-400">
        <tr>
          <th className="px-2 py-1 text-left font-normal" />
          {columns.map((c) => (
            <th key={c.label} className="px-2 py-1 text-right font-normal">
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {METRIC_ROWS.map((row) => (
          <tr key={row.label} className="border-t border-neutral-800/70">
            <th title={row.title} className="px-2 py-0.5 text-left font-normal text-neutral-400">
              {row.label}
            </th>
            {columns.map((c) => (
              <td key={c.label} className={`px-2 py-0.5 text-right ${c.tone}`}>
                {c.metrics === null ? DASH : row.value(c.metrics)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

const sharpeTone = (x: number | null) => (x === null ? 'text-neutral-500' : x < 0 ? 'text-red-300' : 'text-emerald-300')

function FoldTable(props: { data: WalkForwardResponse }) {
  const { folds } = props.data
  const changed = useMemo(() => selectionChanges(folds), [folds])
  return (
    <div className="max-h-96 overflow-y-auto rounded border border-neutral-800">
      <table className="w-full text-xs tabular-nums">
        <thead className="sticky top-0 bg-neutral-900 text-[11px] text-neutral-400">
          <tr>
            <th className="px-2 py-1.5 text-right font-normal">Fold</th>
            <th className="px-2 py-1.5 text-left font-normal">Training window</th>
            <th className="px-2 py-1.5 text-left font-normal">Test window</th>
            <th className="px-2 py-1.5 text-left font-normal" title="The highest training Sharpe; ↻ marks a change from the fold before">
              Chosen
            </th>
            <th className="px-2 py-1.5 text-right font-normal" title="The chosen configuration’s own Sharpe on the training window">
              Train Sharpe
            </th>
            <th className="px-2 py-1.5 text-right font-normal" title="The walk-forward’s Sharpe on the days this fold owns, after switching costs">
              Test Sharpe
            </th>
          </tr>
        </thead>
        <tbody>
          {folds.map((fold, i) => (
            <tr key={fold.test_start} className="border-t border-neutral-800/70 hover:bg-neutral-800/40">
              <td className="px-2 py-1 text-right text-neutral-500">{i}</td>
              <td className="px-2 py-1 text-neutral-400">
                {fold.train_start} → {fold.train_end}
              </td>
              <td className="px-2 py-1">
                {fold.test_start} → {fold.test_end}
              </td>
              <td
                className={'px-2 py-1 ' + (fold.selected === null ? 'text-amber-300' : changed[i] ? 'text-sky-300' : 'text-neutral-200')}
                title={fold.selected === null ? 'No configuration was eligible, so the walk-forward was flat' : undefined}
              >
                {changed[i] ? '↻ ' : ''}
                {configLabel(fold.selected)}
              </td>
              <td className="px-2 py-1 text-right">{fmtNum(fold.train_sharpe)}</td>
              <td className={`px-2 py-1 text-right ${sharpeTone(fold.test_sharpe)}`}>{fmtNum(fold.test_sharpe)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Results(props: { data: WalkForwardResponse; showBenchmark: boolean }) {
  const { data, showBenchmark } = props
  const r = data.resolved
  const fixedLabel = configLabel({ L: r.L, z_entry: r.z_entry })
  const benchmark = benchmarkLabels(r.strategy, r.tickers)
  const rows = useMemo(() => walkForwardRows(data.series), [data.series])
  const bands = useMemo(() => foldBands(data.series), [data.series])
  const dates = data.series.dates
  const changes = selectionChanges(data.folds).filter(Boolean).length
  return (
    <div className="space-y-4">
      {data.warnings.length > 0 && (
        <ul role="status" className="list-disc space-y-0.5 rounded border border-amber-800/70 bg-amber-950/30 py-2 pr-3 pl-7 text-xs text-amber-200">
          {data.warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
      <div className="grid grid-cols-4 gap-2">
        <Stat label="Out-of-sample days" value={fmtInt(data.metrics.n_days)} sub={`${dates[1]} → ${dates[dates.length - 1]}`} />
        <Stat label="Folds" value={fmtInt(data.folds.length)} sub={`the choice changed in ${changes}`} />
        <Stat
          label="Distinct configurations chosen"
          value={`${fmtInt(data.n_distinct_selected)} of ${fmtInt(data.n_configs)}`}
          sub="many means unstable parameters"
          title="A high count means the training Sharpe ratios are noisy, not that the strategy adapts"
        />
        <Stat
          label="Sharpe: walk-forward / fixed"
          value={`${fmtNum(data.metrics.sharpe)} / ${fmtNum(data.fixed_metrics?.sharpe)}`}
          sub={`fixed: ${fixedLabel}, never re-chosen`}
        />
      </div>

      <div>
        <ChartTitle
          legend={
            <>
              <Key color={COLORS.strategy}>Walk-forward</Key>
              <Key color={COLORS.fixed}>Fixed: {fixedLabel}</Key>
              {showBenchmark && <Key color={COLORS.benchmark}>{benchmark.name}</Key>}
              <Key color={COLORS.oos} glyph="▮">
                alternate test folds
              </Key>
            </>
          }
        >
          Equity: the test windows stitched together
        </ChartTitle>
        <EquityChart rows={rows} bands={bands} fixedLabel={fixedLabel} benchmark={benchmark} showBenchmark={showBenchmark} />
      </div>

      <div className="grid gap-4 2xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <div className="max-w-3xl space-y-2 2xl:max-w-none">
          <MetricsTable data={data} fixedLabel={fixedLabel} benchmark={benchmark} showBenchmark={showBenchmark} />
          {data.fixed_metrics === null && (
            <p className="text-[11px] text-amber-300">
              The fixed configuration was ruined before the walk-forward’s first day, so it has no figures here.
            </p>
          )}
        </div>
        <FoldTable data={data} />
      </div>

      <p className="text-[11px] leading-snug text-neutral-500">
        The walk-forward track record is entirely out-of-sample. Each fold’s configuration was chosen on its training
        window, from returns through the close before its test window, and switching configuration is charged as a
        trade. Fixed holds the sidebar’s configuration, {fixedLabel}, on the same days and never re-chooses it. The
        grid, the window lengths and the strategy itself were still your choice, so choosing them after seeing this
        result makes it in-sample again.
      </p>
    </div>
  )
}

export const WalkForwardPanel = memo(function WalkForwardPanel(props: {
  state: Async<WalkForwardResponse>
  onRun: () => void
  stale: boolean
  showBenchmark: boolean
}) {
  const { state } = props
  const data = state.status === 'done' ? state.data : null
  const running = state.status === 'loading'
  const r = data?.resolved
  return (
    <Panel
      title="Walk-forward"
      subtitle={
        r ? (
          <>
            {STRATEGY_LABEL[r.strategy]} {r.tickers.join(' / ')} · {r.train_days}-day {r.anchored ? 'anchored' : 'rolling'}{' '}
            training windows, {r.test_days}-day test windows · the best of {fmtInt(data.n_configs)} configurations, re-chosen
            for each of {fmtInt(data.folds.length)} folds
          </>
        ) : (
          'Re-chooses the parameter grid’s best in-sample configuration on each training window, then trades it on the next test window.'
        )
      }
      actions={
        <div className="flex items-center gap-2">
          {props.stale && !running && <span className="text-[11px] whitespace-nowrap text-amber-400/80">Settings changed since this ran</span>}
          <button
            type="button"
            onClick={props.onRun}
            disabled={running}
            className="rounded bg-emerald-600 px-2.5 py-1 text-xs font-semibold tracking-wider whitespace-nowrap text-neutral-950 uppercase hover:bg-emerald-500 disabled:opacity-60"
          >
            {running ? 'Running…' : 'Run walk-forward'}
          </button>
        </div>
      }
    >
      {state.status === 'idle' ? (
        <Muted>
          Not run yet. It backtests every configuration of the parameter grid with the sidebar’s settings and the
          walk-forward windows, so it has its own button.
        </Muted>
      ) : (
        <AsyncNotice state={state} what="the walk-forward" errorTitle="Walk-forward failed" />
      )}
      {data && <Results data={data} showBenchmark={props.showBenchmark} />}
    </Panel>
  )
})
