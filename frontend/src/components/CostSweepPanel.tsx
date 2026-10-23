// Sharpe against one-way trading cost (§8.1, §13 view 8), for the segment the KPI toggle selects.
// One run gives the whole sweep: positions don't depend on the cost, so excess(c) is linear in c.
import { memo, useMemo } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceLine, XAxis } from 'recharts'
import { fmtBps, fmtNum } from '../format'
import { costSweepView, niceTicks } from '../transform'
import { AXIS_TICK, CHART_MARGIN, COLORS } from '../theme'
import type { BacktestResponse, SegmentKey } from '../types'
import { numTip, tooltip, valueAxis, zeroLine } from './chart'
import { Muted, Panel } from './ui'

const SEGMENT_NAME: Record<SegmentKey, string> = { is: 'in-sample', oos: 'out-of-sample', full: 'full period' }

export const CostSweepPanel = memo(function CostSweepPanel(props: { backtest: BacktestResponse; segment: SegmentKey }) {
  const { backtest: bt, segment } = props
  const cs = bt.cost_sensitivity
  const m = bt.metrics[segment]
  const view = useMemo(() => (cs && m ? costSweepView(cs, segment, m.ann_turnover) : null), [cs, m, segment])
  const own = bt.resolved.cost_bps

  let body
  if (cs === null) {
    body = (
      <Muted>
        Not computed: the run was ruined. Clipping the ruin day at −100% makes returns non-linear in the cost, and a
        different cost would move the ruin day (§8.1).
      </Muted>
    )
  } else if (view === null) {
    body = <Muted>No {SEGMENT_NAME[segment]} segment in this run.</Muted>
  } else {
    const { breakEven } = view
    // A label sits left of its line, or right of it near the y-axis, where it would be cut off.
    const side = (x: number, edge: 'Top' | 'Bottom'): `inside${typeof edge}${'Left' | 'Right'}` =>
      x < view.domainMax / 4 ? `inside${edge}Left` : `inside${edge}Right`
    body = (
      <>
        <LineChart responsive width="100%" height={220} data={view.rows} margin={CHART_MARGIN}>
          <CartesianGrid stroke={COLORS.grid} vertical={false} />
          <XAxis
            dataKey="bps"
            type="number"
            domain={[0, view.domainMax]}
            ticks={niceTicks(0, view.domainMax, 6)}
            tick={AXIS_TICK}
            stroke={COLORS.axis}
            tickLine={false}
            tickFormatter={(v: number) => `${v} bps`}
          />
          {valueAxis((v) => fmtNum(v, 2))}
          {tooltip(numTip, (v) => `${String(v)} bps one-way`)}
          {zeroLine()}
          <ReferenceLine
            x={own}
            stroke={COLORS.split}
            strokeDasharray="4 3"
            label={{ value: `this run ${fmtBps(own)}`, position: side(own, 'Bottom'), fill: COLORS.tick, fontSize: 10 }}
          />
          {breakEven !== null && !view.offChart && (
            <ReferenceLine
              x={breakEven}
              stroke={COLORS.var}
              strokeDasharray="4 3"
              label={{ value: `break-even ${fmtBps(breakEven)}`, position: side(breakEven, 'Top'), fill: COLORS.var, fontSize: 10 }}
            />
          )}
          <Line
            dataKey="sharpe"
            name={`Sharpe (${SEGMENT_NAME[segment]})`}
            stroke={COLORS.strategy}
            strokeWidth={1.5}
            dot={{ r: 3, fill: COLORS.strategy, stroke: 'none' }}
            isAnimationActive={false}
          />
        </LineChart>
        <p className="text-xs text-neutral-400">
          {breakEven === null
            ? view.nullReason === 'no-trades'
              ? 'No break-even cost: the strategy made no trades in this segment.'
              : 'No break-even cost: the strategy loses money before costs, so no cost makes it profitable.'
            : `Break-even ${fmtBps(breakEven)} one-way: the mean excess return is 0 there${view.offChart ? ' (beyond the chart)' : ''}.`}
        </p>
      </>
    )
  }

  return (
    <Panel
      title="Cost sweep"
      subtitle={
        <>
          Sharpe against one-way cost per leg, {SEGMENT_NAME[segment]} (follows the Performance toggle).
          <br />
          Positions are held fixed across costs. A trader facing higher costs would trade less, so the high-cost end
          understates what a re-optimized strategy could earn.
        </>
      }
    >
      {body}
    </Panel>
  )
})
