// In-sample Sharpe over the parameter grid, as a CSS grid (§9, §13).
import { memo } from 'react'
import { DASH, fmtInt, fmtNum } from '../format'
import type { Async } from '../state'
import { divergingColor, heatmapBound, heatmapLayout } from '../transform'
import type { SensitivityResponse } from '../types'
import { AsyncNotice, Panel } from './ui'

function Legend(props: { bound: number }) {
  const stops = [-1, -0.5, 0, 0.5, 1].map((t) => divergingColor(t * props.bound, props.bound))
  return (
    <div className="flex items-center gap-2 text-[11px] text-neutral-500">
      <span>{fmtNum(-props.bound, 2)}</span>
      <span className="h-2 w-40 rounded" style={{ background: `linear-gradient(to right, ${stops.join(', ')})` }} />
      <span>{fmtNum(props.bound, 2)}</span>
      <span>IS Sharpe (colour centred on 0)</span>
    </div>
  )
}

function Heatmap(props: { data: SensitivityResponse }) {
  const layout = heatmapLayout(props.data)
  const bound = heatmapBound(layout.cells)
  const nCols = layout.colLabels.length
  return (
    <div className="space-y-3">
      <div
        className="grid gap-1 text-xs tabular-nums"
        style={{ gridTemplateColumns: `max-content repeat(${nCols}, minmax(4rem, 1fr))` }}
      >
        <div className="self-end pr-2 text-[11px] text-neutral-500">
          {layout.rowAxis && `${layout.rowAxis} ↓`} {layout.colAxis} →
        </div>
        {layout.colLabels.map((label) => (
          <div key={label} className="text-center text-[11px] text-neutral-400">
            {label}
          </div>
        ))}
        {layout.cells.map((row, i) => [
          <div key={`label-${layout.rowLabels[i]}`} className="flex items-center pr-2 text-[11px] text-neutral-400">
            {layout.rowLabels[i]}
          </div>,
          ...row.map((cell) => (
            <div
              key={`${cell.L}-${cell.zEntry}`}
              title={`L = ${cell.L}${cell.zEntry === null ? '' : `, z entry = ${cell.zEntry}`}: IS Sharpe ${fmtNum(cell.sharpe)}${cell.selected ? ' (this run)' : ''}`}
              className="flex h-10 items-center justify-center rounded text-neutral-50"
              style={{
                background: divergingColor(cell.sharpe, bound),
                outline: cell.selected ? '2px solid #fafafa' : undefined,
                outlineOffset: cell.selected ? -2 : undefined,
              }}
            >
              {cell.sharpe === null ? DASH : fmtNum(cell.sharpe)}
            </div>
          )),
        ])}
      </div>
      <Legend bound={bound} />
    </div>
  )
}

export const SensitivityHeatmap = memo(function SensitivityHeatmap(props: { state: Async<SensitivityResponse> }) {
  const { state } = props
  const data = state.status === 'done' ? state.data : null
  return (
    <Panel
      title="Parameter sensitivity"
      subtitle={
        data ? (
          <>
            {fmtInt(data.n_configs_tested)} configs tested · IS window {data.window.start} → before {data.window.end}
            {data.selected === null ? ' · this run’s parameters are not in the grid' : ' · outlined: this run'}
            <br />
            Scored on the common window shared by every config, so it can differ slightly from the KPI grid’s IS
            window. Out-of-sample results are never used here.
          </>
        ) : (
          'In-sample Sharpe over the default grid'
        )
      }
    >
      <AsyncNotice state={state} what="the sensitivity grid" errorTitle="Sensitivity grid failed" />
      {data && <Heatmap data={data} />}
    </Panel>
  )
})
