// In-sample Sharpe over the parameter grid, as a CSS grid, and its deflated Sharpe ratios (§9, §9.1, §13).
import { memo } from 'react'
import { DASH, fmtInt, fmtNum } from '../format'
import type { Async } from '../state'
import { divergingColor, heatmapBound, heatmapLayout, zEntryLabel } from '../transform'
import type { Deflated, SensitivityResponse } from '../types'
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
      <DeflatedSummary deflated={props.data.deflated} />
    </div>
  )
}

function Stat(props: { label: string; value: string; sub: string; title?: string }) {
  return (
    <div className="rounded border border-neutral-800 bg-neutral-950/60 px-3 py-2" title={props.title}>
      <p className="text-[11px] text-neutral-500">{props.label}</p>
      <p className="text-lg leading-tight text-neutral-100 tabular-nums">{props.value}</p>
      <p className="text-[11px] text-neutral-500 tabular-nums">{props.sub}</p>
    </div>
  )
}

/** §9.1: how likely each Sharpe is to beat the best that this many skill-less trials would show. */
function DeflatedSummary(props: { deflated: Deflated }) {
  const d = props.deflated
  const best = d.best
  const cell = best === null ? '' : `L ${best.L}${best.z_entry === null ? '' : `, ${zEntryLabel(best.z_entry)}`}`
  const dsrTitle = 'Probability that the true Sharpe ratio exceeds SR₀*, the best a grid of skill-less trials would show by luck'
  return (
    <div className="space-y-2 border-t border-neutral-800 pt-3">
      <p className="text-[11px] tracking-widest text-neutral-500 uppercase">Deflated Sharpe ratio</p>
      <div className="grid grid-cols-4 gap-2">
        <Stat label="Trials" value={fmtInt(d.n_trials)} sub="cells with a defined Sharpe" />
        <Stat
          label="SR₀*: expected best if no skill"
          value={fmtNum(d.sr0_star)}
          sub="annualized"
          title="The Sharpe ratio the best of N skill-less trials is expected to show, from the spread of the grid's Sharpe ratios"
        />
        <Stat label="DSR of this run’s cell" value={fmtNum(d.selected_dsr)} sub="P(true Sharpe > SR₀*)" title={dsrTitle} />
        <Stat
          label="DSR of the best cell"
          value={fmtNum(best?.dsr)}
          sub={best === null ? 'no cell has a Sharpe' : `${cell} · Sharpe ${fmtNum(best.sharpe)}`}
          title={dsrTitle}
        />
      </div>
      <p className="text-[11px] leading-snug text-neutral-500">
        Bailey &amp; López de Prado (2014). It counts only this grid’s trials: settings tried before, by hand or in
        earlier runs, are hidden trials it can’t see. It also treats neighbouring cells as independent, although they
        are strongly correlated.
      </p>
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
