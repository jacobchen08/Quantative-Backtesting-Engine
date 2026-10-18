// Chart colours and shared Recharts props for the dark terminal theme (§13).
import type { CSSProperties } from 'react'

export const COLORS = {
  strategy: '#34d399', // emerald-400
  benchmark: '#94a3b8', // slate-400
  drawdown: '#f87171', // red-400
  score: '#60a5fa', // blue-400
  entryBand: '#f59e0b', // amber-500: ±z_entry
  exitBand: '#737373', // neutral-500: ±z_exit and zero lines
  beta: '#c084fc', // purple-400
  betaUsed: '#f472b6', // pink-400
  exposure: '#34d399',
  vol: '#fbbf24', // amber-400
  long: '#34d399',
  short: '#f87171',
  exit: '#e5e5e5', // neutral-200
  oos: '#38bdf8', // sky-400, drawn at low opacity
  split: '#a3a3a3', // neutral-400
  grid: '#262626', // neutral-800
  axis: '#525252', // neutral-600
  tick: '#a3a3a3', // neutral-400
  fan: '#34d399',
  path: '#d4d4d4', // neutral-300, drawn at low opacity
  hist: '#60a5fa',
  var: '#ef4444', // red-500
  fixed: '#60a5fa', // blue-400: the walk-forward's fixed-parameter comparison
} as const

/** Every backtest chart shares this syncId and the same `series` rows, so their indexes line up (§13). */
export const BACKTEST_SYNC_ID = 'backtest'

export const AXIS_TICK = { fill: COLORS.tick, fontSize: 11 }

export const CHART_MARGIN = { top: 8, right: 16, bottom: 0, left: 0 }

export const TOOLTIP_STYLE: { contentStyle: CSSProperties; labelStyle: CSSProperties; itemStyle: CSSProperties } = {
  contentStyle: {
    background: '#0a0a0a',
    border: '1px solid #404040',
    borderRadius: 4,
    fontSize: 12,
    padding: '6px 8px',
  },
  labelStyle: { color: '#e5e5e5', marginBottom: 2 },
  itemStyle: { padding: 0 },
}
