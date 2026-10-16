// Sortable trades table built from `trades` (§13).
import { memo, useMemo, useState } from 'react'
import { DASH, fmtInt, fmtNum, fmtSignedPct } from '../format'
import { sortTrades, type SortDir, type TradeKey } from '../transform'
import type { Trade } from '../types'
import { Panel } from './ui'

type Column = { key: TradeKey; label: string; title?: string; align?: 'right'; cell: (t: Trade) => string }

const COLUMNS: readonly Column[] = [
  { key: 'id', label: '#', align: 'right', cell: (t) => String(t.id) },
  { key: 'side', label: 'Side', cell: (t) => t.side },
  { key: 'segment', label: 'Seg', title: 'The segment the trade was entered in', cell: (t) => t.segment.toUpperCase() },
  { key: 'signal_date', label: 'Signal', title: 'Entry signal date', cell: (t) => t.signal_date },
  { key: 'entry_date', label: 'Entry', title: 'Entry fill date', cell: (t) => t.entry_date },
  { key: 'exit_signal_date', label: 'Exit signal', cell: (t) => t.exit_signal_date ?? DASH },
  { key: 'exit_date', label: 'Exit', title: 'Exit fill date', cell: (t) => t.exit_date ?? DASH },
  { key: 'holding_days', label: 'Days', align: 'right', title: 'Days held (P&L days)', cell: (t) => fmtInt(t.holding_days) },
  { key: 'pnl', label: 'P&L', align: 'right', title: 'Excess return as a fraction of equity, summed', cell: (t) => fmtSignedPct(t.pnl, 2) },
]

const BETA: Column = {
  key: 'beta_at_entry',
  label: 'β entry',
  align: 'right',
  title: 'beta_used on the signal date',
  cell: (t) => fmtNum(t.beta_at_entry, 3),
}

const STATUS: Column = { key: 'open', label: 'Status', title: 'Open trades are excluded from the win rate', cell: (t) => (t.open ? 'open' : 'closed') }

function cellTone(key: TradeKey, t: Trade): string {
  if (key === 'pnl') return t.pnl > 0 ? 'text-emerald-300' : t.pnl < 0 ? 'text-red-300' : ''
  if (key === 'side') return t.side === 'long' ? 'text-emerald-300' : 'text-red-300'
  if (key === 'open') return t.open ? 'text-amber-300' : 'text-neutral-500'
  return ''
}

export const TradesTable = memo(function TradesTable(props: { trades: Trade[]; pairs: boolean }) {
  const [sort, setSort] = useState<{ key: TradeKey; dir: SortDir }>({ key: 'id', dir: 'asc' })
  const sorted = useMemo(() => sortTrades(props.trades, sort.key, sort.dir), [props.trades, sort])
  const columns = props.pairs ? [...COLUMNS, BETA, STATUS] : [...COLUMNS, STATUS]
  const nOpen = props.trades.filter((t) => t.open).length

  const toggle = (key: TradeKey) =>
    setSort((s) => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'asc' }))

  return (
    <Panel
      title="Trades"
      subtitle={`${props.trades.length} trades: ${props.trades.length - nOpen} closed, ${nOpen} open. Click a header to sort.`}
    >
      {props.trades.length === 0 ? (
        <p className="text-sm text-neutral-500">No trades in the evaluation window.</p>
      ) : (
        <div className="max-h-96 overflow-y-auto rounded border border-neutral-800">
          <table className="w-full text-xs tabular-nums">
            <thead className="sticky top-0 bg-neutral-900 text-neutral-400">
              <tr>
                {columns.map((c) => {
                  const active = sort.key === c.key
                  return (
                    <th
                      key={c.key}
                      title={c.title}
                      aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}
                      className={'px-2 py-1.5 font-normal ' + (c.align === 'right' ? 'text-right' : 'text-left')}
                    >
                      <button type="button" className="hover:text-neutral-100" onClick={() => toggle(c.key)}>
                        {c.label}
                        <span className="ml-0.5 text-emerald-400">{active ? (sort.dir === 'asc' ? '↑' : '↓') : ''}</span>
                      </button>
                    </th>
                  )
                })}
              </tr>
            </thead>
            <tbody>
              {sorted.map((t) => (
                <tr key={t.id} className="border-t border-neutral-800/70 hover:bg-neutral-800/40">
                  {columns.map((c) => (
                    <td key={c.key} className={`px-2 py-1 ${c.align === 'right' ? 'text-right' : ''} ${cellTone(c.key, t)}`}>
                      {c.cell(t)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
})
