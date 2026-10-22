// The sidebar's building blocks (§13): sections, labelled fields, and inputs that keep the text
// typed, which form.ts parses only when Run is pressed.
import type { ReactNode } from 'react'

export const INPUT =
  'w-full rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm text-neutral-100 ' +
  'placeholder:text-neutral-600 focus:border-emerald-500 focus:outline-none'

const CHIP = 'rounded border px-1.5 py-0.5 text-[11px] '

/** A small preset button; `active` highlights the preset the fields currently match. */
export function Chip(props: { active?: boolean; title?: string; onClick: () => void; children: ReactNode }) {
  const tone =
    props.active === undefined
      ? 'border-neutral-700 text-neutral-400 hover:text-neutral-100' // an action, not a preset
      : props.active
        ? 'border-emerald-600 bg-emerald-500/15 text-emerald-300'
        : 'border-neutral-700 text-neutral-400 hover:border-neutral-500 hover:text-neutral-100'
  return (
    <button type="button" title={props.title} className={CHIP + tone} onClick={props.onClick}>
      {props.children}
    </button>
  )
}

export function Section(props: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section aria-label={props.title} className="space-y-3 border-t border-neutral-800 pt-3">
      <div className="flex items-center justify-between">
        <h3 className="text-[11px] font-semibold tracking-widest text-neutral-500 uppercase">{props.title}</h3>
        {props.aside}
      </div>
      {props.children}
    </section>
  )
}

export function Field(props: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] text-neutral-400">{props.label}</span>
      {props.children}
      {props.hint && <span className="mt-1 block text-[11px] leading-snug text-neutral-500">{props.hint}</span>}
    </label>
  )
}

/** A number typed as text; form.ts parses it on Run. min/max/step are hints for the browser only. */
export function NumberInput(props: {
  value: string
  onChange: (value: string) => void
  min?: number
  max?: number
  step?: number | 'any'
  placeholder?: string
}) {
  return (
    <input
      type="number"
      inputMode="decimal"
      className={INPUT}
      value={props.value}
      min={props.min}
      max={props.max}
      step={props.step ?? 'any'}
      placeholder={props.placeholder}
      onChange={(e) => props.onChange(e.target.value)}
    />
  )
}

export function TickerInput(props: { value: string; onChange: (value: string) => void; placeholder?: string }) {
  return (
    <input
      type="text"
      className={INPUT + ' uppercase'}
      value={props.value}
      placeholder={props.placeholder}
      spellCheck={false}
      autoComplete="off"
      onChange={(e) => props.onChange(e.target.value.toUpperCase())}
    />
  )
}
