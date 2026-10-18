// Small presentational building blocks shared by the views.
import { Component, type ReactNode } from 'react'
import type { Async } from '../state'

/** Contains a rendering error to the one view that threw, instead of blanking the dashboard. */
export class ErrorBoundary extends Component<{ title: string; children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  render() {
    if (this.state.error) {
      return <ErrorBox title={`${this.props.title} could not be drawn`} detail={this.state.error.message} />
    }
    return this.props.children
  }
}

export function Panel(props: { title: string; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded border border-neutral-800 bg-neutral-900/40">
      <header className="flex items-start justify-between gap-4 border-b border-neutral-800 px-4 py-2">
        <div>
          <h2 className="text-xs font-semibold tracking-widest text-neutral-300 uppercase">{props.title}</h2>
          {props.subtitle && <p className="mt-0.5 text-xs text-neutral-500">{props.subtitle}</p>}
        </div>
        {props.actions}
      </header>
      <div className="p-4">{props.children}</div>
    </section>
  )
}

export type Option<T extends string> = { value: T; label: string; title?: string }

export function Segmented<T extends string>(props: {
  value: T
  options: readonly Option<T>[]
  onChange: (value: T) => void
  label: string
}) {
  return (
    <div role="group" aria-label={props.label} className="inline-flex rounded border border-neutral-700 p-0.5">
      {props.options.map((option) => {
        const active = option.value === props.value
        return (
          <button
            key={option.value}
            type="button"
            title={option.title}
            aria-pressed={active}
            onClick={() => props.onChange(option.value)}
            className={
              'rounded px-2 py-0.5 text-xs whitespace-nowrap ' +
              (active ? 'bg-emerald-500/20 text-emerald-300' : 'text-neutral-400 hover:text-neutral-100')
            }
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}

/** An on/off button. Both views that show the benchmark share one of these states (§13). */
export function Toggle(props: { pressed: boolean; onChange: (pressed: boolean) => void; children: ReactNode; title?: string }) {
  return (
    <button
      type="button"
      title={props.title}
      aria-pressed={props.pressed}
      onClick={() => props.onChange(!props.pressed)}
      className={
        'rounded border px-2 py-0.5 text-xs whitespace-nowrap ' +
        (props.pressed
          ? 'border-slate-500 bg-slate-500/20 text-slate-200'
          : 'border-neutral-700 text-neutral-500 hover:text-neutral-200')
      }
    >
      {props.pressed ? '☑' : '☐'} {props.children}
    </button>
  )
}

export function Loading(props: { what: string }) {
  return <p className="animate-pulse text-sm text-neutral-400">Running {props.what}…</p>
}

/** An error state. `detail` is the API's own message (§13), shown verbatim. */
export function ErrorBox(props: { title: string; detail: string; httpStatus?: number }) {
  return (
    <div role="alert" className="rounded border border-red-900 bg-red-950/40 px-3 py-2 text-sm">
      <p className="font-semibold text-red-300">
        {props.title}
        {props.httpStatus ? <span className="ml-2 font-normal text-red-400/70">HTTP {props.httpStatus}</span> : null}
      </p>
      <p className="mt-1 break-words whitespace-pre-wrap text-red-200">{props.detail}</p>
    </div>
  )
}

/** A labelled figure with a line of context under it. */
export function Stat(props: { label: string; value: string; sub: string; title?: string }) {
  return (
    <div className="rounded border border-neutral-800 bg-neutral-950/60 px-3 py-2" title={props.title}>
      <p className="text-[11px] text-neutral-500">{props.label}</p>
      <p className="text-lg leading-tight text-neutral-100 tabular-nums">{props.value}</p>
      <p className="text-[11px] text-neutral-500 tabular-nums">{props.sub}</p>
    </div>
  )
}

export function Muted(props: { children: ReactNode }) {
  return <p className="text-sm text-neutral-500">{props.children}</p>
}

/**
 * The loading, skipped and error states of a request. Renders nothing once the data is in, so a
 * view writes `<AsyncNotice state={s} what="…" />` and then its content when `s.status === 'done'`.
 */
export function AsyncNotice<T>(props: { state: Async<T>; what: string; errorTitle: string }) {
  const { state } = props
  switch (state.status) {
    case 'idle':
      return <Muted>Not run yet.</Muted>
    case 'loading':
      return <Loading what={props.what} />
    case 'skipped':
      return <Muted>{state.reason}</Muted>
    case 'error':
      return <ErrorBox title={props.errorTitle} detail={state.detail} httpStatus={state.httpStatus} />
    case 'done':
      return null
  }
}
