/** The state of one request: what each view renders as its loading, empty or error state (§13). */
export type Async<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'skipped'; reason: string }
  | { status: 'error'; detail: string; httpStatus: number }
  | { status: 'done'; data: T }
