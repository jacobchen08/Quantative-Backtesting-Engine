import { ApiError } from './api'

/** The state of one request: what each view renders as its loading, empty or error state (§13). */
export type Async<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'skipped'; reason: string }
  | { status: 'error'; detail: string; httpStatus: number }
  | { status: 'done'; data: T }

/** The error state of a failed request: the API's detail and HTTP status (0 = no response). */
export function failure(error: unknown): { status: 'error'; detail: string; httpStatus: number } {
  if (error instanceof ApiError) return { status: 'error', detail: error.message, httpStatus: error.status }
  return { status: 'error', detail: error instanceof Error ? error.message : String(error), httpStatus: 0 }
}
