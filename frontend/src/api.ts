// Typed fetch wrappers for the API's endpoints (§11.1): four POSTs and, from Phase 8, the GETs of
// the health check and the run history (§11.4). Every error the API returns is
// {"detail": "..."}; ApiError carries that string so the views can show it verbatim (§13).
import type {
  BacktestRequest,
  BacktestResponse,
  MonteCarloRequest,
  MonteCarloResponse,
  RunSummary,
  SensitivityRequest,
  StoredRun,
  SensitivityResponse,
  WalkForwardRequest,
  WalkForwardResponse,
} from './types'

// Empty = same origin. In development the Vite proxy forwards /api to uvicorn (vite.config.ts).
export const API_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '')

export class ApiError extends Error {
  /** HTTP status, or 0 when the request never got a response. */
  readonly status: number

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
  }
}

/** The `detail` string of an error body, or a fallback that names the HTTP status. */
export function errorDetail(status: number, statusText: string, body: unknown): string {
  if (typeof body === 'object' && body !== null && 'detail' in body) {
    const { detail } = body as { detail: unknown }
    if (typeof detail === 'string') {
      if (detail !== '') return detail
    } else if (detail != null) {
      return JSON.stringify(detail)
    }
  }
  const status_ = statusText ? `${status} ${statusText}` : String(status)
  return `HTTP ${status_} with no error detail. Check that the API is running on port 8000 and read its log.`
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  const { signal } = init
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch (error) {
    if (signal?.aborted) throw error // superseded by a newer run; the caller ignores it
    const reason = error instanceof Error ? error.message : String(error)
    throw new ApiError(0, `Could not reach the API (${reason}). Is it running on port 8000?`)
  }
  const text = await response.text()
  let parsed: unknown
  try {
    parsed = text === '' ? undefined : JSON.parse(text)
  } catch {
    parsed = undefined // not JSON, e.g. a plain-text 500 or a proxy error page
  }
  if (!response.ok) throw new ApiError(response.status, errorDetail(response.status, response.statusText, parsed))
  if (parsed === undefined) throw new ApiError(response.status, `The API sent a response to ${path} that is not JSON.`)
  return parsed as T
}

function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal })
}

function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  return request(path, { method: 'GET', signal })
}

export function postBacktest(request: BacktestRequest, signal?: AbortSignal): Promise<BacktestResponse> {
  return post('/api/backtest', request, signal)
}

export function postSensitivity(request: SensitivityRequest, signal?: AbortSignal): Promise<SensitivityResponse> {
  return post('/api/sensitivity', request, signal)
}

export function postMonteCarlo(request: MonteCarloRequest, signal?: AbortSignal): Promise<MonteCarloResponse> {
  return post('/api/monte-carlo', request, signal)
}

export function postWalkForward(request: WalkForwardRequest, signal?: AbortSignal): Promise<WalkForwardResponse> {
  return post('/api/walk-forward', request, signal)
}

export function getHealth(signal?: AbortSignal): Promise<{ status: string; engine_version: string }> {
  return get('/api/health', signal)
}

/** Stored backtests, newest first (§11.4). */
export function getRuns(limit = 50, signal?: AbortSignal): Promise<RunSummary[]> {
  return get(`/api/runs?limit=${limit}`, signal)
}

/** One stored backtest. Fields an older engine didn't store are null, hence StoredRun. */
export function getRun(id: number, signal?: AbortSignal): Promise<StoredRun> {
  return get(`/api/runs/${id}`, signal)
}
