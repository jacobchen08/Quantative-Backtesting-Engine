// The static demo (§13, "Static demo", Phase 9): a build with VITE_DEMO=1 replays responses that
// backend/scripts/record_demo.py recorded, so it runs on GitHub Pages with no backend. api.ts hands
// each request to replay() instead of fetch(): the request is looked up in public/demo/manifest.json,
// and the recorded response file it names is returned as the API's body.
import { ApiError } from './api'
import { buildRequests, buildWalkForward, defaultForm, type FormState } from './form'
import { resolvedToForm } from './history'
import type { BacktestRequest, MonteCarloRequest, WalkForwardRequest } from './types'

/** True in the demo build. A compile-time constant, so the live build drops the demo code. */
export const DEMO = import.meta.env.VITE_DEMO === '1'

export const NOT_RECORDED = "This configuration isn't in the demo. Run the project locally for live data."

export type Endpoint = 'backtest' | 'sensitivity' | 'monte_carlo' | 'walk_forward'

export type DemoScenario = {
  id: string
  label: string
  backtest: BacktestRequest
  sensitivity: BacktestRequest
  monte_carlo: MonteCarloRequest
  walk_forward: WalkForwardRequest
}

export type DemoManifest = {
  recorded_on: string // YYYY-MM-DD
  data_end: string // the last date in the recorded series
  engine_version: string
  scenarios: DemoScenario[]
  responses: Record<Endpoint, Record<string, string>> // request key -> file name in demo/
}

/** A value's JSON with every object's keys sorted, at every depth: the manifest's key (§13). */
export function canonicalKey(value: unknown): string {
  return JSON.stringify(value, (_key, v: unknown) => {
    if (v === null || typeof v !== 'object' || Array.isArray(v)) return v
    return Object.fromEntries(Object.entries(v).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))
  })
}

/**
 * The manifest's response table with its keys re-serialized here. The recorder writes them with
 * Python's json, which prints 2.0 where JavaScript prints 2; parsing and re-serializing each key
 * makes the lookup compare values, not two languages' number formatting.
 */
export function responseTable(manifest: DemoManifest): Record<Endpoint, Map<string, string>> {
  const table = {} as Record<Endpoint, Map<string, string>>
  for (const [endpoint, files] of Object.entries(manifest.responses) as [Endpoint, Record<string, string>][]) {
    table[endpoint] = new Map(Object.entries(files).map(([key, file]) => [canonicalKey(JSON.parse(key)), file]))
  }
  return table
}

/** Where the demo's files are served: under Vite's base, the repository's path on GitHub Pages. */
function demoUrl(file: string): string {
  return `${import.meta.env.BASE_URL}demo/${file}`
}

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  let response: Response
  try {
    response = await fetch(url, { signal })
  } catch (error) {
    if (signal?.aborted) throw error
    throw new ApiError(0, `Could not load the demo's recorded data (${error instanceof Error ? error.message : String(error)}).`)
  }
  if (!response.ok) throw new ApiError(response.status, `Could not load the demo's recorded data: ${url} returned HTTP ${response.status}.`)
  return (await response.json()) as T
}

let manifest: Promise<{ manifest: DemoManifest; table: Record<Endpoint, Map<string, string>> }> | null = null

/** The manifest, fetched once. */
export function loadManifest(): Promise<DemoManifest> {
  return loaded().then((m) => m.manifest)
}

function loaded() {
  manifest ??= fetchJson<DemoManifest>(demoUrl('manifest.json')).then((m) => ({ manifest: m, table: responseTable(m) }))
  manifest.catch(() => (manifest = null)) // a failed load is retried by the next call
  return manifest
}

/** The recorded response to a request, or an ApiError with NOT_RECORDED if there is none. */
export async function replay<T>(endpoint: Endpoint, request: unknown, signal?: AbortSignal): Promise<T> {
  const { table } = await loaded()
  const file = table[endpoint].get(canonicalKey(request))
  if (file === undefined) throw new ApiError(0, NOT_RECORDED)
  return fetchJson<T>(demoUrl(file), signal)
}

/** The sidebar of a recorded scenario: its backtest's settings, with its Monte Carlo and walk-forward. */
export function scenarioForm(scenario: DemoScenario): FormState {
  const bt = scenario.backtest
  const base = defaultForm(bt.end)
  const form = resolvedToForm(bt, base)
  const mc = scenario.monte_carlo
  const wf = scenario.walk_forward
  return sameText(base, {
    ...form,
    mcPaths: String(mc.n_paths),
    mcHorizon: String(mc.horizon),
    mcBlockAuto: mc.block_len === 'auto',
    mcBlockLen: mc.block_len === 'auto' ? form.mcBlockLen : String(mc.block_len),
    mcSeed: mc.seed == null ? '' : String(mc.seed),
    mcZeroDrift: mc.zero_drift ?? false,
    wfTrainDays: String(wf.train_days),
    wfTestDays: String(wf.test_days),
    wfAnchored: wf.anchored ?? false,
  })
}

/**
 * The form with the defaults' text wherever it holds the same number: a recorded 2.0 reads back
 * as "2", and the default is "2.0". Without this every scenario would differ from the defaults in
 * text only, and the URL would carry those fields (§13: it holds what differs).
 */
function sameText(base: FormState, form: FormState): FormState {
  const out = { ...form }
  for (const key of Object.keys(form) as (keyof FormState)[]) {
    const a = form[key]
    const b = base[key]
    if (typeof a === 'string' && typeof b === 'string' && a !== b && a.trim() !== '' && b.trim() !== '' && Number(a) === Number(b)) {
      ;(out as Record<string, unknown>)[key] = b
    }
  }
  return out
}

/** The recorded scenario a sidebar sends exactly, or null. */
export function matchScenario(form: FormState, scenarios: readonly DemoScenario[]): DemoScenario | null {
  const built = buildRequests(form)
  const wf = buildWalkForward(form)
  if (!built.ok || !wf.ok) return null
  const key = canonicalKey([built.backtest, built.monteCarlo, wf.request])
  return scenarios.find((s) => canonicalKey([s.backtest, s.monte_carlo, s.walk_forward]) === key) ?? null
}
