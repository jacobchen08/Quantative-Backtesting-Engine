import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from './api'
import {
  canonicalKey,
  loadManifest,
  matchScenario,
  NOT_RECORDED,
  replay,
  responseTable,
  scenarioForm,
  type DemoManifest,
  type Endpoint,
} from './demo'
import { buildRequests, buildWalkForward } from './form'
import { formToSearch, searchToForm } from './urlState'

// The recorded demo (backend/scripts/record_demo.py writes it; it is committed with the frontend),
// every file by its name, as text.
const FILES: Record<string, string> = Object.fromEntries(
  Object.entries(import.meta.glob('../public/demo/*.json', { eager: true, query: '?raw', import: 'default' })).map(
    ([path, text]) => [path.slice(path.lastIndexOf('/') + 1), text as string],
  ),
)
const manifest = JSON.parse(FILES['manifest.json']) as DemoManifest
const ENDPOINTS: Endpoint[] = ['backtest', 'sensitivity', 'monte_carlo', 'walk_forward']

describe('canonicalKey', () => {
  it('sorts keys at every depth and keeps array order', () => {
    const a = canonicalKey({ b: 1, a: { d: [3, 1], c: null } })
    expect(a).toBe('{"a":{"c":null,"d":[3,1]},"b":1}')
    expect(canonicalKey({ a: { c: null, d: [3, 1] }, b: 1 })).toBe(a)
  })

  it('tells different values apart', () => {
    expect(canonicalKey({ x: [1, 2] })).not.toBe(canonicalKey({ x: [2, 1] }))
    expect(canonicalKey({ x: 1 })).not.toBe(canonicalKey({ x: '1' }))
  })
})

describe('responseTable', () => {
  it("matches a Python key's 2.0 to JavaScript's 2", () => {
    const table = responseTable({
      ...manifest,
      responses: { backtest: { '{"tickers": ["KO"], "z_entry": 2.0}': 'f.json' }, sensitivity: {}, monte_carlo: {}, walk_forward: {} },
    })
    expect(table.backtest.get(canonicalKey({ z_entry: 2, tickers: ['KO'] }))).toBe('f.json')
  })
})

describe('the recorded demo', () => {
  it('has the scenarios of §13', () => {
    expect(manifest.scenarios.map((s) => s.id)).toEqual([
      'pairs-ko-pep',
      'pairs-xom-cvx',
      'pairs-googl-goog',
      'pairs-nvda-amd',
      'mean-reversion-ko',
      'trend-spy',
      'trend-spy-vol',
      'trend-portfolio-etfs',
    ])
    expect(manifest.scenarios.every((s) => s.monte_carlo.seed === 42)).toBe(true)
    expect(manifest.recorded_on).toMatch(/^\d{4}-\d{2}-\d{2}$/)
    expect(manifest.data_end).toMatch(/^\d{4}-\d{2}-\d{2}$/)
  })

  // The demo can only replay what the sidebar sends, byte for byte after canonicalization.
  it.each(manifest.scenarios.map((s) => [s.id, s] as const))('%s: the sidebar sends exactly the recorded requests', (_id, scenario) => {
    const form = scenarioForm(scenario)
    const built = buildRequests(form)
    const wf = buildWalkForward(form)
    if (!built.ok || !wf.ok) throw new Error('the scenario form does not build')
    expect(built.backtest).toEqual(scenario.backtest)
    expect(built.sensitivity).toEqual(scenario.sensitivity)
    expect(built.monteCarlo).toEqual(scenario.monte_carlo)
    expect(wf.request).toEqual(scenario.walk_forward)
    const table = responseTable(manifest)
    expect(table.backtest.has(canonicalKey(built.backtest))).toBe(true)
    expect(table.sensitivity.has(canonicalKey(built.sensitivity))).toBe(true)
    expect(table.monte_carlo.has(canonicalKey(built.monteCarlo))).toBe(true)
    expect(table.walk_forward.has(canonicalKey(wf.request))).toBe(true)
    expect(matchScenario(form, manifest.scenarios)).toBe(scenario)
  })

  it('names files that exist, and no file goes unnamed', () => {
    const named = ENDPOINTS.flatMap((e) => Object.values(manifest.responses[e]))
    expect(new Set(named).size).toBe(named.length)
    expect(named).toHaveLength(manifest.scenarios.length * ENDPOINTS.length)
    const onDisk = Object.keys(FILES).filter((f) => f !== 'manifest.json')
    expect([...named].sort()).toEqual(onDisk.sort())
  })

  it('stays under 10 MB (§13)', () => {
    const encoder = new TextEncoder()
    const total = Object.values(FILES).reduce((sum, text) => sum + encoder.encode(text).length, 0)
    expect(total).toBeLessThan(10_000_000)
  })

  it("puts the first scenario at the bare URL, and every other one's URL round-trips", () => {
    const defaults = scenarioForm(manifest.scenarios[0])
    expect(formToSearch(defaults, defaults)).toBe('')
    for (const scenario of manifest.scenarios) {
      const form = scenarioForm(scenario)
      const search = formToSearch(form, defaults)
      expect(search).not.toMatch(/zEntry|zExit|mcSeed=&/) // equal numbers keep the defaults' text
      expect(searchToForm(search, defaults)).toEqual({ form, notices: [] })
    }
  })

  it('matches no scenario once a setting changes', () => {
    const form = scenarioForm(manifest.scenarios[0])
    expect(matchScenario({ ...form, zEntry: '1.5' }, manifest.scenarios)).toBeNull()
    expect(matchScenario({ ...form, mcSeed: '7' }, manifest.scenarios)).toBeNull()
    expect(matchScenario({ ...form, wfAnchored: true }, manifest.scenarios)).toBeNull()
    expect(matchScenario({ ...form, zEntry: '' }, manifest.scenarios)).toBeNull()
  })
})

describe('replay', () => {
  afterEach(() => vi.unstubAllGlobals())

  /** A fetch that serves public/demo as the static host would, under Vite's base "/" (tests). */
  function serveDemo() {
    const fetch = vi.fn(async (url: string) => {
      const text = FILES[url.replace(/^\/demo\//, '')]
      return text === undefined ? new Response('not found', { status: 404 }) : new Response(text, { status: 200 })
    })
    vi.stubGlobal('fetch', fetch)
    return fetch
  }

  // First in this block: the manifest is fetched once per page, so later tests find it loaded.
  it('retries the manifest after a failed load', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 503 })))
    await expect(loadManifest()).rejects.toThrow('HTTP 503')
    serveDemo()
    await expect(loadManifest()).resolves.toMatchObject({ data_end: manifest.data_end })
  })

  it('returns the recorded response to a recorded request', async () => {
    const fetch = serveDemo()
    const scenario = manifest.scenarios[0]
    const body = await replay<{ resolved: { tickers: string[] } }>('backtest', scenario.backtest)
    expect(body.resolved.tickers).toEqual(scenario.backtest.tickers)
    expect(fetch).toHaveBeenLastCalledWith(`/demo/${scenario.id}-backtest.json`, expect.anything())
  })

  it('looks a request up under its own endpoint', async () => {
    serveDemo()
    const scenario = manifest.scenarios[0]
    // The sensitivity request is the backtest's body; its endpoint picks the grid's file.
    const grid = await replay<{ sharpe: unknown[]; n_configs_tested: number }>('sensitivity', scenario.sensitivity)
    expect(grid.sharpe.length).toBeGreaterThan(0)
    expect(grid).not.toHaveProperty('series')
    // The walk-forward request isn't a backtest request.
    await expect(replay('backtest', scenario.walk_forward)).rejects.toThrow(NOT_RECORDED)
  })

  it('rejects a request that was not recorded, with no HTTP status', async () => {
    const fetch = serveDemo()
    const calls = fetch.mock.calls.length
    const request = { ...manifest.scenarios[0].backtest, z_entry: 1.5 }
    const error = await replay('backtest', request).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).message).toBe(NOT_RECORDED)
    expect((error as ApiError).status).toBe(0)
    expect(fetch.mock.calls.length).toBe(calls) // nothing is fetched for it
  })

  it('reports a recorded file that is missing from the host', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })))
    await expect(replay('backtest', manifest.scenarios[0].backtest)).rejects.toThrow(/returned HTTP 404/)
  })
})
