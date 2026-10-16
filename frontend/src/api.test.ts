import { afterEach, describe, expect, it, vi } from 'vitest'
import { API_BASE, ApiError, errorDetail, postBacktest, postMonteCarlo, postSensitivity } from './api'
import type { BacktestRequest } from './types'

const REQUEST: BacktestRequest = { strategy: 'pairs', tickers: ['KO', 'PEP'], start: '2015-01-02', end: '2025-12-31' }

function stubFetch(response: Response | Error) {
  const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => {
    if (response instanceof Error) throw response
    return response
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

const json = (body: unknown, status = 200, statusText = '') =>
  new Response(JSON.stringify(body), { status, statusText, headers: { 'Content-Type': 'application/json' } })

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('errorDetail: every API error is {"detail": "..."} (§11.1)', () => {
  it('returns the detail string verbatim', () => {
    expect(errorDetail(422, 'Unprocessable Entity', { detail: 'L must be between 10 and 252 for pairs, got 5' })).toBe(
      'L must be between 10 and 252 for pairs, got 5',
    )
  })

  it('serializes a detail that is not a string', () => {
    expect(errorDetail(422, '', { detail: [{ msg: 'bad' }] })).toBe('[{"msg":"bad"}]')
  })

  it('falls back to the HTTP status when there is no usable detail', () => {
    expect(errorDetail(500, 'Internal Server Error', undefined)).toMatch(/^HTTP 500 Internal Server Error with no error detail/)
    expect(errorDetail(502, '', { detail: '' })).toMatch(/^HTTP 502 with no error detail/)
    expect(errorDetail(504, '', 'gateway timeout')).toMatch(/^HTTP 504 with no error detail/)
  })
})

describe('typed fetch wrappers', () => {
  it('POSTs the request as JSON and returns the parsed body', async () => {
    const fetchMock = stubFetch(json({ run_id: 7 }))
    const result = await postBacktest(REQUEST)
    expect(result).toEqual({ run_id: 7 })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe(`${API_BASE}/api/backtest`)
    expect(init?.method).toBe('POST')
    expect(init?.headers).toEqual({ 'Content-Type': 'application/json' })
    expect(JSON.parse(init?.body as string)).toEqual(REQUEST)
  })

  it('uses one path per endpoint', async () => {
    const sensitivity = stubFetch(json({}))
    await postSensitivity(REQUEST)
    expect(sensitivity.mock.calls[0][0]).toBe(`${API_BASE}/api/sensitivity`)
    const monteCarlo = stubFetch(json({}))
    await postMonteCarlo({ source: 'strategy', backtest: REQUEST })
    expect(monteCarlo.mock.calls[0][0]).toBe(`${API_BASE}/api/monte-carlo`)
  })

  it('turns an error response into an ApiError carrying the status and detail', async () => {
    stubFetch(json({ detail: 'No data for ZZZZ between 2015-01-02 and 2025-12-31' }, 404, 'Not Found'))
    const error = await postBacktest(REQUEST).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 404, message: 'No data for ZZZZ between 2015-01-02 and 2025-12-31' })
  })

  it('handles an error body that is not JSON', async () => {
    stubFetch(new Response('Internal Server Error', { status: 500, statusText: 'Internal Server Error' }))
    const error = await postBacktest(REQUEST).catch((e: unknown) => e)
    expect(error).toMatchObject({ status: 500, message: expect.stringMatching(/^HTTP 500 Internal Server Error/) })
  })

  it('rejects a successful response that is not JSON', async () => {
    stubFetch(new Response('<html>', { status: 200 }))
    await expect(postBacktest(REQUEST)).rejects.toMatchObject({ status: 200, message: expect.stringMatching(/not JSON/) })
  })

  it('reports an unreachable API with status 0', async () => {
    stubFetch(new TypeError('Failed to fetch'))
    const error = await postBacktest(REQUEST).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 0, message: expect.stringMatching(/Could not reach the API \(Failed to fetch\)/) })
  })

  it('rethrows an abort untouched, so a superseded run can be ignored', async () => {
    const controller = new AbortController()
    controller.abort()
    const abort = new DOMException('The operation was aborted.', 'AbortError')
    stubFetch(abort)
    await expect(postBacktest(REQUEST, controller.signal)).rejects.toBe(abort)
  })
})
