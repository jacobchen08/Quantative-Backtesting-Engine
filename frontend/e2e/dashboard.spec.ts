// Browser tests (§12): the built dashboard against the offline API on the seeded synthetic market.
// SYNY/SYNX are a cointegrated pair and SYNT a trending asset, all on 2012-01-03 to 2019-12-31;
// anything else is not in the cache, so the API answers 503 instead of downloading (§4.3).
import { expect, test, type Locator, type Page, type Request } from '@playwright/test'

const SYN = { start: '2012-01-03', end: '2019-12-31' }
const FAST_MC = { mcPaths: '2000', mcSeed: '42' } // a smaller Monte Carlo; the seed makes it repeatable

/** A dashboard URL: settings go in through the query string (§13, URL state). */
function url(settings: Record<string, string>): string {
  return `/?${new URLSearchParams(settings)}`
}

/** The sidebar control under a field label (the label's first line). */
function field(page: Page, label: string): Locator {
  return page
    .locator('form label')
    .filter({ has: page.locator('xpath=./span[1]', { hasText: new RegExp(`^${label.replace(/[()[\]]/g, '\\$&')}$`) }) })
    .locator('input, select')
    .first()
}

/** A main-panel section by its title. */
function panel(page: Page, title: string): Locator {
  return page.locator('main section').filter({ has: page.getByRole('heading', { name: title, exact: true }) })
}

/** Every request the page sends to the API. */
function apiRequests(page: Page): Request[] {
  const seen: Request[] = []
  page.on('request', (r) => {
    if (new URL(r.url()).pathname.startsWith('/api/')) seen.push(r)
  })
  return seen
}

/** Press Run and wait until the backtest, the sensitivity grid and the Monte Carlo have all settled. */
async function run(page: Page) {
  const runIds = page.locator('main p', { hasText: /run #\d+/ })
  const before = (await runIds.count()) > 0 ? await runIds.first().innerText() : null
  await page.getByRole('button', { name: 'Run', exact: true }).click()
  // The run header ("… · run #N · …") is the backtest's response; a new run has a new id.
  await expect(runIds.first().or(page.locator('main [role=alert]').first())).toBeVisible({ timeout: 60_000 })
  if (before !== null) await expect(runIds.first()).not.toHaveText(before, { timeout: 60_000 })
  await expect(page.locator('main .animate-pulse')).toHaveCount(0, { timeout: 60_000 })
}

test('a pairs run renders every panel, with no console errors', async ({ page }) => {
  const errors: string[] = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto(url({ pairY: 'SYNY', pairX: 'SYNX', ...SYN, ...FAST_MC }))
  await run(page)
  await expect(page.getByRole('heading', { name: /Pairs · SYNY \/ SYNX/ })).toBeVisible()

  for (const title of [
    'Performance',
    'Equity and drawdown',
    'Rolling statistics',
    'Calendar returns',
    'Signal',
    'Trades',
    'Cost sweep',
    'Parameter sensitivity',
    'Walk-forward',
    'Monte Carlo',
    'Run history',
  ]) {
    await expect(panel(page, title), title).toBeVisible()
  }
  await expect(panel(page, 'Equity and drawdown').locator('svg.recharts-surface')).toHaveCount(2)
  await expect(panel(page, 'Trades').locator('tbody tr').first()).toBeVisible()
  await expect(panel(page, 'Parameter sensitivity').locator('[title^="L = "]')).toHaveCount(20) // 5 L × 4 z_entry
  const seed = panel(page, 'Monte Carlo').locator('dt', { hasText: 'Seed' }).locator('xpath=following-sibling::dd[1]')
  await expect(seed).toContainText('42')

  await panel(page, 'Walk-forward').getByRole('button', { name: 'Run walk-forward' }).click()
  await expect(panel(page, 'Walk-forward')).toContainText('Out-of-sample days', { timeout: 60_000 })

  await page.getByRole('group', { name: 'Equity scale' }).getByRole('button', { name: 'Log' }).click()
  await expect(panel(page, 'Equity and drawdown').locator('.recharts-yAxis-tick-labels').first()).toContainText('$100K')

  await expect(page.locator('main [role=alert]')).toHaveCount(0)
  expect(errors).toEqual([])
})

test('a trend run shows the single-row heatmap', async ({ page }) => {
  await page.goto(url({ strategy: 'trend', ticker: 'SYNT', ...SYN, ...FAST_MC }))
  await run(page)
  const cells = panel(page, 'Parameter sensitivity').locator('[title^="L = "]')
  await expect(cells).toHaveCount(5) // L = 21, 63, 126, 252, 378; no z_entry axis
  const tops = await cells.evaluateAll((els) => els.map((e) => Math.round(e.getBoundingClientRect().top)))
  expect(new Set(tops).size).toBe(1)
  for (const title of await cells.evaluateAll((els) => els.map((e) => e.getAttribute('title') ?? ''))) {
    expect(title).not.toContain('z entry')
  }
})

test("an uncached ticker shows the API's 503 offline detail", async ({ page }) => {
  await page.goto(url({ pairY: 'KO', pairX: 'SYNX', ...SYN }))
  await page.getByRole('button', { name: 'Run', exact: true }).click()
  const alert = page.locator('main [role=alert]')
  await expect(alert).toContainText('Backtest failed')
  await expect(alert).toContainText('HTTP 503')
  await expect(alert).toContainText('Offline mode: KO is not in the local cache')
})

test('a too-short range shows the sensitivity 422 while the backtest renders', async ({ page }) => {
  await page.goto(url({ strategy: 'trend', ticker: 'SYNT', start: '2017-01-03', end: SYN.end, ...FAST_MC }))
  await run(page)
  await expect(page.getByRole('heading', { name: /Trend · SYNT/ })).toBeVisible()
  await expect(panel(page, 'Performance')).toBeVisible()
  const grid = panel(page, 'Parameter sensitivity').locator('[role=alert]')
  await expect(grid).toContainText('Sensitivity grid failed')
  await expect(grid).toContainText('HTTP 422')
  await expect(grid).toContainText('L = 378')
})

test('a blank field blocks the Run with zero requests', async ({ page }) => {
  await page.goto(url({ pairY: 'SYNY', pairX: 'SYNX', ...SYN }))
  const requests = apiRequests(page)
  await field(page, 'z entry').fill('')
  await page.getByRole('button', { name: 'Run', exact: true }).click()
  await expect(page.locator('form [role=alert]')).toHaveText('z entry is required.')
  await expect(page.getByText('No run yet.')).toBeVisible()
  await page.waitForTimeout(500) // nothing in flight either
  expect(requests).toEqual([])
})

test('after a run, the history panel lists it and "Load settings" restores the sidebar', async ({ page }) => {
  await page.goto(url({ pairY: 'SYNY', pairX: 'SYNX', ...SYN, zEntry: '2.5', costBps: '5', ...FAST_MC }))
  await run(page)
  const runId = (await page.locator('main p', { hasText: 'run #' }).first().innerText()).match(/run #(\d+)/)?.[1]
  expect(runId).toBeDefined()

  // Change the sidebar, then load the run back.
  await page.getByRole('group', { name: 'Strategy' }).getByRole('button', { name: 'Trend', exact: true }).click()
  await field(page, 'Cost (bps, one-way)').fill('20')

  const history = panel(page, 'Run history')
  await history.getByRole('button', { name: 'Show run history' }).click()
  const row = history.locator(`tr[data-run-id="${runId}"]`)
  await expect(row).toContainText('SYNY SYNX')
  await expect(row).toContainText(`${SYN.start} → ${SYN.end}`)
  await row.getByRole('button', { name: 'Load settings' }).click()

  await expect(page.getByRole('note')).toContainText(`Loaded the settings of run #${runId}`)
  await expect(page.getByRole('group', { name: 'Strategy' }).getByRole('button', { name: 'Pairs' })).toHaveAttribute('aria-pressed', 'true')
  await expect(field(page, 'Y (tickers[0])')).toHaveValue('SYNY')
  await expect(field(page, 'X (tickers[1])')).toHaveValue('SYNX')
  await expect(field(page, 'z entry')).toHaveValue('2.5')
  await expect(field(page, 'Cost (bps, one-way)')).toHaveValue('5')
  await expect(field(page, 'Start')).toHaveValue(SYN.start)
  await expect(field(page, 'End')).toHaveValue(SYN.end)

  // Selecting it compares its stored metrics.
  await row.getByRole('checkbox').check()
  await expect(history.getByRole('table', { name: 'Run comparison' })).toContainText(`Run #${runId}`)
})

test('the trades CSV downloads with the right header row', async ({ page }) => {
  await page.goto(url({ pairY: 'SYNY', pairX: 'SYNX', ...SYN, ...FAST_MC }))
  await run(page)
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('group', { name: 'Export CSV' }).getByRole('button', { name: 'Trades' }).click(),
  ])
  expect(download.suggestedFilename()).toBe(`pairs-SYNY_SYNX-${SYN.start}-${SYN.end}-trades.csv`)
  const stream = await download.createReadStream()
  const chunks: Buffer[] = []
  for await (const chunk of stream) chunks.push(chunk as Buffer)
  const text = Buffer.concat(chunks).toString('utf8')
  const lines = text.split('\r\n')
  expect(lines[0]).toBe('id,ticker,side,segment,open,signal_date,entry_date,exit_signal_date,exit_date,holding_days,pnl,beta_at_entry')
  const trades = await panel(page, 'Trades').locator('tbody tr').count()
  expect(lines.length - 2).toBe(trades) // a header, one line per trade, and the final CRLF
  expect(lines[1]).toMatch(/^1,,(long|short),is,/)
})

test('the URL round-trips the settings without sending a request on load', async ({ page }) => {
  const settings = {
    strategy: 'trend',
    ticker: 'SYNT',
    ...SYN,
    L: '126',
    signalFrequency: 'daily',
    costBps: '2.5',
    volOn: 'true',
    volTargetPct: '12',
    mcPaths: '2000',
    mcSeed: '7',
  }
  const requests = apiRequests(page)
  await page.goto(url(settings))
  await expect(page.getByRole('group', { name: 'Strategy' }).getByRole('button', { name: 'Trend', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  )
  await expect(field(page, 'Ticker')).toHaveValue('SYNT')
  await expect(field(page, 'Start')).toHaveValue(SYN.start)
  await expect(field(page, 'Cost (bps, one-way)')).toHaveValue('2.5')
  await expect(field(page, 'Seed')).toHaveValue('7')
  await expect(page.getByRole('note')).toHaveCount(0) // every parameter was valid
  await page.waitForTimeout(500)
  expect(requests).toEqual([])

  await run(page)
  await expect(page.getByRole('heading', { name: /Trend · SYNT/ })).toBeVisible()
  expect(requests.map((r) => new URL(r.url()).pathname)).toContain('/api/backtest')
  // Run writes back exactly the settings that differ from the defaults, in the form's order.
  expect(new URL(page.url()).search).toBe(new URL(url(settings), 'http://x').search)

  const before = requests.length
  await page.reload()
  await expect(field(page, 'Ticker')).toHaveValue('SYNT')
  await expect(page.getByText('No run yet.')).toBeVisible() // the URL holds settings, never results
  await page.waitForTimeout(500)
  expect(requests.length).toBe(before)
})

test('an invalid URL parameter is ignored with a notice naming it', async ({ page }) => {
  const requests = apiRequests(page)
  await page.goto('/?strategy=trend&L=abc&colour=red')
  const note = page.getByRole('note')
  await expect(note).toContainText('Ignored URL parameter L="abc": not a number.')
  await expect(note).toContainText('Ignored unknown URL parameter "colour".')
  await expect(page.getByRole('group', { name: 'Strategy' }).getByRole('button', { name: 'Trend', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  )
  await expect(field(page, 'Lookback L (trading days)')).toHaveValue('252') // trend's default, not "abc"
  expect(requests).toEqual([])
})

test('Run is not blocked by the browser: off-step values reach the API, and out-of-range ones get its 422', async ({ page }) => {
  // costBps 2.5 is off the input's step of 1, and borrowBps 10 off its step of 25; both are valid.
  await page.goto(url({ pairY: 'SYNY', pairX: 'SYNX', ...SYN, costBps: '2.5', ...FAST_MC }))
  await field(page, 'Borrow fee (bps a year)').fill('10')
  await run(page)
  await expect(page.getByRole('heading', { name: /Pairs · SYNY \/ SYNX/ })).toBeVisible()
  await expect(page.locator('main footer')).toContainText('2.5 bps')

  await field(page, 'Lookback L (trading days)').fill('600') // above the input's max of 252
  await page.getByRole('button', { name: 'Run', exact: true }).click()
  const alert = page.locator('main [role=alert]')
  await expect(alert).toContainText('HTTP 422')
  await expect(alert).toContainText('L')
})
