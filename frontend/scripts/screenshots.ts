// Regenerates docs/screenshots (§12): `npm run screenshots` in frontend/.
//
// Needs the API on :8000 and the Vite dev server on :5173, with real Yahoo data: it downloads what
// the price cache lacks, so it needs the network and never runs in CI. Each scenario is set up
// through the URL (§13, URL state), then run like a user would: Run, and Run walk-forward where shown.
//
//   npm run screenshots                     # writes ../docs/screenshots
//   npm run screenshots -- --out C:\tmp\shots --app http://localhost:5173
import { chromium, expect, type Locator, type Page } from '@playwright/test'
import { mkdirSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { parseArgs } from 'node:util'

const { values: args } = parseArgs({
  options: {
    out: { type: 'string', default: join(import.meta.dirname, '..', '..', 'docs', 'screenshots') },
    app: { type: 'string', default: 'http://localhost:5173' },
  },
})
const OUT = resolve(args.out)
const APP = args.app
const WIDTH = 1440
const HEIGHT = 900
const RUN_TIMEOUT = 240_000 // a first run on new tickers downloads them from Yahoo

mkdirSync(OUT, { recursive: true })

function panel(page: Page, title: string): Locator {
  return page.locator('main section').filter({ has: page.getByRole('heading', { name: title, exact: true }) })
}

async function open(page: Page, settings: Record<string, string>) {
  await page.setViewportSize({ width: WIDTH, height: HEIGHT })
  await page.goto(`${APP}/?${new URLSearchParams(settings)}`)
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeVisible()
}

/** Run, then wait for the backtest (a new run id, or an error) and for the grid and Monte Carlo. */
async function run(page: Page): Promise<number> {
  const started = Date.now()
  const header = page.locator('main p', { hasText: /run #\d+/ })
  const before = (await header.count()) > 0 ? await header.first().innerText() : null
  await page.getByRole('button', { name: 'Run', exact: true }).click()
  await expect(header.first().or(page.locator('main [role=alert]').first())).toBeVisible({ timeout: RUN_TIMEOUT })
  if (before !== null) await expect(header.first()).not.toHaveText(before, { timeout: RUN_TIMEOUT })
  await expect(page.locator('main .animate-pulse')).toHaveCount(0, { timeout: RUN_TIMEOUT })
  return Date.now() - started
}

async function walkForward(page: Page): Promise<number> {
  const started = Date.now()
  const wf = panel(page, 'Walk-forward')
  await wf.getByRole('button', { name: 'Run walk-forward' }).click()
  await expect(wf.getByText('Out-of-sample days').or(wf.locator('[role=alert]'))).toBeVisible({ timeout: RUN_TIMEOUT })
  return Date.now() - started
}

async function settle(page: Page) {
  await page.waitForTimeout(500) // Recharts lays out after its resize observer fires
}

async function shot(page: Page, name: string) {
  await settle(page)
  await page.screenshot({ path: join(OUT, `${name}.png`) })
  console.log('saved', `${name}.png`)
}

/** The main panel scrolls inside the window, so a full-page shot grows the viewport to its height. */
async function tall(page: Page) {
  const height = await page.locator('main').evaluate((m) => m.scrollHeight)
  await page.setViewportSize({ width: WIDTH, height: Math.max(HEIGHT, Math.ceil(height)) })
  await settle(page)
}

async function section(page: Page, name: string, title: string) {
  await settle(page)
  const box = await panel(page, title).boundingBox()
  if (box === null) throw new Error(`no panel "${title}"`)
  await page.screenshot({ path: join(OUT, `${name}.png`), clip: { x: box.x - 8, y: box.y - 8, width: box.width + 16, height: box.height + 16 } })
  console.log('saved', `${name}.png`)
}

async function report(page: Page, name: string) {
  for (const alert of await page.locator('main [role=alert]').allInnerTexts()) console.log(name, 'alert:', alert)
  for (const warning of await page.locator('main [role=status] li').allInnerTexts()) console.log(name, 'warning:', warning)
}

const segment = (page: Page, label: 'IS' | 'OOS' | 'Full') =>
  page.getByRole('group', { name: 'Segment' }).getByRole('button', { name: label, exact: true }).click()

const PAIRS = { start: '2015-01-02', end: '2025-12-31', mcSeed: '42' } // KO/PEP: the default pair
const DIVERSIFIED = 'SPY QQQ IWM EFA EEM TLT IEF GLD DBC VNQ'

const browser = await chromium.launch()
try {
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT } })

  // KO/PEP, the README's set. The sections show the IS segment (the cost sweep has a break-even line
  // there), and the walk-forward is shown without the benchmark line (approved in Phase 6).
  await open(page, PAIRS)
  console.log('pairs finished in', await run(page), 'ms')
  console.log('pairs walk-forward finished in', await walkForward(page), 'ms')
  await report(page, 'pairs')
  await page.locator('main').evaluate((m) => m.scrollTo(0, 0)) // the walk-forward button scrolled it down
  await shot(page, 'dashboard-pairs')
  await tall(page)
  await shot(page, 'dashboard-pairs-full')
  await segment(page, 'IS')
  for (const [name, title] of [
    ['kpi-grid', 'Performance'],
    ['equity-drawdown', 'Equity and drawdown'],
    ['rolling-statistics', 'Rolling statistics'],
    ['calendar-returns', 'Calendar returns'],
    ['signal-panel', 'Signal'],
    ['trades-table', 'Trades'],
    ['cost-sweep', 'Cost sweep'],
    ['sensitivity-heatmap', 'Parameter sensitivity'],
    ['monte-carlo', 'Monte Carlo'],
  ]) {
    await section(page, name, title)
  }
  await panel(page, 'Equity and drawdown').getByRole('button', { name: /Buy & hold SPY/ }).click()
  await section(page, 'walk-forward', 'Walk-forward')

  // The same pair with the Kalman model (§5.7, default delta).
  await open(page, { ...PAIRS, pairsModel: 'kalman' })
  console.log('pairs-kalman finished in', await run(page), 'ms')
  await report(page, 'pairs-kalman')
  await tall(page)
  await section(page, 'signal-panel-kalman', 'Signal')

  // SPY trend with a 10% volatility target, 2006-2025, and its equity on the log scale (Phase 8).
  await open(page, { strategy: 'trend', start: '2006-01-03', end: '2025-12-31', volOn: 'true', mcSeed: '42' })
  console.log('trend finished in', await run(page), 'ms')
  console.log('trend walk-forward finished in', await walkForward(page), 'ms')
  await report(page, 'trend')
  await tall(page)
  await shot(page, 'dashboard-trend-voltarget-full')
  await page.getByRole('group', { name: 'Equity scale' }).getByRole('button', { name: 'Log' }).click()
  await section(page, 'equity-drawdown-log', 'Equity and drawdown')
  await page.getByRole('group', { name: 'Equity scale' }).getByRole('button', { name: 'Linear' }).click()

  // The diversified trend portfolio (§5.8), with the automatic block length (§10.8) and the
  // stationary bootstrap shown in the Monte Carlo panel.
  await open(page, {
    strategy: 'trend_portfolio',
    portfolioTickers: DIVERSIFIED,
    start: '2015-01-02',
    end: '2025-12-31',
    mcBlockAuto: 'true',
    mcSeed: '42',
  })
  console.log('trend-portfolio finished in', await run(page), 'ms')
  console.log('trend-portfolio walk-forward finished in', await walkForward(page), 'ms')
  await report(page, 'trend-portfolio')
  await tall(page)
  await shot(page, 'dashboard-trend-portfolio-full')
  await section(page, 'signal-panel-trend-portfolio', 'Signal')
  await section(page, 'kpi-grid-trend-portfolio', 'Performance')
  await panel(page, 'Monte Carlo').getByRole('button', { name: 'Stationary' }).click()
  await section(page, 'monte-carlo-stationary', 'Monte Carlo')

  // Run history (Phase 8): the four runs above, compared on the full window.
  await open(page, {})
  const history = panel(page, 'Run history')
  await history.getByRole('button', { name: 'Show run history' }).click()
  const rows = history.locator('table[aria-label="Stored runs"] tbody tr')
  await expect(rows.first()).toBeVisible()
  for (let i = 3; i >= 0; i -= 1) await rows.nth(i).getByRole('checkbox').check()
  await expect(history.getByRole('table', { name: 'Run comparison' })).toContainText('Borrow drag')
  await tall(page)
  await section(page, 'run-history', 'Run history')

  // Error states: an unknown ticker (404), and a trend history too short for the grid (422).
  await open(page, { strategy: 'mean_reversion', ticker: 'ZZZZQQ', ...PAIRS })
  await run(page)
  await shot(page, 'error-unknown-ticker')
  await open(page, { strategy: 'trend', start: '2023-01-03', end: '2025-12-31', mcSeed: '42' })
  await run(page)
  await report(page, 'short-trend')
  await tall(page)
  await section(page, 'error-sensitivity-grid', 'Parameter sensitivity')
} finally {
  await browser.close()
}
