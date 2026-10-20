// Browser tests of the static demo (§13, "Static demo"; Phase 9): the demo build, served as static
// files under its GitHub Pages path by `vite preview` on :4174 (playwright.config.ts), replaying the
// responses recorded in public/demo/. No API is involved: the tests check that none is called.
import { expect, test, type Page } from '@playwright/test'
import manifest from '../public/demo/manifest.json' with { type: 'json' }

const DEMO = 'http://localhost:4174/Quantative-Backtesting-Engine/' // vite.config.ts, DEMO_BASE
const NOT_RECORDED = "This configuration isn't in the demo. Run the project locally for live data."

const HEADINGS: Record<string, string> = {
  'pairs-ko-pep': 'Pairs · KO / PEP',
  'pairs-xom-cvx': 'Pairs · XOM / CVX',
  'pairs-googl-goog': 'Pairs · GOOGL / GOOG',
  'pairs-nvda-amd': 'Pairs · NVDA / AMD',
  'mean-reversion-ko': 'Mean reversion · KO',
  'trend-spy': 'Trend · SPY',
  'trend-spy-vol': 'Trend · SPY',
  'trend-portfolio-etfs': 'Trend portfolio · SPY, QQQ, GLD, TLT',
}

/** Console errors, and every request that isn't for the demo's own static files. */
function watch(page: Page) {
  const errors: string[] = []
  const foreign: string[] = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push(e.message))
  page.on('request', (r) => {
    if (!r.url().startsWith(DEMO)) foreign.push(r.url())
  })
  return { errors, foreign }
}

/** Wait until the backtest, the sensitivity grid and the Monte Carlo have been replayed. */
async function settled(page: Page) {
  await expect(page.locator('main h2').first().or(page.locator('main [role=alert]').first())).toBeVisible()
  await expect(page.locator('main .animate-pulse')).toHaveCount(0)
}

function panel(page: Page, title: string) {
  return page.locator('main section').filter({ has: page.getByRole('heading', { name: title, exact: true }) })
}

test('the demo replays its first scenario on load, with the banner and no API', async ({ page }) => {
  const { errors, foreign } = watch(page)
  await page.goto(DEMO)
  await settled(page)
  await expect(page.getByRole('heading', { name: 'Pairs · KO / PEP' })).toBeVisible()
  await expect(page.getByRole('status').filter({ hasText: 'Demo:' })).toHaveText(
    new RegExp(
      `^Demo: results recorded on ${manifest.recorded_on} from Yahoo data through ${manifest.data_end}\\. ` +
        'Run locally for live data\\.',
    ),
  )
  await expect(page.locator('main p', { hasText: /· recorded · engine/ })).toBeVisible()
  for (const title of ['Performance', 'Equity and drawdown', 'Trades', 'Parameter sensitivity', 'Monte Carlo']) {
    await expect(panel(page, title), title).toBeVisible()
  }
  await expect(panel(page, 'Parameter sensitivity').locator('[title^="L = "]')).toHaveCount(20)
  const seed = panel(page, 'Monte Carlo').locator('dt', { hasText: 'Seed' }).locator('xpath=following-sibling::dd[1]')
  await expect(seed).toContainText('42')
  await expect(panel(page, 'Run history')).toHaveCount(0) // no database in the demo (§13)

  // Only the scenario picker and Run can be used in the sidebar.
  await expect(page.getByLabel('Demo scenario')).toBeEnabled()
  await expect(page.getByRole('group', { name: 'Strategy' }).getByRole('button', { name: 'Trend' })).toBeDisabled()
  await expect(page.locator('form fieldset input:enabled, form fieldset select:enabled')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled()

  expect(foreign).toEqual([])
  expect(errors).toEqual([])
})

test('every scenario replays, with its walk-forward', async ({ page }) => {
  const { errors, foreign } = watch(page)
  await page.goto(DEMO)
  await settled(page)
  for (const scenario of manifest.scenarios) {
    await page.getByLabel('Demo scenario').selectOption(scenario.id)
    await settled(page)
    await expect(page.getByRole('heading', { name: HEADINGS[scenario.id], exact: true })).toBeVisible()
    await expect(page.locator('main [role=alert]')).toHaveCount(0)
    const wf = panel(page, 'Walk-forward')
    await wf.getByRole('button', { name: 'Run walk-forward' }).click()
    await expect(wf).toContainText('Out-of-sample days')
    await expect(wf.locator('[role=alert]')).toHaveCount(0)
  }
  expect(foreign).toEqual([])
  expect(errors).toEqual([])
})

test("a scenario's URL reopens it", async ({ page }) => {
  await page.goto(DEMO)
  await settled(page)
  await page.getByLabel('Demo scenario').selectOption('trend-spy-vol')
  await settled(page)
  expect(page.url()).toContain('volOn=true')
  await page.reload()
  await settled(page)
  await expect(page.getByLabel('Demo scenario')).toHaveValue('trend-spy-vol')
  await expect(page.getByRole('heading', { name: 'Trend · SPY', exact: true })).toBeVisible()
  await expect(page.locator('footer')).toContainText('annual target') // the vol target is on
})

test('a configuration that was not recorded says so', async ({ page }) => {
  const { foreign } = watch(page)
  await page.goto(`${DEMO}?zEntry=1.5`)
  await expect(page.locator('main [role=alert]')).toContainText(NOT_RECORDED)
  await expect(page.getByLabel('Demo scenario')).toHaveValue('')
  await expect(page.getByLabel('Demo scenario').locator('option:checked')).toHaveText('Not a recorded scenario')
  await page.getByLabel('Demo scenario').selectOption('pairs-ko-pep')
  await settled(page)
  await expect(page.getByRole('heading', { name: 'Pairs · KO / PEP' })).toBeVisible()
  expect(foreign).toEqual([])
})
