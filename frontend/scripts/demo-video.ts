// Records the README's demo video: `npm run demo-video` in frontend/.
//
// Like scripts/screenshots.ts, it needs the API on :8000 and the Vite dev server on :5173 with real
// Yahoo data, so it never runs in CI. It drives the dashboard the way a user would: a KO/PEP pairs
// run, a tour of every panel, a walk-forward, then the diversified trend portfolio. Playwright records
// the window as WebM; a drawn cursor shows where each click lands, because the recording has none.
// It then makes a smaller animated GIF of the same recording, which GitHub plays inline in the README
// (it doesn't play a video file from the repository), using Playwright's own ffmpeg to read the frames.
//
//   npm run demo-video                         # writes ../docs/media/demo.webm and demo.gif
//   npm run demo-video -- --gif-only           # remakes demo.gif from the existing demo.webm
import { chromium, expect, type Locator, type Page } from '@playwright/test'
import { spawnSync } from 'node:child_process'
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { buildPalette, GifWriter, readPng } from './gif.ts'

const { values: args } = parseArgs({
  options: {
    out: { type: 'string', default: join(import.meta.dirname, '..', '..', 'docs', 'media') },
    app: { type: 'string', default: 'http://localhost:5173' },
    'gif-only': { type: 'boolean', default: false },
  },
})
const OUT = resolve(args.out)
const WEBM = join(OUT, 'demo.webm')
const GIF = join(OUT, 'demo.gif')
const APP = args.app
const GIF_WIDTH = 800
const GIF_FPS = 6
const GIF_HOLD = 3 // seconds on the last frame before it loops
const WIDTH = 1440
const HEIGHT = 900
const RUN_TIMEOUT = 240_000 // a first run on new tickers downloads them from Yahoo

/** A cursor drawn into the page, with a ring on each click; the recording shows no system cursor. */
const CURSOR = `
  window.addEventListener('DOMContentLoaded', () => {
    const dot = document.createElement('div')
    dot.style.cssText = 'position:fixed;left:0;top:0;width:18px;height:18px;margin:-9px 0 0 -9px;border-radius:50%;' +
      'background:rgba(255,255,255,.85);border:2px solid #10b981;box-shadow:0 0 6px rgba(0,0,0,.6);' +
      'pointer-events:none;z-index:2147483647;transition:transform .12s'
    document.body.appendChild(dot)
    document.addEventListener('mousemove', (e) => { dot.style.left = e.clientX + 'px'; dot.style.top = e.clientY + 'px' }, true)
    document.addEventListener('mousedown', () => { dot.style.transform = 'scale(.7)' }, true)
    document.addEventListener('mouseup', () => { dot.style.transform = '' }, true)
  })`

function panel(page: Page, title: string): Locator {
  return page.locator('main section').filter({ has: page.getByRole('heading', { name: title, exact: true }) })
}

/** Glide the cursor to an element, pause so the viewer sees where it is going, and click it. */
async function press(page: Page, target: Locator, pause = 700) {
  await target.scrollIntoViewIfNeeded()
  const box = await target.boundingBox()
  if (box === null) throw new Error('nothing to click')
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 25 })
  await page.waitForTimeout(250)
  await page.mouse.down()
  await page.mouse.up()
  await target.click({ trial: true }) // fails loudly if the mouse click landed on something else
  await page.waitForTimeout(pause)
}

/** Scroll the main panel smoothly so a panel's header sits near the top, then hold. */
async function scrollTo(page: Page, title: string, hold = 2500) {
  const top = await panel(page, title).evaluate((s) => {
    const main = s.closest('main') as HTMLElement
    return main.scrollTop + s.getBoundingClientRect().top - main.getBoundingClientRect().top - 12
  })
  await page.locator('main').evaluate(
    (main, to) =>
      new Promise<void>((done) => {
        const from = main.scrollTop
        const start = performance.now()
        const ms = Math.min(1400, 500 + Math.abs(to - from) / 3)
        const step = (now: number) => {
          const t = Math.min(1, (now - start) / ms)
          main.scrollTop = from + (to - from) * (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2)
          if (t < 1) requestAnimationFrame(step)
          else done()
        }
        requestAnimationFrame(step)
      }),
    top,
  )
  await page.waitForTimeout(hold)
}

/** Run, then wait for the backtest and for the grid and Monte Carlo that follow it. */
async function run(page: Page) {
  await press(page, page.getByRole('button', { name: 'Run', exact: true }), 0)
  await expect(page.locator('main p', { hasText: /run #\d+/ }).first()).toBeVisible({ timeout: RUN_TIMEOUT })
  await expect(page.locator('main .animate-pulse')).toHaveCount(0, { timeout: RUN_TIMEOUT })
  await page.waitForTimeout(500) // Recharts lays out after its resize observer fires
}

const segment = (page: Page, label: 'IS' | 'OOS' | 'Full') =>
  page.getByRole('group', { name: 'Segment' }).getByRole('button', { name: label, exact: true })

async function record() {
  const video = mkdtempSync(join(tmpdir(), 'demo-video-'))
  const browser = await chromium.launch()
  try {
    const context = await browser.newContext({
      viewport: { width: WIDTH, height: HEIGHT },
      recordVideo: { dir: video, size: { width: WIDTH, height: HEIGHT } },
    })
    await context.addInitScript(CURSOR)
    const page = await context.newPage()

    // KO/PEP pairs over 2015-2025, every other setting default. The URL fills the sidebar.
    await page.goto(`${APP}/?${new URLSearchParams({ start: '2015-01-02', end: '2025-12-31', mcSeed: '42' })}`)
    await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeVisible()
    await page.mouse.move(WIDTH / 2, HEIGHT / 2)
    await page.waitForTimeout(1200)
    await press(page, page.getByRole('button', { name: 'Pairs', exact: true }), 400)
    await press(page, page.getByRole('button', { name: 'KO/PEP', exact: true }), 600)
    await run(page)
    await page.waitForTimeout(2500)

    // The scorecard on the in-sample, out-of-sample and full periods.
    for (const label of ['IS', 'OOS', 'Full'] as const) await press(page, segment(page, label), 1600)

    // A tour of the panels, top to bottom.
    await scrollTo(page, 'Equity and drawdown', 1500)
    const scale = page.getByRole('group', { name: 'Equity scale' })
    await press(page, scale.getByRole('button', { name: 'Log' }), 1800)
    await press(page, scale.getByRole('button', { name: 'Linear' }), 800)
    await scrollTo(page, 'Rolling statistics')
    await scrollTo(page, 'Calendar returns')
    await scrollTo(page, 'Signal', 3000)
    await scrollTo(page, 'Trades', 1800)
    await scrollTo(page, 'Cost sweep')
    await scrollTo(page, 'Parameter sensitivity', 3000)
    await scrollTo(page, 'Walk-forward', 800)
    const wf = panel(page, 'Walk-forward')
    await press(page, wf.getByRole('button', { name: 'Run walk-forward' }), 0)
    await expect(wf.getByText('Out-of-sample days')).toBeVisible({ timeout: RUN_TIMEOUT })
    await page.waitForTimeout(500)
    await scrollTo(page, 'Walk-forward', 3500)
    await scrollTo(page, 'Monte Carlo', 1500)
    const mc = panel(page, 'Monte Carlo')
    for (const model of ['Block bootstrap', 'Stationary', 'GBM']) await press(page, mc.getByRole('button', { name: model, exact: true }), 1400)

    // The diversified trend portfolio: ten ETFs, each sized to the same risk.
    await page.locator('main').evaluate((m) => m.scrollTo({ top: 0, behavior: 'smooth' }))
    await page.waitForTimeout(900)
    await press(page, page.getByRole('button', { name: 'Portfolio', exact: true }), 500)
    await press(page, page.getByRole('button', { name: 'Diversified', exact: true }), 600)
    await run(page)
    await page.waitForTimeout(3000)
    await scrollTo(page, 'Equity and drawdown', 2500)
    await scrollTo(page, 'Signal', 4000)
    await page.locator('main').evaluate((m) => m.scrollTo({ top: 0, behavior: 'smooth' }))
    await page.waitForTimeout(2000)

    await context.close() // finishes writing the video
    mkdirSync(OUT, { recursive: true })
    copyFileSync(await page.video()!.path(), WEBM)
    console.log('saved', WEBM)
  } finally {
    await browser.close()
    rmSync(video, { recursive: true, force: true })
  }
}

/** The ffmpeg that `npx playwright install` puts beside its browsers. */
function playwrightFfmpeg(): string {
  const root =
    process.env.PLAYWRIGHT_BROWSERS_PATH ??
    (process.platform === 'win32'
      ? join(process.env.LOCALAPPDATA ?? '', 'ms-playwright')
      : process.platform === 'darwin'
        ? join(homedir(), 'Library', 'Caches', 'ms-playwright')
        : join(homedir(), '.cache', 'ms-playwright'))
  const exe = process.platform === 'win32' ? 'ffmpeg-win64.exe' : process.platform === 'darwin' ? 'ffmpeg-mac' : 'ffmpeg-linux'
  const dirs = existsSync(root) ? readdirSync(root).filter((d) => d.startsWith('ffmpeg-')).sort().reverse() : []
  const found = dirs.map((d) => join(root, d, exe)).find((p) => existsSync(p))
  if (found === undefined) throw new Error(`no Playwright ffmpeg under ${root}; run npx playwright install chromium`)
  return found
}

/** demo.webm -> demo.gif, at GIF_WIDTH wide and GIF_FPS frames a second. */
function makeGif() {
  const frames = mkdtempSync(join(tmpdir(), 'demo-gif-'))
  try {
    const args = ['-loglevel', 'error', '-i', WEBM, '-vf', `scale=${GIF_WIDTH}:-2`, '-r', String(GIF_FPS), join(frames, '%05d.png')]
    const ffmpeg = spawnSync(playwrightFfmpeg(), args, { encoding: 'utf8' })
    if (ffmpeg.status !== 0) throw new Error(`ffmpeg failed: ${ffmpeg.stderr}`)
    const files = readdirSync(frames).sort().map((f) => join(frames, f))
    const { palette, lookup } = buildPalette(files.filter((_, i) => i % 10 === 0).map(readPng))
    const first = readPng(files[0])
    const gif = new GifWriter(first.width, first.height, palette, lookup)
    files.forEach((f, i) => gif.add(readPng(f), i / GIF_FPS))
    writeFileSync(GIF, gif.finish(files.length / GIF_FPS + GIF_HOLD))
    console.log('saved', GIF, `(${files.length} frames, ${(statSync(GIF).size / 1e6).toFixed(2)} MB)`)
  } finally {
    rmSync(frames, { recursive: true, force: true })
  }
}

if (!args['gif-only']) await record()
makeGif()
