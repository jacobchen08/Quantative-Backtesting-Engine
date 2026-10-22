// Browser tests (§12, "Browser tests"): `npx playwright test` in frontend/, locally and in CI.
//
// webServer starts every server on a port of its own, so a dev API on :8000 or a Vite dev server on
// :5173 is never reused or disturbed:
// - the API on :8001 with QUANT_OFFLINE=1 and a temporary SQLite database that
//   backend/scripts/seed_e2e.py fills with the synthetic market (2012-01-03 to 2019-12-31), so every
//   price read is a cache hit and a miss is a 503, never a download;
// - `vite preview` on :4173, serving a fresh build in dist-e2e/ that calls the API on :8001;
// - `vite preview` on :4174, serving the static demo's build from dist-demo/.
import { defineConfig, devices } from '@playwright/test'
import { existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const API_PORT = 8001
const WEB_PORT = 4173
const DEMO_PORT = 4174
const BACKEND = join(import.meta.dirname, '..', 'backend')

// The backend's venv if there is one (the README's setup), otherwise the python on PATH (CI).
const VENV_PYTHON = process.platform === 'win32' ? join('.venv', 'Scripts', 'python.exe') : join('.venv', 'bin', 'python')
const PYTHON = process.env.E2E_PYTHON ?? (existsSync(join(BACKEND, VENV_PYTHON)) ? VENV_PYTHON : 'python')
const DB = join(tmpdir(), 'quant-engine-e2e.db') // recreated by seed_e2e.py on every start

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1, // the tests share one database, and the run-history test reads what was stored
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  timeout: 60_000,
  expect: { timeout: 20_000 },
  use: {
    baseURL: `http://localhost:${WEB_PORT}`,
    viewport: { width: 1440, height: 900 }, // a desktop tool, at least 1280 px wide (§13)
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } }],
  webServer: [
    {
      command: `"${PYTHON}" scripts/seed_e2e.py "${DB}" && "${PYTHON}" -m uvicorn app.main:app --port ${API_PORT}`,
      cwd: BACKEND,
      url: `http://localhost:${API_PORT}/api/health`,
      env: {
        QUANT_OFFLINE: '1',
        DATABASE_URL: `sqlite:///${DB.replaceAll('\\', '/')}`,
        CORS_ORIGINS: `http://localhost:${WEB_PORT}`,
        LOG_LEVEL: 'WARNING',
      },
      reuseExistingServer: false,
      timeout: 120_000,
      stdout: 'pipe',
    },
    {
      command: `npx tsc -b && npx vite build --outDir dist-e2e --emptyOutDir && npx vite preview --outDir dist-e2e --port ${WEB_PORT} --strictPort`,
      url: `http://localhost:${WEB_PORT}`,
      env: { VITE_API_URL: `http://localhost:${API_PORT}` },
      reuseExistingServer: false,
      timeout: 180_000,
    },
    {
      // The static demo (e2e/demo.spec.ts): its build in dist-demo/, served under its GitHub
      // Pages path. It replays public/demo/ and calls no API. (The build above already typechecks.)
      command: `npx vite build --mode demo --outDir dist-demo --emptyOutDir && npx vite preview --mode demo --outDir dist-demo --port ${DEMO_PORT} --strictPort`,
      url: `http://localhost:${DEMO_PORT}/Quantitative-Backtesting-Engine/`,
      reuseExistingServer: false,
      timeout: 180_000,
    },
  ],
})
