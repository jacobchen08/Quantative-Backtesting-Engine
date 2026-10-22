import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { rmSync } from 'node:fs'
import { resolve } from 'node:path'
import { loadEnv, type Plugin } from 'vite'
import { defineConfig } from 'vitest/config'

// The static demo's path on GitHub Pages (§13, "Static demo"): https://<user>.github.io/<repository>/.
// pages.yml passes the repository's name; this default is the name it has now.
const DEMO_BASE = '/Quantitative-Backtesting-Engine/'

/**
 * Leaves the recorded demo responses out of every build but the demo's. Vite copies all of public/
 * into the output, and the live dashboard has no use for 9 MB of recorded JSON.
 */
function dropDemoData(): Plugin {
  let outDir = ''
  return {
    name: 'drop-demo-data',
    apply: 'build',
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir) // dist, or dist-e2e for the browser tests
    },
    closeBundle() {
      rmSync(resolve(outDir, 'demo'), { recursive: true, force: true })
    },
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '..', 'VITE_') // process.env wins over the .env file, as in the app
  // `vite build --mode demo` (npm run build:demo) or VITE_DEMO=1: the static demo.
  const demo = mode === 'demo' || env.VITE_DEMO === '1'
  return {
    plugins: [react(), tailwindcss(), ...(demo ? [] : [dropDemoData()])],
    envDir: '..', // one .env at the repo root (see .env.example); only VITE_* reaches the browser
    base: demo ? (process.env.DEMO_BASE ?? DEMO_BASE) : '/',
    define: demo ? { 'import.meta.env.VITE_DEMO': JSON.stringify('1') } : {},

    server: {
      // VITE_API_URL defaults to same origin; in dev, /api is forwarded to uvicorn (§13).
      proxy: { '/api': 'http://localhost:8000' },
    },
    build: {
      rolldownOptions: {
        output: {
          // Libraries in their own chunks: each stays under Vite's 500 kB advisory limit, and they
          // stay cached in the browser across app rebuilds.
          codeSplitting: {
            groups: [
              { name: 'react', test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/, priority: 2 },
              { name: 'charts', test: /node_modules[\\/]/, priority: 1 },
            ],
          },
        },
      },
    },
    test: {
      include: ['src/**/*.test.ts'],
    },
  }
})
