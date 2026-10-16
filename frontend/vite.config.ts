import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  envDir: '..', // one .env at the repo root (see .env.example); only VITE_* reaches the browser

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
})
