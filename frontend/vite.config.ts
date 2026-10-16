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
  test: {
    include: ['src/**/*.test.ts'],
  },
})
