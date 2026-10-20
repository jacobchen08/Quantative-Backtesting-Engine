interface ImportMetaEnv {
  /** API base URL; empty means same origin (the Vite dev proxy forwards /api). */
  readonly VITE_API_URL?: string
  /** "1" in the static demo build (Phase 9): requests are replayed from public/demo/. */
  readonly VITE_DEMO?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
