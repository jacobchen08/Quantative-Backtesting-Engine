interface ImportMetaEnv {
  /** API base URL; empty means same origin (the Vite dev proxy forwards /api). */
  readonly VITE_API_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
