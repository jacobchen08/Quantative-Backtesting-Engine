# Methodology

Written in Phase 4 (CLAUDE.md §15).

## Justifications

## Limitations

## Deviations
Approved departures from CLAUDE.md, with reasons (§0, rule 7).

### Phase 0 (approved 2026-10-15)
1. **Frontend linter: oxlint instead of ESLint (§3).** The spec says "ESLint from the Vite template", but the current template (create-vite 9.2) ships oxlint, not ESLint. We kept the template's linter, which preserves the intent: use whatever the template provides rather than a hand-rolled config. `npm run lint` runs oxlint with the template's `.oxlintrc.json`.
2. **Test client dependency: `httpx2` instead of `httpx` (§3, `requirements-dev.txt`).** FastAPI's `TestClient` needs an HTTP client library. Starlette 1.8 deprecates `httpx` for this and emits a `StarletteDeprecationWarning` that asks for `httpx2`. `requirements-dev.txt` pins `httpx2` alongside `pytest` and `ruff`.
3. **`npm test` runs `vitest run --passWithNoTests` until Phase 3.** The frontend has no pure helpers to test yet, and Vitest exits with an error when it finds no test files, which would fail CI. **Phase 3 removes the flag** once the first Vitest tests exist.
4. **CI also runs the backend on `windows-latest` (§3, addition).** The spec asks for Python 3.12 and 3.14. The backend job runs both versions on Ubuntu *and* Windows, because the project is Windows-first (§0, rule 6). The frontend job runs on Ubuntu only.
