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

### Phase 1 (approved 2026-10-16)
1. **Extra input checks in `run_backtest` (§6.1, addition).** Besides the §6.1 list, the backtester rejects:
   - a signal outside {−1, 0, +1};
   - weights that are non-finite, or non-zero on a day the signal is flat;
   - non-finite returns or rf (§6.1 only rules out NaN).

   Trades are identified from the executed position (§7), but P&L comes from the executed weights. A weight held on a flat day would earn P&L that no trade owns and would silently break the §7.2 reconciliation. The §6.1 snippet's `assert` is implemented as `raise ValueError`, because Python drops `assert` statements under `-O`.
2. **`resolve_split` computes ⌊0.7·n⌋ in integer arithmetic (§6.3).** `7 * n // 10` is the exact floor. The float expression `int(0.7 * n)` is one too small for n = 90, 170, 180, 650 and 37,443 values of n up to 2,000,000, because 0.7 has no exact binary representation: 0.7 × 650 evaluates to 454.99999999999994. The §12 counts (880 evaluated days → 616 IS) are the same either way.
3. **The Monte Carlo core requires a seed (§10.6).** `monte_carlo.simulate` takes an explicit seed, so the core stays deterministic (§0, rule 8). When a request has no seed, the API layer (Phase 2) draws one with `secrets.randbelow(2**53)` and returns it. `mu_daily` in the result is the drift that GBM and the closed form actually used, so it is 0 with `zero_drift`.
4. **`static_alpha_is` is fitted on raw log prices (§5.4; the spec doesn't say).** The diagnostics regress ln Y on ln X without rebasing, so α̂ reads directly as ln Y = α + β ln X. Rebasing would change α̂ only; β̂, both Engle–Granger p-values and the half-life stay the same.
5. **Warning texts add context (§8).** Each warning keeps the spec's wording and adds the segment, a count or a date. For example: "Out-of-sample: 3 closed trades (fewer than 10), so trade statistics are unreliable."
6. **`run_strategy` and `build_report` are separate functions (§5.0, §9).**
   - `run_strategy` runs §5.0 steps 1–6 and is still the only code that does.
   - `build_report` computes the metrics, benchmark, sizing summary, pairs diagnostics, warnings and chart series from its result.
   - The sensitivity grid calls only `run_strategy`, so it doesn't run two Engle–Granger tests for each of up to 64 configurations.
7. **Two more `null` cases (§2.4).** The Engle–Granger p-value is `null` when a leg is constant over the test window, because statsmodels raises an error on constant input. CAGR is `null` if (E_n / E₀)^(252/n) overflows a float.
8. **The one-day-segment ruin test uses the split date (§12, `test_backtester.py` item 9).** The spec's original example, ruin on the first evaluated day, can't happen: no position is held into that day, because the first position is entered at its close. The test instead puts the ruin day on the split date. That leaves a one-day OOS segment, and the test checks the same `null` fields. CLAUDE.md §12 now says this.
9. **The Pydantic part of that test waits for Phase 2.** "The full response still validates against the Pydantic response model" needs `schemas.py`, which Phase 2 adds; the assertion is added then.
10. **pytest treats every warning as an error (`pyproject.toml`, addition).** The config sets `filterwarnings = ["error"]`, so any warning fails the test that raised it, locally and in CI. In numerical code a `RuntimeWarning` usually signals a silent NaN or inf.
11. **The Phase 1 `conftest.py` has synthetic data only (§12).** It provides the 1,000-day market (a cointegrated Y/X pair and SPY, from a fixed seed) and rf from a constant 2% ^IRX. The fake network, the frozen clock, the yfinance-layout frames and the temp-file database need the loader and the API, so they arrive in Phase 2.
