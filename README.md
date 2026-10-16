# Quant Engine

A backtesting and Monte Carlo risk engine for daily US equity strategies: mean reversion, pairs trading and time-series momentum, with an optional volatility-targeting overlay. The specification is [CLAUDE.md](CLAUDE.md).

> Work in progress: the backend (price cache, engine and API) is done through Phase 2; the dashboard arrives in Phase 3. The full README (screenshots, benchmarks) comes in Phase 4.

## Setup (PowerShell)

Requires Python 3.12+ and Node 24+.

```powershell
Copy-Item .env.example .env

cd backend
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

cd ..\frontend
npm ci
```

## Run

```powershell
# Terminal 1: API on http://localhost:8000 (interactive docs at /docs)
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --env-file ..\.env

# Terminal 2: dashboard on http://localhost:5173
cd frontend
npm run dev
```

## Tests and linters

```powershell
cd backend
.venv\Scripts\ruff.exe check
.venv\Scripts\ruff.exe format --check
.venv\Scripts\python.exe -m pytest

cd ..\frontend
npm run lint
npm test
npm run build
```
