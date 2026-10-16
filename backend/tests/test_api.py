"""API tests (§12, test_api.py). Phase 0 covers only the health endpoint."""

from fastapi.testclient import TestClient

from app import __version__
from app.main import app


def test_health() -> None:
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "engine_version": __version__}
