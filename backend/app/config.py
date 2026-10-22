"""Settings read from environment variables. Defaults match `.env.example`."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    cors_origins: list[str]
    log_level: str


def load_settings() -> Settings:
    origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
    return Settings(
        database_url=os.environ.get("DATABASE_URL", "sqlite:///./quant.db"),
        cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    )


settings = load_settings()


def offline_mode() -> bool:
    """QUANT_OFFLINE=1 (§4.3): a cache miss raises 503 instead of downloading. Read on every call,
    so a test can set it with monkeypatch.setenv."""
    return os.environ.get("QUANT_OFFLINE") == "1"
