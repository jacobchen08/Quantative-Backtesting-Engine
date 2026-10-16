"""FastAPI application: lifespan (create tables), CORS and routes (§11)."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__, models  # noqa: F401  (models registers the ORM tables on Base)
from app.config import settings
from app.database import Base, engine

logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="Quant Engine", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine_version": __version__}
