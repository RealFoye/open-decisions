"""HTTP API.

``POST /v1beta/decisions`` (Gemini-style path) and ``POST /v1/decisions`` take a
:class:`~opendecisions.schema.DecisionRequest` and return a
:class:`~opendecisions.schema.DecisionResponse`.
"""

from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from . import __version__
from .orchestrator import Orchestrator
from .schema import DecisionRequest, DecisionResponse
from .service import DecisionService


def create_app(engine: DecisionService | Orchestrator) -> FastAPI:
    app = FastAPI(title="open-decisions", version=__version__,
                  description="Self-hosted Decisions API: calibrated probabilities for many "
                              "closed-set questions over one shared input.")
    lock = threading.Lock()  # one forward pass at a time on a local model

    def _run(req: DecisionRequest) -> DecisionResponse:
        with lock:
            return engine.decide(req)

    async def decide(req: DecisionRequest) -> DecisionResponse:
        try:
            return await run_in_threadpool(_run, req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    app.post("/v1beta/decisions", response_model=DecisionResponse)(decide)
    app.post("/v1/decisions", response_model=DecisionResponse)(decide)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "model": engine.model_name}

    return app
