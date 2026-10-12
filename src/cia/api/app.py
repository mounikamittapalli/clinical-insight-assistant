"""FastAPI service.

    uvicorn cia.api.app:app --host 0.0.0.0 --port 8000

GET  /         -> chat web page
GET  /health  -> liveness and index stats
GET  /patients -> sample patient IDs (for example questions)
POST /ask     -> {"question": "...", "mode": "rag" | "prompt_only"}
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cia.config import load_config


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["rag", "prompt_only"] = "rag"
    top_k: int | None = Field(None, ge=1, le=20)


class Citation(BaseModel):
    index: int
    source: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    mode: str
    blocked: bool
    reason: str | None = None
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models load once at startup. Tests inject a fake pipeline instead.
    if getattr(app.state, "pipeline", None) is None:
        from cia.factory import build_pipeline

        cfg = load_config()
        app.state.config = cfg
        app.state.pipeline = build_pipeline(cfg)
    yield


app = FastAPI(
    title="Clinical & Policy Insight Assistant",
    version="0.1.0",
    description="RAG over synthetic patient records and CMS policy documents.",
    lifespan=lifespan,
)


def get_pipeline(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(503, "Pipeline not loaded")
    return pipeline


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Chat web page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/patients")
def sample_patients(request: Request, limit: int = 5) -> dict:
    """A few patient IDs, used by the web page to suggest example questions."""
    cfg = getattr(request.app.state, "config", None)
    db = Path(cfg.paths.sqlite_db) if cfg is not None else None
    if db is None or not db.exists():
        return {"patients": []}
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT Id FROM patients LIMIT ?", (min(limit, 20),)).fetchall()
    return {"patients": [r[0] for r in rows]}


@app.get("/health")
def health(request: Request) -> dict:
    stats = {}
    cfg = getattr(request.app.state, "config", None)
    if cfg is not None:
        stats_file = Path(cfg.paths.index_dir) / "stats.json"
        if stats_file.exists():
            stats = json.loads(stats_file.read_text())
    return {"status": "ok", "index": stats}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: Annotated[object, Depends(get_pipeline)]) -> AskResponse:
    t0 = time.perf_counter()
    result = pipeline.ask(req.question, mode=req.mode, top_k=req.top_k)
    index_of = {id(c): i for i, c in enumerate(result.contexts, start=1)}
    citations = [
        Citation(index=index_of.get(id(c), 0), source=c.chunk.source, snippet=c.chunk.text[:300])
        for c in result.citations
    ]
    return AskResponse(
        answer=result.text,
        citations=citations,
        mode=result.mode,
        blocked=result.blocked,
        reason=result.reason,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
