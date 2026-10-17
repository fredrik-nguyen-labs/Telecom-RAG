from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rag.config import (  # noqa: E402
    CLOUDFLARE_GENERATOR_MODEL,
    CLOUDFLARE_ROUTER_MODEL,
    MAX_OUTPUT_TOKENS,
    MAX_QUESTION_CHARS,
    TOP_K,
)
from telecom_rag.graph import build_graph  # noqa: E402
from telecom_rag.rag import get_llm  # noqa: E402
from telecom_rag.request_context import (  # noqa: E402
    conversation_question,
    sanitize_observation,
)
from telecom_rag.supabase_backend import (  # noqa: E402
    get_supabase_status,
    load_kpis_from_supabase,
    load_supabase_retriever,
    supabase_runtime_configured,
)

logger = logging.getLogger("telecom_rag.api")
MAX_REQUEST_BYTES = 32 * 1024

PUBLIC_KPI_COLUMNS = [
    "observation_id",
    "timestamp",
    "orientation",
    "lte_rsrp_dbm",
    "nr_rsrp_dbm",
    "lte_sinr_db",
    "nr_sinr_db",
    "nr_cqi",
    "nr_mcs",
    "nr_ri",
    "throughput_mbps",
    "anomaly_score",
    "anomaly_flag",
]


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    observation: dict[str, Any] | None = Field(default=None, max_length=24)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=16)


@dataclass
class Runtime:
    graph: Any
    kpis: pd.DataFrame
    status: dict[str, Any]


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int = 3600):
        self.limit = max(limit, 1)
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._checks = 0

    def check(self, key: str) -> int:
        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            self._checks += 1
            if self._checks % 100 == 0:
                for client, times in list(self._events.items()):
                    while times and times[0] <= cutoff:
                        times.popleft()
                    if not times:
                        del self._events[client]
            if key not in self._events and len(self._events) >= 5000:
                raise HTTPException(
                    status_code=429,
                    detail="The service is temporarily busy. Please try later.",
                    headers={"Retry-After": "60"},
                )
            events = self._events[key]
            while events and events[0] < cutoff:
                events.popleft()
            if len(events) >= self.limit:
                raise HTTPException(
                    status_code=429,
                    detail="Request limit reached. Please try again later.",
                    headers={
                        "Retry-After": str(
                            max(1, math.ceil(events[0] + self.window_seconds - now))
                        )
                    },
                )
            events.append(now)
            return self.limit - len(events)


rate_limiter = SlidingWindowRateLimiter(
    int(os.getenv("API_RATE_LIMIT_PER_HOUR", "30"))
)


def _cors_origins() -> list[str]:
    value = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in value.split(",") if origin.strip()]


app = FastAPI(
    title="Telecom-RAG API",
    version="1.0.0",
    description="Hosted API for the 5G network diagnostics RAG assistant.",
    docs_url=None if os.getenv("RENDER") else "/docs",
    redoc_url=None,
    openapi_url=None if os.getenv("RENDER") else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def rejected_request(request: Request, status: int, detail: str) -> JSONResponse:
    response = JSONResponse(status_code=status, content={"detail": detail})
    # Keep browser-readable errors for allowed origins even on early rejections.
    origin = request.headers.get("origin", "")
    if origin in _cors_origins():
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.middleware("http")
async def security_headers(request: Request, call_next):
    # Never log prompts or KPI values. Reject oversized chat payloads.
    if request.method == "POST" and request.url.path == "/api/chat":
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            return rejected_request(request, 415, "Expected application/json.")
        length = request.headers.get("content-length")
        if length is not None:
            try:
                if int(length) > MAX_REQUEST_BYTES:
                    return rejected_request(request, 413, "Request is too large.")
            except ValueError:
                return rejected_request(request, 400, "Invalid content length.")
        if len(await request.body()) > MAX_REQUEST_BYTES:
            return rejected_request(request, 413, "Request is too large.")

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Frame-Options"] = "DENY"
    return response


def _client_key(request: Request) -> str:
    # Never use raw, caller-supplied X-Forwarded-For for identity.
    return request.client.host if request.client else "unknown"


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    if not supabase_runtime_configured():
        raise RuntimeError(
            "Supabase runtime is not configured. Set USE_SUPABASE, SUPABASE_URL, "
            "and SUPABASE_PUBLISHABLE_KEY."
        )
    if not os.getenv("CLOUDFLARE_ACCOUNT_ID") or not os.getenv("CLOUDFLARE_API_TOKEN"):
        raise RuntimeError(
            "Cloudflare Workers AI is not configured. Set CLOUDFLARE_ACCOUNT_ID "
            "and CLOUDFLARE_API_TOKEN."
        )

    status = get_supabase_status()
    if int(status.get("document_chunks_current", 0)) <= 0:
        raise RuntimeError("Supabase is configured but the RAG corpus is not seeded.")

    kpis = load_kpis_from_supabase()
    retriever = load_supabase_retriever()
    llm = get_llm(
        provider="cloudflare",
        model=os.getenv("CLOUDFLARE_GENERATOR_MODEL", CLOUDFLARE_GENERATOR_MODEL),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    router_llm = get_llm(
        provider="cloudflare",
        model=os.getenv("CLOUDFLARE_ROUTER_MODEL", CLOUDFLARE_ROUTER_MODEL),
        max_output_tokens=8,
    )
    graph = build_graph(
        llm,
        retriever,
        router_llm=router_llm,
        reference_df=None if kpis.empty else kpis,
        top_k=TOP_K,
        retrieval_mode="reranked",
    )
    return Runtime(graph=graph, kpis=kpis, status=status)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    try:
        runtime = await asyncio.to_thread(get_runtime)
        return {
            "status": "ok",
            "documents": int(runtime.status.get("document_chunks_current", 0)),
            "observations": int(runtime.status.get("kpi_observations", len(runtime.kpis))),
            "generator": os.getenv(
                "CLOUDFLARE_GENERATOR_MODEL", CLOUDFLARE_GENERATOR_MODEL
            ),
            "router": os.getenv("CLOUDFLARE_ROUTER_MODEL", CLOUDFLARE_ROUTER_MODEL),
        }
    except Exception as exc:
        logger.warning("Health check failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable.") from exc


@app.get("/api/observations")
async def observations(
    limit: int = Query(default=40, ge=1, le=100),
) -> dict[str, Any]:
    try:
        runtime = await asyncio.to_thread(get_runtime)
    except Exception as exc:
        logger.warning("Observation data unavailable (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable.") from exc
    if runtime.kpis.empty:
        return {"items": []}

    frame = runtime.kpis.copy()
    if "anomaly_score" in frame.columns:
        frame = frame.sort_values("anomaly_score", ascending=False, na_position="last")
    columns = [column for column in PUBLIC_KPI_COLUMNS if column in frame.columns]
    items = []
    for record in frame.loc[:, columns].head(limit).to_dict(orient="records"):
        items.append({key: _clean_scalar(value) for key, value in record.items()})
    return {"items": items}


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
    remaining = rate_limiter.check(_client_key(request))
    try:
        runtime = await asyncio.to_thread(get_runtime)
    except Exception as exc:
        logger.warning("Chat runtime unavailable (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable.") from exc

    question = payload.question.strip()
    graph_question = conversation_question(
        question,
        [message.model_dump() for message in payload.history],
    )
    observation = sanitize_observation(payload.observation)

    try:
        result = await asyncio.to_thread(
            runtime.graph.invoke,
            {
                "question": graph_question,
                "use_rag": True,
                "observation": observation,
            },
        )
    except Exception as exc:
        logger.warning("RAG request failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=502, detail="Answer generation failed. Please retry shortly."
        ) from exc

    answer = str(result.get("answer", "")).strip()
    if not answer:
        raise HTTPException(status_code=502, detail="The model returned no visible answer.")

    return {
        "answer": answer,
        "sources": result.get("sources", []),
        "route": result.get("route"),
        "route_reason": result.get("route_reason"),
        "latency_s": result.get("total_latency_s", result.get("latency_s")),
        "rate_limit_remaining": remaining,
    }
