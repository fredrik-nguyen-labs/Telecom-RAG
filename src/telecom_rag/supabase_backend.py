from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from supabase import Client, create_client

from .config import (
    CHUNKING_VERSION,
    EMBEDDING_MODEL,
    RERANK_CANDIDATES,
    RERANKER_MODEL,
    RRF_K,
    SUPABASE_SYNC_BATCH_SIZE,
    TOP_K,
)
from .documents import chunk_documents, load_documents
from .retrieval import RetrievalResult


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def supabase_runtime_configured() -> bool:
    """Whether the app should prefer the hosted Supabase backend."""
    return (
        _truthy(os.getenv("USE_SUPABASE"))
        and bool(os.getenv("SUPABASE_URL"))
        and bool(
            os.getenv("SUPABASE_PUBLISHABLE_KEY")
            or os.getenv("SUPABASE_ANON_KEY")
        )
    )


def supabase_admin_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL")) and bool(
        os.getenv("SUPABASE_SECRET_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )


def _runtime_key() -> str | None:
    return os.getenv("SUPABASE_PUBLISHABLE_KEY") or os.getenv("SUPABASE_ANON_KEY")


def _admin_key() -> str | None:
    return os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")


def get_supabase_client(admin: bool = False) -> Client:
    url = os.getenv("SUPABASE_URL")
    key = _admin_key() if admin else _runtime_key()
    expected = (
        "SUPABASE_SECRET_KEY (or legacy SUPABASE_SERVICE_ROLE_KEY)"
        if admin
        else "SUPABASE_PUBLISHABLE_KEY (or legacy SUPABASE_ANON_KEY)"
    )

    if not url or not key:
        raise RuntimeError(
            f"Supabase is not configured. Set SUPABASE_URL and {expected}."
        )
    return create_client(url, key)


def get_supabase_status(client: Client | None = None) -> dict[str, Any]:
    client = client or get_supabase_client(admin=False)
    response = client.rpc(
        "telecom_rag_status",
        {
            "p_embedding_model": EMBEDDING_MODEL,
            "p_chunking_version": CHUNKING_VERSION,
        },
    ).execute()
    return dict(response.data or {})


def supabase_is_seeded(client: Client | None = None) -> bool:
    try:
        status = get_supabase_status(client)
        return int(status.get("document_chunks_current", 0)) > 0
    except Exception:
        return False


def _row_to_document(row: dict[str, Any]) -> Document:
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "chunk_id": row.get("chunk_id"),
            "source_id": row.get("source_id"),
            "source": row.get("source"),
            "title": row.get("title"),
            "page": row.get("page"),
            "section": row.get("section"),
            "similarity": row.get("similarity"),
            "text_rank": row.get("text_rank"),
            "rrf_score": row.get("rrf_score"),
            "retrieval_methods": row.get("retrieval_methods"),
            "retrieval_backend": "supabase",
        }
    )
    return Document(page_content=row["content"], metadata=metadata)


class SupabaseHybridRetriever:
    """Hosted pgvector + Postgres FTS retrieval with local cross-encoder reranking."""

    def __init__(
        self,
        client: Client,
        embeddings: Any,
        reranker_model: str = RERANKER_MODEL,
    ):
        self.client = client
        self.embeddings = embeddings
        self.reranker_model_name = reranker_model
        self._reranker: CrossEncoder | None = None

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    def _query_embedding(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)

    def dense_search(self, query: str, k: int) -> list[Document]:
        response = self.client.rpc(
            "match_document_chunks",
            {
                "p_query_embedding": self._query_embedding(query),
                "p_match_count": int(k),
                "p_embedding_model": EMBEDDING_MODEL,
                "p_chunking_version": CHUNKING_VERSION,
            },
        ).execute()
        return [_row_to_document(row) for row in (response.data or [])]

    def hybrid_search(self, query: str, k: int) -> list[Document]:
        response = self.client.rpc(
            "hybrid_search_document_chunks",
            {
                "p_query_text": query,
                "p_query_embedding": self._query_embedding(query),
                "p_match_count": int(k),
                "p_rrf_k": int(RRF_K),
                "p_embedding_model": EMBEDDING_MODEL,
                "p_chunking_version": CHUNKING_VERSION,
            },
        ).execute()
        return [_row_to_document(row) for row in (response.data or [])]

    def rerank(
        self,
        query: str,
        candidates: list[Document],
        k: int = TOP_K,
    ) -> list[Document]:
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = np.asarray(self._get_reranker().predict(pairs), dtype=float)
        order = np.argsort(-scores)[: min(k, len(candidates))]

        output: list[Document] = []
        for rank, idx in enumerate(order, start=1):
            doc = candidates[int(idx)]
            doc = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            doc.metadata["rerank_rank"] = rank
            doc.metadata["rerank_score"] = float(scores[int(idx)])
            output.append(doc)
        return output

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        mode: str = "reranked",
    ) -> RetrievalResult:
        mode = mode.lower()
        if mode == "dense":
            docs = self.dense_search(query, k)
        elif mode == "hybrid":
            docs = self.hybrid_search(query, k)
        elif mode == "reranked":
            candidates = self.hybrid_search(query, RERANK_CANDIDATES)
            docs = self.rerank(query, candidates, k)
        else:
            raise ValueError("mode must be one of: dense, hybrid, reranked")

        return RetrievalResult(documents=docs, query=query, mode=mode)


def load_supabase_retriever() -> SupabaseHybridRetriever:
    # Lazy import avoids a module cycle: rag owns the embedding implementation.
    from .rag import get_embeddings

    return SupabaseHybridRetriever(
        client=get_supabase_client(admin=False),
        embeddings=get_embeddings(),
    )


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def sync_document_chunks(
    client: Client | None = None,
    batch_size: int = SUPABASE_SYNC_BATCH_SIZE,
) -> int:
    """Embed the local corpus and upsert chunks into Supabase."""
    from .rag import get_embeddings

    client = client or get_supabase_client(admin=True)
    chunks = chunk_documents(load_documents())
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents([doc.page_content for doc in chunks])

    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    for doc, vector in zip(chunks, vectors):
        metadata = {k: _json_value(v) for k, v in doc.metadata.items()}
        rows.append(
            {
                "chunk_id": str(doc.metadata["chunk_id"]),
                "source_id": str(doc.metadata.get("source_id") or ""),
                "source": str(doc.metadata.get("source") or ""),
                "title": _json_value(doc.metadata.get("title")),
                "page": _json_value(doc.metadata.get("page")),
                "section": _json_value(doc.metadata.get("section")),
                "content": doc.page_content,
                "metadata": metadata,
                "embedding": vector,
                "embedding_model": EMBEDDING_MODEL,
                "chunking_version": CHUNKING_VERSION,
                "updated_at": now,
            }
        )

    for batch in _batched(rows, batch_size):
        client.table("document_chunks").upsert(
            batch,
            on_conflict="chunk_id",
        ).execute()

    return len(rows)


_KPI_COLUMNS = [
    "observation_id",
    "timestamp",
    "orientation",
    "latitude",
    "longitude",
    "altitude",
    "lte_cell_id",
    "nr_cell_id",
    "lte_rsrp_dbm",
    "nr_rsrp_dbm",
    "lte_sinr_db",
    "nr_sinr_db",
    "nr_cqi",
    "nr_mcs",
    "nr_ri",
    "throughput_mbps",
    "source_sampling_interval_s",
    "anomaly_score",
    "anomaly_flag",
]


def sync_kpi_observations(
    df: pd.DataFrame,
    client: Client | None = None,
    batch_size: int = 250,
) -> int:
    """Upsert the processed public KPI table into Supabase."""
    client = client or get_supabase_client(admin=True)
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    for _, record in df.iterrows():
        payload = {str(k): _json_value(v) for k, v in record.to_dict().items()}
        row = {
            col: _json_value(record[col]) if col in record.index else None
            for col in _KPI_COLUMNS
        }
        if row.get("lte_cell_id") is not None:
            row["lte_cell_id"] = str(row["lte_cell_id"])
        if row.get("nr_cell_id") is not None:
            row["nr_cell_id"] = str(row["nr_cell_id"])
        if row.get("anomaly_flag") is not None:
            row["anomaly_flag"] = bool(row["anomaly_flag"])
        row["payload"] = payload
        row["updated_at"] = now
        rows.append(row)

    for batch in _batched(rows, batch_size):
        client.table("kpi_observations").upsert(
            batch,
            on_conflict="observation_id",
        ).execute()

    return len(rows)


def load_kpis_from_supabase(
    client: Client | None = None,
    page_size: int = 1000,
) -> pd.DataFrame:
    client = client or get_supabase_client(admin=False)
    rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        response = (
            client.table("kpi_observations")
            .select("*")
            .order("observation_id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = list(response.data or [])
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "payload" in df.columns:
        df = df.drop(columns=["payload", "updated_at"], errors="ignore")
    return df



# Cloudflare Workers AI free allocation and current default-model conversion.
# Cloudflare's pricing page (2026-09) lists Llama 3.2 3B at:
#   4,625 Neurons / 1M input tokens
#  30,475 Neurons / 1M output tokens
CLOUDFLARE_FREE_NEURONS_PER_DAY = 10_000.0
_CLOUDFLARE_NEURON_RATES: dict[str, tuple[float, float]] = {
    "@cf/meta/llama-3.2-3b-instruct": (4625.0, 30475.0),
    "@cf/qwen/qwen3-30b-a3b-fp8": (4625.0, 30475.0),
    "@cf/meta/llama-3.2-1b-instruct": (2457.0, 18252.0),
}


def estimate_cloudflare_neurons(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    rates = _CLOUDFLARE_NEURON_RATES.get(model)
    if rates is None:
        return None
    input_rate, output_rate = rates
    return (
        max(int(input_tokens), 0) * input_rate / 1_000_000
        + max(int(output_tokens), 0) * output_rate / 1_000_000
    )


def record_cloudflare_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    client: Client | None = None,
) -> None:
    client = client or get_supabase_client(admin=False)
    client.rpc(
        "record_cloudflare_usage",
        {
            "p_model": model,
            "p_input_tokens": int(input_tokens),
            "p_output_tokens": int(output_tokens),
        },
    ).execute()


def get_cloudflare_usage_today(
    model: str,
    client: Client | None = None,
) -> dict[str, Any]:
    client = client or get_supabase_client(admin=False)
    response = client.rpc("cloudflare_usage_today").execute()
    usage = dict(response.data or {})
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    estimated = estimate_cloudflare_neurons(model, input_tokens, output_tokens)
    usage["estimated_neurons"] = estimated
    usage["estimated_remaining_neurons"] = (
        max(0.0, CLOUDFLARE_FREE_NEURONS_PER_DAY - estimated)
        if estimated is not None
        else None
    )
    usage["free_neurons_per_day"] = CLOUDFLARE_FREE_NEURONS_PER_DAY
    return usage
