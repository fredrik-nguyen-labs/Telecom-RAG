from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import (
    BM25_CANDIDATES,
    DENSE_CANDIDATES,
    RERANK_CANDIDATES,
    RERANKER_MODEL,
    RRF_K,
    TOP_K,
)


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./+-]*", flags=re.IGNORECASE)


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenization that preserves telecom/file tokens such as NR5G_RSRP and .csv."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _doc_key(doc: Document) -> str:
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"{doc.metadata.get('source_id')}|{doc.metadata.get('page')}|{hash(doc.page_content)}"


@dataclass
class RetrievalResult:
    documents: list[Document]
    query: str
    mode: str


class AdvancedRetriever:
    """Dense + BM25 hybrid retrieval with RRF and optional cross-encoder reranking."""

    def __init__(
        self,
        dense_store: FAISS,
        chunks: list[Document],
        reranker_model: str = RERANKER_MODEL,
    ):
        self.dense_store = dense_store
        self.chunks = chunks
        self.reranker_model_name = reranker_model
        self._tokenized = [tokenize_for_bm25(doc.page_content) for doc in chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        self._reranker: CrossEncoder | None = None

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    def dense_search(self, query: str, k: int = DENSE_CANDIDATES) -> list[Document]:
        docs = self.dense_store.similarity_search(query, k=min(k, len(self.chunks)))
        for rank, doc in enumerate(docs, start=1):
            doc.metadata["dense_rank"] = rank
        return docs

    def bm25_search(self, query: str, k: int = BM25_CANDIDATES) -> list[Document]:
        scores = np.asarray(self._bm25.get_scores(tokenize_for_bm25(query)), dtype=float)
        if scores.size == 0:
            return []
        order = np.argsort(-scores)[: min(k, len(scores))]
        docs: list[Document] = []
        for rank, idx in enumerate(order, start=1):
            doc = self.chunks[int(idx)]
            # Clone metadata so one query does not leak rank state into another.
            doc = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            doc.metadata["bm25_rank"] = rank
            doc.metadata["bm25_score"] = float(scores[int(idx)])
            docs.append(doc)
        return docs

    def hybrid_candidates(
        self,
        query: str,
        dense_k: int = DENSE_CANDIDATES,
        bm25_k: int = BM25_CANDIDATES,
        candidate_k: int = RERANK_CANDIDATES,
    ) -> list[Document]:
        """Fuse dense and lexical rankings with reciprocal-rank fusion."""
        rankings = [self.dense_search(query, dense_k), self.bm25_search(query, bm25_k)]
        fused: dict[str, dict] = {}

        for source_name, docs in zip(("dense", "bm25"), rankings):
            for rank, doc in enumerate(docs, start=1):
                key = _doc_key(doc)
                if key not in fused:
                    fused[key] = {
                        "doc": Document(
                            page_content=doc.page_content,
                            metadata=dict(doc.metadata),
                        ),
                        "score": 0.0,
                        "methods": [],
                    }
                fused[key]["score"] += 1.0 / (RRF_K + rank)
                fused[key]["methods"].append(source_name)

        ordered = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
        out: list[Document] = []
        for rank, item in enumerate(ordered[:candidate_k], start=1):
            doc = item["doc"]
            doc.metadata["rrf_rank"] = rank
            doc.metadata["rrf_score"] = float(item["score"])
            doc.metadata["retrieval_methods"] = "+".join(item["methods"])
            out.append(doc)
        return out

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
        out: list[Document] = []
        for rank, idx in enumerate(order, start=1):
            doc = candidates[int(idx)]
            doc = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            doc.metadata["rerank_rank"] = rank
            doc.metadata["rerank_score"] = float(scores[int(idx)])
            out.append(doc)
        return out

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        mode: str = "reranked",
    ) -> RetrievalResult:
        mode = mode.lower()
        if mode == "dense":
            docs = self.dense_search(query, k=k)
        elif mode == "hybrid":
            docs = self.hybrid_candidates(query, candidate_k=k)
        elif mode == "reranked":
            candidates = self.hybrid_candidates(query, candidate_k=RERANK_CANDIDATES)
            docs = self.rerank(query, candidates, k=k)
        else:
            raise ValueError("mode must be one of: dense, hybrid, reranked")
        return RetrievalResult(documents=docs, query=query, mode=mode)


def build_retrieval_query(
    question: str,
    observation: dict | None = None,
) -> str:
    """Build a clean search query without dumping numeric KPI context into embeddings.

    The previous implementation appended the entire percentile/anomaly summary to the
    query. That adds many numbers and generic words that can distract semantic search.
    Here the detailed KPI context is reserved for generation; retrieval only gets the
    original question plus a compact set of relevant telecom concepts.
    """
    question = " ".join(question.split())
    if not observation:
        return question

    lower = question.lower()
    terms: list[str] = []

    mapping = {
        "rsrp": ("lte_rsrp_dbm", "nr_rsrp_dbm"),
        "sinr": ("lte_sinr_db", "nr_sinr_db"),
        "cqi": ("nr_cqi",),
        "mcs": ("nr_mcs",),
        "rank indicator": ("nr_ri",),
        "throughput": ("throughput_mbps",),
    }

    for label, columns in mapping.items():
        if label in lower and any(observation.get(c) is not None for c in columns):
            terms.append(label)

    diagnostic_words = {
        "why", "diagnose", "diagnosis", "performance", "poor", "low",
        "degradation", "issue", "problem", "investigate", "throughput",
    }
    if any(word in lower for word in diagnostic_words):
        available = []
        for label, columns in mapping.items():
            if any(observation.get(c) is not None for c in columns):
                available.append(label)
        terms.extend(available)

    unique_terms = list(dict.fromkeys(terms))
    if not unique_terms:
        return question
    return f"{question} Telecom radio measurements: {' '.join(unique_terms)}"
