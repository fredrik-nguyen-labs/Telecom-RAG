from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel

from .config import EVAL_PATH
from .rag import (
    LocalSentenceTransformerEmbeddings,
    answer_with_rag,
    answer_without_rag,
    retrieve,
)


def load_eval_questions(path: Path = EVAL_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _term_recall(answer: str, required_terms: list[str]) -> float:
    if not required_terms:
        return np.nan
    lower = answer.lower()
    hits = sum(term.lower() in lower for term in required_terms)
    return hits / len(required_terms)


def _semantic_similarity(
    embeddings: LocalSentenceTransformerEmbeddings,
    answer: str,
    reference: str,
) -> float:
    a = np.asarray(embeddings.embed_query(answer), dtype=float)
    b = np.asarray(embeddings.embed_query(reference), dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else np.nan


def _has_citation(answer: str) -> bool:
    return bool(re.search(r"\[S\d+\]", answer))


def evaluate_retrieval(store: FAISS, questions: list[dict[str, Any]], k: int = 4) -> pd.DataFrame:
    rows = []
    for item in questions:
        docs = retrieve(store, item["question"], k=k)
        source_ids = [doc.metadata.get("source_id", "") for doc in docs]
        expected = item.get("expected_source_id", "")
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_source_id": expected,
                "retrieved_source_ids": source_ids,
                f"hit@{k}": expected in source_ids if expected else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compare_baseline_and_rag(
    llm: BaseChatModel,
    store: FAISS,
    embeddings: LocalSentenceTransformerEmbeddings,
    questions: list[dict[str, Any]],
    k: int = 4,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run the same LLM with and without retrieval and return transparent metrics."""
    rows = []
    for item in questions[:limit]:
        docs = retrieve(store, item["question"], k=k)
        source_ids = [doc.metadata.get("source_id", "") for doc in docs]
        baseline = answer_without_rag(llm, item["question"])
        rag = answer_with_rag(llm, item["question"], docs)

        for mode, result in (("llm_only", baseline), ("rag", rag)):
            rows.append(
                {
                    "id": item["id"],
                    "mode": mode,
                    "question": item["question"],
                    "answer": result["answer"],
                    "semantic_similarity": _semantic_similarity(
                        embeddings, result["answer"], item["reference_answer"]
                    ),
                    "required_term_recall": _term_recall(
                        result["answer"], item.get("required_terms", [])
                    ),
                    "citation_present": _has_citation(result["answer"]),
                    "latency_s": result["latency_s"],
                    "retrieval_hit": (
                        item.get("expected_source_id") in source_ids if mode == "rag" else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_comparison(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["semantic_similarity", "required_term_recall", "citation_present", "latency_s"]
    available = [m for m in metrics if m in results.columns]
    return results.groupby("mode")[available].mean(numeric_only=True).round(3)
