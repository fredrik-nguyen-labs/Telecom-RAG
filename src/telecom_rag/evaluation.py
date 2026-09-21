from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel

from .config import EVAL_PATH
from .rag import LocalSentenceTransformerEmbeddings, answer_with_rag, answer_without_rag
from .retrieval import AdvancedRetriever


def load_eval_questions(path: Path = EVAL_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_sources(item: dict[str, Any]) -> list[str]:
    if item.get("expected_source_ids"):
        return list(item["expected_source_ids"])
    if item.get("expected_source_id"):
        return [item["expected_source_id"]]
    return []


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
    a = np.asarray(embeddings.embed_text_for_similarity(answer), dtype=float)
    b = np.asarray(embeddings.embed_text_for_similarity(reference), dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else np.nan


def _citation_numbers(answer: str) -> list[int]:
    return [int(n) for n in re.findall(r"\[S(\d+)\]", answer)]


def _citation_validity(answer: str, num_sources: int) -> float:
    cites = _citation_numbers(answer)
    if not cites:
        return 0.0
    return float(all(1 <= n <= num_sources for n in cites))


def _context_support_similarity(
    embeddings: LocalSentenceTransformerEmbeddings,
    answer: str,
    docs: list,
) -> float:
    """Cheap grounding proxy: answer/reference-free similarity to retrieved evidence.

    This is not a factual entailment metric. It is useful as a transparent proxy alongside
    citation validity, retrieval metrics and manual inspection.
    """
    if not docs:
        return np.nan
    answer_vec = np.asarray(embeddings.embed_text_for_similarity(answer), dtype=float)
    context = "\n".join(doc.page_content for doc in docs)
    context_vec = np.asarray(embeddings.embed_text_for_similarity(context), dtype=float)
    denom = np.linalg.norm(answer_vec) * np.linalg.norm(context_vec)
    return float(np.dot(answer_vec, context_vec) / denom) if denom else np.nan


def _retrieval_metrics(docs: list, expected: list[str]) -> dict[str, float]:
    if not expected:
        return {
            "source_hit": np.nan,
            "source_recall": np.nan,
            "source_precision": np.nan,
            "mrr": np.nan,
        }

    source_ids = [doc.metadata.get("source_id", "") for doc in docs]
    expected_set = set(expected)
    matched_positions = [
        i + 1 for i, sid in enumerate(source_ids) if sid in expected_set
    ]
    retrieved_expected = expected_set.intersection(source_ids)
    return {
        "source_hit": float(bool(retrieved_expected)),
        "source_recall": len(retrieved_expected) / len(expected_set),
        "source_precision": (
            sum(sid in expected_set for sid in source_ids) / len(source_ids)
            if source_ids
            else 0.0
        ),
        "mrr": (1.0 / min(matched_positions)) if matched_positions else 0.0,
    }


def evaluate_retrieval(
    retriever: AdvancedRetriever,
    questions: list[dict[str, Any]],
    k: int = 4,
    modes: tuple[str, ...] = ("dense", "hybrid", "reranked"),
) -> pd.DataFrame:
    """Ablate dense-only, hybrid RRF and hybrid+cross-encoder retrieval."""
    rows = []
    for item in questions:
        expected = _expected_sources(item)
        for mode in modes:
            result = retriever.retrieve(item["question"], k=k, mode=mode)
            source_ids = [doc.metadata.get("source_id", "") for doc in result.documents]
            metrics = _retrieval_metrics(result.documents, expected)
            rows.append(
                {
                    "id": item["id"],
                    "category": item.get("category", "unspecified"),
                    "difficulty": item.get("difficulty", "unspecified"),
                    "mode": mode,
                    "question": item["question"],
                    "expected_source_ids": expected,
                    "retrieved_source_ids": source_ids,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_retrieval(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["source_hit", "source_recall", "source_precision", "mrr"]
    return (
        results.groupby("mode")[metrics]
        .mean(numeric_only=True)
        .sort_values(["source_recall", "mrr"], ascending=False)
        .round(3)
    )


def compare_baseline_and_rag(
    llm: BaseChatModel,
    retriever: AdvancedRetriever,
    embeddings: LocalSentenceTransformerEmbeddings,
    questions: list[dict[str, Any]],
    k: int = 4,
    retrieval_mode: str = "reranked",
    limit: int | None = None,
) -> pd.DataFrame:
    """Run the same LLM with and without the final retrieval pipeline."""
    selected = questions if limit is None else questions[:limit]
    rows = []

    for item in selected:
        retrieval = retriever.retrieve(item["question"], k=k, mode=retrieval_mode)
        docs = retrieval.documents
        expected = _expected_sources(item)
        retrieval_metrics = _retrieval_metrics(docs, expected)

        baseline = answer_without_rag(llm, item["question"])
        rag = answer_with_rag(llm, item["question"], docs)

        for mode, result in (("llm_only", baseline), ("rag", rag)):
            answer = result["answer"]
            is_rag = mode == "rag"
            rows.append(
                {
                    "id": item["id"],
                    "category": item.get("category", "unspecified"),
                    "difficulty": item.get("difficulty", "unspecified"),
                    "mode": mode,
                    "question": item["question"],
                    "answer": answer,
                    "semantic_similarity": _semantic_similarity(
                        embeddings, answer, item["reference_answer"]
                    ),
                    "required_term_recall": _term_recall(
                        answer, item.get("required_terms", [])
                    ),
                    "citation_present": float(bool(_citation_numbers(answer))),
                    "citation_validity": (
                        _citation_validity(answer, len(docs)) if is_rag else np.nan
                    ),
                    "context_support_similarity": (
                        _context_support_similarity(embeddings, answer, docs)
                        if is_rag
                        else np.nan
                    ),
                    "latency_s": result["latency_s"],
                    "retrieval_source_hit": (
                        retrieval_metrics["source_hit"] if is_rag else np.nan
                    ),
                    "retrieval_source_recall": (
                        retrieval_metrics["source_recall"] if is_rag else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_comparison(
    results: pd.DataFrame,
    by_category: bool = False,
) -> pd.DataFrame:
    metrics = [
        "semantic_similarity",
        "required_term_recall",
        "citation_present",
        "citation_validity",
        "context_support_similarity",
        "latency_s",
    ]
    available = [m for m in metrics if m in results.columns]
    group_cols = ["category", "mode"] if by_category else ["mode"]
    return results.groupby(group_cols)[available].mean(numeric_only=True).round(3)
