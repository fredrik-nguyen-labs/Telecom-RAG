from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel

from .config import EVAL_PATH
from .rag import LocalSentenceTransformerEmbeddings, answer_with_rag, answer_without_rag
from .retrieval import RetrieverProtocol


def load_eval_questions(path: Path = EVAL_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_sources(item: dict[str, Any]) -> list[str]:
    if item.get("expected_source_ids"):
        return list(item["expected_source_ids"])
    if item.get("expected_source_id"):
        return [item["expected_source_id"]]
    return []


def _term_recall(text: str, required_terms: list[str]) -> float:
    if not required_terms:
        return np.nan
    lower = text.lower()
    hits = sum(term.lower() in lower for term in required_terms)
    return hits / len(required_terms)


def _semantic_similarity(
    embeddings: LocalSentenceTransformerEmbeddings,
    text_a: str,
    text_b: str,
) -> float:
    a = np.asarray(embeddings.embed_text_for_similarity(text_a), dtype=float)
    b = np.asarray(embeddings.embed_text_for_similarity(text_b), dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else np.nan


def _citation_numbers(answer: str) -> list[int]:
    return [int(n) for n in re.findall(r"\[S(\d+)\]", answer)]


def _citation_reference_validity(answer: str, num_sources: int) -> float:
    """Whether every citation marker points to a retrieved source that exists.

    This intentionally does *not* claim that the cited source supports the sentence.
    """
    cites = _citation_numbers(answer)
    if not cites:
        return 0.0
    return float(all(1 <= n <= num_sources for n in cites))


def _citation_support_similarity(
    embeddings: LocalSentenceTransformerEmbeddings,
    answer: str,
    docs: list,
) -> float:
    """Embedding-similarity proxy between cited claims and their cited chunks.

    This is stricter than checking that [S1] exists, but it is still not an entailment
    or factuality judge. It should be interpreted as a transparent support proxy.
    """
    if not docs:
        return np.nan

    # Capture sentence-ish spans ending at punctuation/newline while preserving citation tags.
    spans = [
        span.strip()
        for span in re.split(r"(?<=[.!?])\s+|\n+", answer)
        if span.strip() and _citation_numbers(span)
    ]
    if not spans:
        return 0.0

    scores: list[float] = []
    for span in spans:
        cites = _citation_numbers(span)
        claim = re.sub(r"\[S\d+\]", "", span).strip()
        if not claim:
            continue
        valid_docs = [docs[n - 1] for n in cites if 1 <= n <= len(docs)]
        if not valid_docs:
            scores.append(0.0)
            continue
        cited_context = "\n".join(doc.page_content for doc in valid_docs)
        scores.append(_semantic_similarity(embeddings, claim, cited_context))

    return float(np.mean(scores)) if scores else 0.0


def _answer_context_similarity(
    embeddings: LocalSentenceTransformerEmbeddings,
    answer: str,
    docs: list,
) -> float:
    """Whole-answer similarity to retrieved evidence; a coarse grounding proxy only."""
    if not docs:
        return np.nan
    context = "\n".join(doc.page_content for doc in docs)
    return _semantic_similarity(embeddings, answer, context)


def _evidence_term_recall(docs: list, required_terms: list[str]) -> float:
    """Passage-level evidence coverage using benchmark-required terms.

    This catches cases where the expected PDF is retrieved but the returned top-k chunks
    do not contain the benchmark's key evidence. It is lexical and intentionally simple.
    """
    if not required_terms:
        return np.nan
    evidence = "\n".join(doc.page_content for doc in docs)
    return _term_recall(evidence, required_terms)


def _retrieval_metrics(
    docs: list,
    expected: list[str],
    required_terms: list[str] | None = None,
) -> dict[str, float]:
    required_terms = required_terms or []
    source_ids = [doc.metadata.get("source_id", "") for doc in docs]

    if expected:
        expected_set = set(expected)
        matched_positions = [
            i + 1 for i, sid in enumerate(source_ids) if sid in expected_set
        ]
        retrieved_expected = expected_set.intersection(source_ids)
        source_hit = float(bool(retrieved_expected))
        source_recall = len(retrieved_expected) / len(expected_set)
        source_precision = (
            sum(sid in expected_set for sid in source_ids) / len(source_ids)
            if source_ids
            else 0.0
        )
        mrr = (1.0 / min(matched_positions)) if matched_positions else 0.0
    else:
        source_hit = source_recall = source_precision = mrr = np.nan

    return {
        "source_hit": source_hit,
        "source_recall": source_recall,
        "source_precision": source_precision,
        "mrr": mrr,
        "evidence_term_recall": _evidence_term_recall(docs, required_terms),
    }


def evaluate_retrieval(
    retriever: RetrieverProtocol,
    questions: list[dict[str, Any]],
    k: int = 4,
    modes: tuple[str, ...] = ("dense", "hybrid", "reranked"),
) -> pd.DataFrame:
    """Ablate dense-only, hybrid RRF and hybrid+cross-encoder retrieval."""
    rows = []
    for item in questions:
        expected = _expected_sources(item)
        required_terms = item.get("required_terms", [])
        for mode in modes:
            started = time.perf_counter()
            result = retriever.retrieve(item["question"], k=k, mode=mode)
            retrieval_latency_s = time.perf_counter() - started
            source_ids = [
                doc.metadata.get("source_id", "") for doc in result.documents
            ]
            metrics = _retrieval_metrics(
                result.documents,
                expected,
                required_terms=required_terms,
            )
            rows.append(
                {
                    "id": item["id"],
                    "category": item.get("category", "unspecified"),
                    "difficulty": item.get("difficulty", "unspecified"),
                    "mode": mode,
                    "question": item["question"],
                    "expected_source_ids": expected,
                    "retrieved_source_ids": source_ids,
                    "retrieval_latency_s": retrieval_latency_s,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_retrieval(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "source_hit",
        "source_recall",
        "source_precision",
        "mrr",
        "evidence_term_recall",
        "retrieval_latency_s",
    ]
    available = [m for m in metrics if m in results.columns]
    return (
        results.groupby("mode")[available]
        .mean(numeric_only=True)
        .sort_values(["source_recall", "mrr"], ascending=False)
        .round(3)
    )


def compare_baseline_and_rag(
    llm: BaseChatModel,
    retriever: RetrieverProtocol,
    embeddings: LocalSentenceTransformerEmbeddings,
    questions: list[dict[str, Any]],
    k: int = 4,
    retrieval_mode: str = "reranked",
    limit: int | None = None,
) -> pd.DataFrame:
    """Run the same LLM with and without final retrieval and record stage latency."""
    selected = questions if limit is None else questions[:limit]
    rows = []

    for item in selected:
        retrieval_started = time.perf_counter()
        retrieval = retriever.retrieve(
            item["question"],
            k=k,
            mode=retrieval_mode,
        )
        retrieval_latency_s = time.perf_counter() - retrieval_started
        docs = retrieval.documents
        expected = _expected_sources(item)
        required_terms = item.get("required_terms", [])
        retrieval_metrics = _retrieval_metrics(
            docs,
            expected,
            required_terms=required_terms,
        )

        baseline_started = time.perf_counter()
        baseline = answer_without_rag(llm, item["question"])
        baseline_total_latency_s = time.perf_counter() - baseline_started

        rag_started = time.perf_counter()
        rag = answer_with_rag(llm, item["question"], docs)
        rag_generation_total_s = time.perf_counter() - rag_started
        rag_total_latency_s = retrieval_latency_s + rag_generation_total_s

        for mode, result in (("llm_only", baseline), ("rag", rag)):
            answer = result["answer"]
            is_rag = mode == "rag"
            generation_latency_s = float(result["latency_s"])
            total_latency_s = (
                rag_total_latency_s if is_rag else baseline_total_latency_s
            )
            rows.append(
                {
                    "id": item["id"],
                    "category": item.get("category", "unspecified"),
                    "difficulty": item.get("difficulty", "unspecified"),
                    "mode": mode,
                    "question": item["question"],
                    "answer": answer,
                    "semantic_similarity": _semantic_similarity(
                        embeddings,
                        answer,
                        item["reference_answer"],
                    ),
                    "required_term_recall": _term_recall(
                        answer,
                        required_terms,
                    ),
                    "citation_present": float(bool(_citation_numbers(answer))),
                    "citation_reference_validity": (
                        _citation_reference_validity(answer, len(docs))
                        if is_rag
                        else np.nan
                    ),
                    "citation_support_similarity": (
                        _citation_support_similarity(embeddings, answer, docs)
                        if is_rag
                        else np.nan
                    ),
                    "answer_context_similarity": (
                        _answer_context_similarity(embeddings, answer, docs)
                        if is_rag
                        else np.nan
                    ),
                    "evidence_term_recall": (
                        retrieval_metrics["evidence_term_recall"]
                        if is_rag
                        else np.nan
                    ),
                    "generation_latency_s": generation_latency_s,
                    "retrieval_latency_s": (
                        retrieval_latency_s if is_rag else 0.0
                    ),
                    "total_latency_s": total_latency_s,
                    # Backward-compatible alias; now explicitly end-to-end for the row.
                    "latency_s": total_latency_s,
                    "retrieval_source_hit": (
                        retrieval_metrics["source_hit"] if is_rag else np.nan
                    ),
                    "retrieval_source_recall": (
                        retrieval_metrics["source_recall"] if is_rag else np.nan
                    ),
                    "retrieval_source_precision": (
                        retrieval_metrics["source_precision"] if is_rag else np.nan
                    ),
                    "retrieval_mrr": (
                        retrieval_metrics["mrr"] if is_rag else np.nan
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
        "citation_reference_validity",
        "citation_support_similarity",
        "answer_context_similarity",
        "evidence_term_recall",
        "generation_latency_s",
        "retrieval_latency_s",
        "total_latency_s",
    ]
    available = [m for m in metrics if m in results.columns]
    group_cols = ["category", "mode"] if by_category else ["mode"]
    return (
        results.groupby(group_cols)[available]
        .mean(numeric_only=True)
        .round(3)
    )
