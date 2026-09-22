#!/usr/bin/env python3
"""Run the production-style evaluation against hosted Supabase + Cloudflare services.

This path intentionally avoids local SentenceTransformers, FAISS and CrossEncoder model
loading. Results are written as artifacts for inspection; this script does not tune or
modify the system based on the scores.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rag.config import (  # noqa: E402
    CLOUDFLARE_GENERATOR_MODEL,
    CLOUDFLARE_ROUTER_MODEL,
)
from telecom_rag.graph import (  # noqa: E402
    ROUTER_SYSTEM_PROMPT,
    _observation_schema,
    _parse_router_response,
)
from telecom_rag.rag import answer_with_rag, answer_without_rag, get_llm  # noqa: E402
from telecom_rag.supabase_backend import load_supabase_retriever  # noqa: E402


EVAL_QUESTIONS = ROOT / "eval" / "questions.json"
ROUTING_QUESTIONS = ROOT / "eval" / "routing_questions.json"

ROUTING_OBSERVATION = {
    "observation_id": "routing_eval",
    "observation_source": "user-entered",
    "nr_rsrp_dbm": -96.0,
    "nr_sinr_db": 6.0,
    "nr_cqi": 8.0,
    "nr_mcs": 12.0,
    "nr_ri": 2.0,
    "lte_rsrp_dbm": -91.0,
    "lte_sinr_db": 11.0,
    "throughput_mbps": 24.0,
}


def _load_json(path: Path) -> list[dict[str, Any]]:
    return list(json.loads(path.read_text(encoding="utf-8")))


def _term_recall(text: str, required_terms: list[str]) -> float | None:
    if not required_terms:
        return None
    lower = text.lower()
    return sum(term.lower() in lower for term in required_terms) / len(required_terms)


def _citation_numbers(answer: str) -> list[int]:
    return [int(n) for n in re.findall(r"\[S(\d+)\]", answer)]


def _citation_reference_validity(answer: str, num_sources: int) -> float:
    cites = _citation_numbers(answer)
    if not cites:
        return 0.0
    return float(all(1 <= number <= num_sources for number in cites))


def _retrieval_metrics(
    docs: list[Any],
    expected_sources: list[str],
    required_terms: list[str],
) -> dict[str, float | None]:
    source_ids = [str(doc.metadata.get("source_id", "")) for doc in docs]
    expected = set(expected_sources)

    if expected:
        positions = [
            index + 1
            for index, source_id in enumerate(source_ids)
            if source_id in expected
        ]
        retrieved_expected = expected.intersection(source_ids)
        source_hit = float(bool(retrieved_expected))
        source_recall = len(retrieved_expected) / len(expected)
        source_precision = (
            sum(source_id in expected for source_id in source_ids) / len(source_ids)
            if source_ids
            else 0.0
        )
        mrr = 1.0 / min(positions) if positions else 0.0
    else:
        source_hit = source_recall = source_precision = mrr = None

    evidence = "\n".join(doc.page_content for doc in docs)
    return {
        "source_hit": source_hit,
        "source_recall": source_recall,
        "source_precision": source_precision,
        "mrr": mrr,
        "evidence_term_recall": _term_recall(evidence, required_terms),
    }


def evaluate_routing(router_llm: Any, items: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in items:
        expected = str(item["expected_route"])
        has_observation = bool(item.get("has_observation", True))
        started = time.perf_counter()
        error = None

        if not has_observation:
            predicted = "docs-only"
            reason = "no KPI observation is available"
        else:
            prompt = (
                f"Question:\n{item['question']}\n\n"
                f"Available context:\n{_observation_schema(ROUTING_OBSERVATION)}"
            )
            try:
                response = router_llm.invoke(
                    [
                        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                        HumanMessage(content=prompt),
                    ]
                )
                parsed = _parse_router_response(response.content)
                if parsed is None:
                    predicted = "invalid"
                    reason = str(response.content)
                else:
                    predicted, reason = parsed
            except Exception as exc:  # record provider failures without losing the run
                predicted = "error"
                reason = ""
                error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "has_observation": has_observation,
                "expected_route": expected,
                "predicted_route": predicted,
                "correct": predicted == expected,
                "latency_s": time.perf_counter() - started,
                "reason": reason,
                "error": error,
            }
        )

    return pd.DataFrame(rows)


def evaluate_retrieval(
    retriever: Any,
    questions: list[dict[str, Any]],
    modes: tuple[str, ...] = ("dense", "hybrid", "reranked"),
    k: int = 4,
) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[list[Any], float]]]:
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], tuple[list[Any], float]] = {}

    for item in questions:
        for mode in modes:
            started = time.perf_counter()
            docs: list[Any] = []
            error = None
            try:
                result = retriever.retrieve(item["question"], k=k, mode=mode)
                docs = result.documents
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - started
            cache[(item["id"], mode)] = (docs, latency)

            metrics = _retrieval_metrics(
                docs,
                list(item.get("expected_source_ids") or []),
                list(item.get("required_terms") or []),
            )
            rows.append(
                {
                    "id": item["id"],
                    "category": item.get("category"),
                    "difficulty": item.get("difficulty"),
                    "mode": mode,
                    "question": item["question"],
                    "retrieved_source_ids": [
                        doc.metadata.get("source_id") for doc in docs
                    ],
                    "retrieval_methods": [
                        doc.metadata.get("retrieval_methods") for doc in docs
                    ],
                    "reranker_backends": [
                        doc.metadata.get("reranker_backend") for doc in docs
                    ],
                    "retrieval_latency_s": latency,
                    "error": error,
                    **metrics,
                }
            )

    return pd.DataFrame(rows), cache


def evaluate_generation(
    generator: Any,
    questions: list[dict[str, Any]],
    retrieval_cache: dict[tuple[str, str], tuple[list[Any], float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in questions:
        docs, retrieval_latency = retrieval_cache.get(
            (item["id"], "reranked"),
            ([], 0.0),
        )
        required_terms = list(item.get("required_terms") or [])

        for mode in ("llm_only", "rag"):
            error = None
            started = time.perf_counter()
            try:
                if mode == "rag":
                    result = answer_with_rag(generator, item["question"], docs)
                else:
                    result = answer_without_rag(generator, item["question"])
                answer = str(result["answer"])
                generation_latency = float(result.get("latency_s", 0.0))
            except Exception as exc:
                answer = ""
                generation_latency = time.perf_counter() - started
                error = f"{type(exc).__name__}: {exc}"

            retrieval_metrics = _retrieval_metrics(
                docs,
                list(item.get("expected_source_ids") or []),
                required_terms,
            )
            rows.append(
                {
                    "id": item["id"],
                    "category": item.get("category"),
                    "difficulty": item.get("difficulty"),
                    "mode": mode,
                    "question": item["question"],
                    "answer": answer,
                    "answer_words": len(answer.split()),
                    "required_term_recall": _term_recall(answer, required_terms),
                    "citation_present": float(bool(_citation_numbers(answer))),
                    "citation_reference_validity": (
                        _citation_reference_validity(answer, len(docs))
                        if mode == "rag"
                        else None
                    ),
                    "retrieval_source_hit": (
                        retrieval_metrics["source_hit"] if mode == "rag" else None
                    ),
                    "retrieval_source_recall": (
                        retrieval_metrics["source_recall"] if mode == "rag" else None
                    ),
                    "retrieval_mrr": (
                        retrieval_metrics["mrr"] if mode == "rag" else None
                    ),
                    "generation_latency_s": generation_latency,
                    "retrieval_latency_s": (
                        retrieval_latency if mode == "rag" else 0.0
                    ),
                    "total_latency_s": generation_latency
                    + (retrieval_latency if mode == "rag" else 0.0),
                    "error": error,
                }
            )

    return pd.DataFrame(rows)


def _mean_summary(df: pd.DataFrame, group: str, metrics: list[str]) -> pd.DataFrame:
    available = [column for column in metrics if column in df.columns]
    return df.groupby(group)[available].mean(numeric_only=True).round(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "eval" / "results" / "hosted",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of main benchmark questions to run.",
    )
    args = parser.parse_args()

    questions = _load_json(EVAL_QUESTIONS)
    if args.limit is not None:
        questions = questions[: args.limit]
    routing_questions = _load_json(ROUTING_QUESTIONS)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading hosted Supabase/Cloudflare retrieval...")
    retriever = load_supabase_retriever()
    print("Loading hosted generator/router...")
    generator = get_llm(
        provider="cloudflare",
        model=CLOUDFLARE_GENERATOR_MODEL,
    )
    router = get_llm(
        provider="cloudflare",
        model=CLOUDFLARE_ROUTER_MODEL,
        max_output_tokens=8,
    )

    print(f"Routing benchmark: {len(routing_questions)} questions")
    routing = evaluate_routing(router, routing_questions)
    routing.to_csv(output_dir / "routing.csv", index=False)
    routing_accuracy = float(routing["correct"].mean()) if not routing.empty else 0.0
    print(f"Routing accuracy: {routing_accuracy:.3f}")

    print(f"Hosted retrieval benchmark: {len(questions)} questions x 3 modes")
    retrieval, retrieval_cache = evaluate_retrieval(retriever, questions)
    retrieval.to_csv(output_dir / "retrieval.csv", index=False)
    retrieval_summary = _mean_summary(
        retrieval,
        "mode",
        [
            "source_hit",
            "source_recall",
            "source_precision",
            "mrr",
            "evidence_term_recall",
            "retrieval_latency_s",
        ],
    )
    retrieval_summary.to_csv(output_dir / "retrieval_summary.csv")
    print(retrieval_summary.to_string())

    print(f"Hosted generation benchmark: {len(questions)} questions x 2 modes")
    generation = evaluate_generation(generator, questions, retrieval_cache)
    generation.to_csv(output_dir / "generation.csv", index=False)
    generation_summary = _mean_summary(
        generation,
        "mode",
        [
            "required_term_recall",
            "citation_present",
            "citation_reference_validity",
            "answer_words",
            "generation_latency_s",
            "retrieval_latency_s",
            "total_latency_s",
        ],
    )
    generation_summary.to_csv(output_dir / "generation_summary.csv")
    print(generation_summary.to_string())

    metadata = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_count": len(questions),
        "routing_question_count": len(routing_questions),
        "routing_accuracy": routing_accuracy,
        "generator_model": CLOUDFLARE_GENERATOR_MODEL,
        "router_model": CLOUDFLARE_ROUTER_MODEL,
        "notes": (
            "Hosted-only evaluation: Supabase + Cloudflare. "
            "No local SentenceTransformers/FAISS/CrossEncoder models were loaded."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    successful_routing = int(routing["error"].isna().sum())
    successful_retrieval = int(retrieval["error"].isna().sum())
    successful_generation = int(generation["error"].isna().sum())
    if not successful_routing or not successful_retrieval or not successful_generation:
        print("Hosted evaluation produced no successful rows in at least one stage.")
        return 2

    print(f"Results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
