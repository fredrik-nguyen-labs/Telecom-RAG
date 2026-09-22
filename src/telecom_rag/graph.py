from __future__ import annotations

import json
import time
from typing import Any, Literal, TypedDict

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .kpi import analyze_observation
from .rag import answer_with_rag, answer_without_rag, extract_token_usage
from .retrieval import RetrieverProtocol, build_retrieval_query


class AppState(TypedDict, total=False):
    question: str
    use_rag: bool
    observation: dict[str, Any] | None
    kpi_context: str
    kpi_analysis: dict[str, Any]
    retrieval_query: str
    retrieval_mode: str
    retrieved_docs: list[Any]
    answer: str
    answer_sections: dict[str, str]
    sources: list[dict[str, Any]]
    latency_s: float
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float
    llm_usage: dict[str, int]
    route: str
    route_reason: str
    router_latency_s: float
    router_usage: dict[str, int]


ROUTER_SYSTEM_PROMPT = """You are an intent router for a telecom RAG application.

Choose exactly one route:

- docs-only: the question can be answered from telecom knowledge/technical documents
  without analyzing the currently available KPI observation or the reference KPI dataset.
- kpi+docs: answering the question requires or materially benefits from analyzing the
  available KPI values or statistics from the reference dataset. This includes diagnosis,
  comparison, anomaly/outlier analysis, correlations/relationships/patterns in the data,
  expected-vs-actual behavior, or indirect references such as "the values given".

A selected observation being available is NOT enough by itself to choose kpi+docs.
Generic conceptual questions should remain docs-only even when they mention KPI names.

Return one compact JSON object and nothing else:
{"route":"docs-only"|"kpi+docs","reason":"short explanation"}
"""


def _observation_schema(observation: dict[str, Any] | None) -> str:
    if not observation:
        return "No KPI observation is available."
    fields = [
        str(key)
        for key, value in observation.items()
        if value is not None
        and key not in {"observation_id", "observation_source", "timestamp"}
    ]
    source = observation.get("observation_source", "dataset")
    return (
        f"A KPI observation is available (source={source}). "
        f"Available fields: {', '.join(fields) if fields else 'none'}."
    )


def _parse_router_response(content: Any) -> tuple[str, str] | None:
    text = str(content).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    route = str(payload.get("route", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if route not in {"docs-only", "kpi+docs"}:
        return None
    return route, reason or "semantic intent classification"


def _docs_only_answer(sections: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Keep only factual docs-only sections and rebuild the raw answer consistently."""
    filtered: dict[str, str] = {}
    if sections.get("answer"):
        filtered["answer"] = sections["answer"]
    if sections.get("technical_interpretation"):
        filtered["technical_interpretation"] = sections["technical_interpretation"]

    parts: list[str] = []
    if filtered.get("answer"):
        parts.append(f"## Answer\n{filtered['answer']}")
    if filtered.get("technical_interpretation"):
        parts.append(
            "## Technical interpretation\n"
            f"{filtered['technical_interpretation']}"
        )
    return "\n\n".join(parts).strip(), filtered


def build_graph(
    llm: BaseChatModel,
    retriever: RetrieverProtocol,
    reference_df: pd.DataFrame | None = None,
    top_k: int = 4,
    retrieval_mode: str = "reranked",
):
    """Build the conditional KPI + advanced-RAG workflow.

    Retrieval gets a compact query derived from the question/KPI names, while the full
    numeric KPI context is kept for generation. This avoids polluting the embedding query
    with percentiles, timestamps, anomaly scores and raw values.
    """

    def route_node(state: AppState) -> AppState:
        observation = state.get("observation")
        if not observation:
            return {
                "route": "docs-only",
                "route_reason": "no KPI observation is available",
                "router_latency_s": 0.0,
                "router_usage": {},
            }

        user_prompt = (
            f"Question:\n{state['question']}\n\n"
            f"Available context:\n{_observation_schema(observation)}"
        )
        started = time.perf_counter()
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            router_latency_s = time.perf_counter() - started
            parsed = _parse_router_response(response.content)
            if parsed is None:
                return {
                    "route": "docs-only",
                    "route_reason": "semantic router returned an invalid response",
                    "router_latency_s": router_latency_s,
                    "router_usage": extract_token_usage(response),
                }
            route, reason = parsed
            return {
                "route": route,
                "route_reason": reason,
                "router_latency_s": router_latency_s,
                "router_usage": extract_token_usage(response),
            }
        except Exception:
            return {
                "route": "docs-only",
                "route_reason": "semantic router unavailable; safe docs-only fallback",
                "router_latency_s": time.perf_counter() - started,
                "router_usage": {},
            }

    def route_edge(state: AppState) -> Literal["analyze_kpi", "retrieve"]:
        return "analyze_kpi" if state.get("route") == "kpi+docs" else "retrieve"

    def analyze_kpi_node(state: AppState) -> AppState:
        if reference_df is None or not state.get("observation"):
            return {"kpi_context": "", "kpi_analysis": {}}
        row = pd.Series(state["observation"])
        analysis = analyze_observation(row, reference_df)
        return {
            "kpi_context": str(analysis.get("context", "")),
            "kpi_analysis": analysis,
        }

    def retrieve_node(state: AppState) -> AppState:
        query = build_retrieval_query(
            state["question"],
            observation=state.get("observation") if state.get("route") == "kpi+docs" else None,
        )
        if not state.get("use_rag", True):
            return {
                "retrieval_query": query,
                "retrieval_mode": retrieval_mode,
                "retrieved_docs": [],
                "retrieval_latency_s": 0.0,
            }
        started = time.perf_counter()
        result = retriever.retrieve(query, k=top_k, mode=retrieval_mode)
        retrieval_latency_s = time.perf_counter() - started
        return {
            "retrieval_query": result.query,
            "retrieval_mode": result.mode,
            "retrieved_docs": result.documents,
            "retrieval_latency_s": retrieval_latency_s,
        }

    def generate_node(state: AppState) -> AppState:
        if state.get("use_rag", True):
            result = answer_with_rag(
                llm,
                state["question"],
                state.get("retrieved_docs", []),
                state.get("kpi_context", ""),
            )
        else:
            result = answer_without_rag(
                llm,
                state["question"],
                state.get("kpi_context", ""),
            )

        generation_latency_s = float(result.get("latency_s", 0.0))
        retrieval_latency_s = float(state.get("retrieval_latency_s", 0.0))
        router_latency_s = float(state.get("router_latency_s", 0.0))
        if state.get("route") != "kpi+docs":
            # Hard guard: docs-only output contains only the factual answer and optional
            # technical interpretation. Rebuild the raw answer too so hidden hypothesis
            # text cannot leak into citation parsing or fallback rendering.
            raw_sections = dict(result.get("answer_sections") or {})
            clean_answer, clean_sections = _docs_only_answer(raw_sections)
            result["answer_sections"] = clean_sections
            if clean_answer:
                result["answer"] = clean_answer

        generation_usage = result.get("llm_usage") or {}
        router_usage = state.get("router_usage") or {}
        result["llm_usage"] = {
            "input_tokens": int(generation_usage.get("input_tokens", 0))
            + int(router_usage.get("input_tokens", 0)),
            "output_tokens": int(generation_usage.get("output_tokens", 0))
            + int(router_usage.get("output_tokens", 0)),
            "total_tokens": int(generation_usage.get("total_tokens", 0))
            + int(router_usage.get("total_tokens", 0)),
        }
        result["router_latency_s"] = router_latency_s
        result["generation_latency_s"] = generation_latency_s
        result["retrieval_latency_s"] = retrieval_latency_s
        result["total_latency_s"] = (
            router_latency_s + retrieval_latency_s + generation_latency_s
        )
        # Keep latency_s for UI/backward compatibility, now as end-to-end RAG latency.
        result["latency_s"] = result["total_latency_s"]
        return result

    builder = StateGraph(AppState)
    builder.add_node("route", route_node)
    builder.add_node("analyze_kpi", analyze_kpi_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", route_edge)
    builder.add_edge("analyze_kpi", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
