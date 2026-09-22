from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from .kpi import summarize_observation
from .rag import answer_with_rag, answer_without_rag
from .retrieval import RetrieverProtocol, build_retrieval_query


class AppState(TypedDict, total=False):
    question: str
    use_rag: bool
    observation: dict[str, Any] | None
    kpi_context: str
    retrieval_query: str
    retrieval_mode: str
    retrieved_docs: list[Any]
    answer: str
    sources: list[dict[str, Any]]
    latency_s: float
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float
    llm_usage: dict[str, int]
    route: str


KPI_REFERENCE_PHRASES = (
    "this observation",
    "this measurement",
    "this row",
    "this sample",
    "these values",
    "these kpis",
    "these measurements",
    "selected observation",
    "selected measurement",
    "current observation",
    "current measurement",
    "this throughput",
    "this rsrp",
    "this sinr",
    "this cqi",
    "this mcs",
)

KPI_DIAGNOSTIC_PHRASES = (
    "diagnose this",
    "diagnose the selected",
    "investigate this",
    "investigate the selected",
    "what is unusual about this",
    "what is wrong with this",
    "why is this observation",
    "why is this measurement",
    "why might this observation",
    "why might this measurement",
    "what explains this observation",
    "what explains this measurement",
)


def question_needs_kpi(question: str, has_observation: bool) -> bool:
    """Route to KPI analysis only when the user explicitly refers to selected data.

    Merely mentioning RSRP/SINR/throughput is a technical-doc question. This prevents a
    selected UI row from contaminating generic questions such as "What does RSRP mean?".
    """
    if not has_observation:
        return False

    q = " ".join(question.lower().split())
    return any(phrase in q for phrase in KPI_REFERENCE_PHRASES + KPI_DIAGNOSTIC_PHRASES)


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
        needs_kpi = question_needs_kpi(
            state["question"],
            has_observation=bool(state.get("observation")),
        )
        return {"route": "kpi+docs" if needs_kpi else "docs-only"}

    def route_edge(state: AppState) -> Literal["analyze_kpi", "retrieve"]:
        return "analyze_kpi" if state.get("route") == "kpi+docs" else "retrieve"

    def analyze_kpi_node(state: AppState) -> AppState:
        if reference_df is None or not state.get("observation"):
            return {"kpi_context": ""}
        row = pd.Series(state["observation"])
        return {"kpi_context": summarize_observation(row, reference_df)}

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
        result["generation_latency_s"] = generation_latency_s
        result["retrieval_latency_s"] = retrieval_latency_s
        result["total_latency_s"] = retrieval_latency_s + generation_latency_s
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
