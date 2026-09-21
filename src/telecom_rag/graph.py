from __future__ import annotations

from typing import Any, Literal, TypedDict

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from .kpi import summarize_observation
from .rag import answer_with_rag, answer_without_rag
from .retrieval import AdvancedRetriever, build_retrieval_query


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
    route: str


KPI_TERMS = {
    "observation", "throughput", "rsrp", "rsrq", "sinr", "cqi", "mcs",
    "cell", "kpi", "signal", "performance", "radio", "quality",
}


def build_graph(
    llm: BaseChatModel,
    retriever: AdvancedRetriever,
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
        q = state["question"].lower()
        has_observation = bool(state.get("observation"))
        needs_kpi = has_observation and any(term in q for term in KPI_TERMS)
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
            }
        result = retriever.retrieve(query, k=top_k, mode=retrieval_mode)
        return {
            "retrieval_query": result.query,
            "retrieval_mode": result.mode,
            "retrieved_docs": result.documents,
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
