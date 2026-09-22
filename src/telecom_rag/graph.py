from __future__ import annotations

import re
import time
from typing import Any, Literal, TypedDict

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from .kpi import analyze_observation
from .rag import answer_with_rag, answer_without_rag
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


DEFINITION_INTENT_RE = re.compile(
    r"\b(?:what\s+does\s+.+?\s+stand\s+for|"
    r"what\s+is\s+(?:an?\s+)?(?:rsrp|sinr|cqi|mcs|rsrq)|"
    r"define\s+(?:rsrp|sinr|cqi|mcs|rsrq)|"
    r"(?:meaning|definition)\s+of\s+(?:rsrp|sinr|cqi|mcs|rsrq)|"
    r"what\s+does\s+(?:rsrp|sinr|cqi|mcs|rsrq)\s+mean)\b",
    flags=re.IGNORECASE,
)

OBSERVATION_REFERENCE_RE = re.compile(
    r"\b(?:this|these|selected|current|my|our)\s+"
    r"(?:observation|measurement|row|sample|values?|kpis?|"
    r"throughput|rsrp|sinr|cqi|mcs|rsrq)\b",
    flags=re.IGNORECASE,
)

DIAGNOSTIC_REFERENCE_RE = re.compile(
    r"\b(?:diagnose|investigate|analy[sz]e|explain)\s+"
    r"(?:this|these|my|our|the\s+selected|the\s+current)\b",
    flags=re.IGNORECASE,
)

KPI_DIAGNOSTIC_INTENT_RE = re.compile(
    r"\b(?:why\s+(?:is|are|might|could)|what\s+(?:stands\s+out|is\s+unusual)|"
    r"diagnose|troubleshoot|investigate)\b.*"
    r"\b(?:throughput|rsrp|rsrq|sinr|cqi|mcs|kpis?|values?|performance)\b",
    flags=re.IGNORECASE,
)

KPI_RELATIONSHIP_INTENT_RE = re.compile(
    r"(?:"
    r"\b(?:correlations?|correlat(?:e|ed|ion)|relationships?|associations?|patterns?|trends?)\b"
    r".*\b(?:values?|kpis?|measurements?|observations?|data|rsrp|rsrq|sinr|cqi|mcs|throughput)\b"
    r"|"
    r"\b(?:values?|kpis?|measurements?|observations?|data|rsrp|rsrq|sinr|cqi|mcs|throughput)\b"
    r".*\b(?:correlations?|correlat(?:e|ed|ion)|relationships?|associations?|patterns?|trends?)\b"
    r")",
    flags=re.IGNORECASE,
)


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


def classify_question_route(question: str, has_observation: bool) -> tuple[str, str]:
    """Return the deterministic route and a human-readable reason.

    Technical definitions never inherit the selected KPI row. KPI analysis requires
    explicit reference to the selected/current measurement or a diagnostic instruction
    aimed at it.
    """
    q = " ".join(question.strip().split())
    if not has_observation:
        return "docs-only", "no selected observation is available"

    if DEFINITION_INTENT_RE.search(q):
        return "docs-only", "definition/abbreviation question"

    if OBSERVATION_REFERENCE_RE.search(q):
        return "kpi+docs", "question explicitly references the selected measurement"

    if DIAGNOSTIC_REFERENCE_RE.search(q):
        return "kpi+docs", "question explicitly asks to diagnose the selected measurement"

    if KPI_DIAGNOSTIC_INTENT_RE.search(q):
        return "kpi+docs", "diagnostic KPI question with observation values available"

    if KPI_RELATIONSHIP_INTENT_RE.search(q):
        return "kpi+docs", "question asks for data relationships/correlations"

    return "docs-only", "no diagnostic reference to the available observation"


def question_needs_kpi(question: str, has_observation: bool) -> bool:
    route, _ = classify_question_route(question, has_observation)
    return route == "kpi+docs"


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
        route, reason = classify_question_route(
            state["question"],
            has_observation=bool(state.get("observation")),
        )
        return {"route": route, "route_reason": reason}

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
        if state.get("route") != "kpi+docs":
            # Hard guard: docs-only output contains only the factual answer and optional
            # technical interpretation. Rebuild the raw answer too so hidden hypothesis
            # text cannot leak into citation parsing or fallback rendering.
            raw_sections = dict(result.get("answer_sections") or {})
            clean_answer, clean_sections = _docs_only_answer(raw_sections)
            result["answer_sections"] = clean_sections
            if clean_answer:
                result["answer"] = clean_answer

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
