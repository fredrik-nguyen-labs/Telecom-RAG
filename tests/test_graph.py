from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from telecom_rag.graph import _parse_router_response, build_graph
from telecom_rag.retrieval import RetrievalResult


class EmptyRetriever:
    def retrieve(self, query: str, k: int = 4, mode: str = "reranked") -> RetrievalResult:
        return RetrievalResult(documents=[], query=query, mode=mode)


def test_router_parser_accepts_expected_tokens() -> None:
    assert _parse_router_response("KPI") == (
        "kpi+docs",
        "semantic router selected KPI/data analysis",
    )
    assert _parse_router_response("**DOCS**") == (
        "docs-only",
        "semantic router selected technical-document analysis",
    )
    assert _parse_router_response("something else") is None


def test_graph_uses_kpi_route_for_semantic_router_decision() -> None:
    router = FakeListChatModel(responses=["KPI"])
    generator = FakeListChatModel(responses=["## Answer\nKPI answer"])
    graph = build_graph(
        generator,
        EmptyRetriever(),
        router_llm=router,
        reference_df=None,
    )

    result = graph.invoke(
        {
            "question": "What stands out in these measurements?",
            "use_rag": True,
            "observation": {
                "observation_source": "user-entered",
                "nr_rsrp_dbm": -95,
                "nr_sinr_db": 5,
                "throughput_mbps": 20,
            },
        }
    )

    assert result["route"] == "kpi+docs"


def test_docs_route_strips_hypothesis_section() -> None:
    router = FakeListChatModel(responses=["DOCS"])
    generator = FakeListChatModel(
        responses=[
            "## Answer\nRSRP is a reference-signal power measurement.\n\n"
            "## Hypotheses\nThis section must not survive a docs-only route."
        ]
    )
    graph = build_graph(
        generator,
        EmptyRetriever(),
        router_llm=router,
        reference_df=None,
    )

    result = graph.invoke(
        {
            "question": "What is RSRP?",
            "use_rag": True,
            "observation": {"nr_rsrp_dbm": -95},
        }
    )

    assert result["route"] == "docs-only"
    assert "hypoth" not in result["answer"].lower()
    assert "hypotheses" not in result["answer_sections"]


def test_no_observation_skips_router_and_uses_docs() -> None:
    router = FakeListChatModel(responses=["KPI"])
    generator = FakeListChatModel(responses=["## Answer\nGeneral answer"])
    graph = build_graph(generator, EmptyRetriever(), router_llm=router)

    result = graph.invoke(
        {
            "question": "Diagnose these values",
            "use_rag": True,
            "observation": None,
        }
    )

    assert result["route"] == "docs-only"
    assert result["router_latency_s"] == 0.0
