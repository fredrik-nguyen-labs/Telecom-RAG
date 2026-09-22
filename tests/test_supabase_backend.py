from __future__ import annotations

from langchain_core.documents import Document

from telecom_rag.supabase_backend import SupabaseHybridRetriever


class DummyEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 384


def test_cloudflare_reranker_failure_falls_back_to_rrf(monkeypatch) -> None:
    retriever = SupabaseHybridRetriever(
        client=object(),
        embeddings=DummyEmbeddings(),
        cloudflare_account_id="account",
        cloudflare_api_token="token",
    )
    candidates = [
        Document(page_content="first", metadata={"chunk_id": "a"}),
        Document(page_content="second", metadata={"chunk_id": "b"}),
    ]

    def fail(*args, **kwargs):
        raise RuntimeError("temporary hosted reranker failure")

    monkeypatch.setattr(retriever, "_cloudflare_rerank_scores", fail)
    output = retriever.rerank("query", candidates, k=2)

    assert [doc.metadata["chunk_id"] for doc in output] == ["a", "b"]
    assert all(
        doc.metadata["reranker_backend"] == "hybrid-rrf-fallback"
        for doc in output
    )
    assert all(doc.metadata["rerank_score"] is None for doc in output)
