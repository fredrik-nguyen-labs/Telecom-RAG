from __future__ import annotations

from langchain_core.documents import Document

from telecom_rag.retrieval import AdvancedRetriever


class FakeDenseStore:
    def __init__(self, docs: list[Document]):
        self.docs = docs

    def similarity_search(self, query: str, k: int):
        return [
            Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            for doc in self.docs[:k]
        ]


def _docs() -> list[Document]:
    return [
        Document(
            page_content="NR RSRP reference signal received power",
            metadata={"chunk_id": "a", "source_id": "s1"},
        ),
        Document(
            page_content="SINR describes signal interference and noise",
            metadata={"chunk_id": "b", "source_id": "s2"},
        ),
        Document(
            page_content="CQI supports channel quality reporting",
            metadata={"chunk_id": "c", "source_id": "s3"},
        ),
    ]


def test_bm25_returns_no_arbitrary_documents_for_zero_match() -> None:
    docs = _docs()
    retriever = AdvancedRetriever(FakeDenseStore(docs), docs)

    assert retriever.bm25_search("banana pineapple", k=3) == []


def test_rrf_marks_documents_retrieved_by_both_methods() -> None:
    docs = _docs()
    retriever = AdvancedRetriever(FakeDenseStore(docs), docs)

    fused = retriever.hybrid_candidates(
        "RSRP reference signal",
        dense_k=2,
        bm25_k=3,
        candidate_k=3,
    )

    by_id = {doc.metadata["chunk_id"]: doc for doc in fused}
    assert "a" in by_id
    assert by_id["a"].metadata["retrieval_methods"] == "dense+bm25"
    assert by_id["a"].metadata["rrf_score"] > 0


def test_dense_reranked_uses_only_dense_candidates() -> None:
    docs = _docs()
    retriever = AdvancedRetriever(FakeDenseStore(docs), docs)

    captured: list[str] = []

    def fake_rerank(query: str, candidates: list[Document], k: int):
        captured.extend(doc.metadata["chunk_id"] for doc in candidates)
        return candidates[:k]

    retriever.rerank = fake_rerank  # type: ignore[method-assign]
    result = retriever.retrieve("RSRP", k=2, mode="dense_reranked")

    assert result.mode == "dense_reranked"
    assert captured == ["a", "b", "c"]
    assert [doc.metadata["chunk_id"] for doc in result.documents] == ["a", "b"]
