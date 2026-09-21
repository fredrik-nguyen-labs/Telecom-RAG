from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_MODEL,
    MAX_OUTPUT_TOKENS,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    TOP_K,
    VECTOR_STORE_DIR,
)
from .documents import chunk_documents, load_documents


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Small LangChain Embeddings adapter around sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vector, dtype=np.float32).tolist()


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> LocalSentenceTransformerEmbeddings:
    return LocalSentenceTransformerEmbeddings(model_name=model_name)


def build_vector_store(
    index_dir: Path = VECTOR_STORE_DIR,
    embedding_model: str = EMBEDDING_MODEL,
) -> FAISS:
    docs = load_documents()
    chunks = chunk_documents(docs)
    embeddings = get_embeddings(embedding_model)
    store = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    return store


def load_vector_store(
    index_dir: Path = VECTOR_STORE_DIR,
    embedding_model: str = EMBEDDING_MODEL,
    build_if_missing: bool = True,
) -> FAISS:
    embeddings = get_embeddings(embedding_model)
    if (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists():
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    if not build_if_missing:
        raise FileNotFoundError(f"FAISS index not found at {index_dir}")
    docs = load_documents()
    chunks = chunk_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    return store


def get_llm(provider: str = "ollama", model: str | None = None) -> BaseChatModel:
    """Return a deterministic, bounded-output chat model for the demo."""
    provider = provider.lower().strip()
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or OLLAMA_MODEL,
            temperature=0,
            num_predict=MAX_OUTPUT_TOKENS,
        )
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or OPENAI_MODEL,
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=45,
            max_retries=1,
        )
    raise ValueError("provider must be 'ollama' or 'openai'")


def retrieve(
    store: FAISS,
    query: str,
    k: int = TOP_K,
) -> list[Document]:
    return store.similarity_search(query, k=k)


def format_context(docs: list[Document]) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        source_id = doc.metadata.get("source_id", source)
        page_text = f", page {page}" if page else ""
        blocks.append(f"[S{idx}] {source}{page_text}\n{doc.page_content}")
        sources.append(
            {
                "citation": f"S{idx}",
                "source": source,
                "source_id": source_id,
                "page": page,
                "chunk_id": doc.metadata.get("chunk_id"),
                "excerpt": doc.page_content[:500],
            }
        )
    return "\n\n---\n\n".join(blocks), sources


RAG_SYSTEM_PROMPT = """You are a telecom network analysis assistant.
Use the retrieved technical context as your factual knowledge source. If KPI context is
provided, treat it as measured evidence, but do not invent universal thresholds that are
not present in the sources. Distinguish measured facts, dataset-relative statistics, and
possible explanations. If the context is insufficient, say what is missing.

Cite factual statements from retrieved context using [S1], [S2], etc. Do not cite a source
that does not support the statement. Keep the answer concise but technically useful."""

BASELINE_SYSTEM_PROMPT = """You are a telecom network analysis assistant.
Answer from your pretrained knowledge only. If you are unsure, say so. Do not invent
citations or pretend you consulted documents. Keep the answer concise and technical."""


def answer_with_rag(
    llm: BaseChatModel,
    question: str,
    retrieved_docs: list[Document],
    kpi_context: str = "",
) -> dict[str, Any]:
    context, sources = format_context(retrieved_docs)
    user = f"Question:\n{question}\n\n"
    if kpi_context:
        user += f"KPI context:\n{kpi_context}\n\n"
    user += f"Retrieved technical context:\n{context}"

    start = time.perf_counter()
    response = llm.invoke([SystemMessage(content=RAG_SYSTEM_PROMPT), HumanMessage(content=user)])
    latency = time.perf_counter() - start
    return {"answer": str(response.content), "sources": sources, "latency_s": latency}


def answer_without_rag(
    llm: BaseChatModel,
    question: str,
    kpi_context: str = "",
) -> dict[str, Any]:
    user = f"Question:\n{question}"
    if kpi_context:
        user += f"\n\nKPI context:\n{kpi_context}"
    start = time.perf_counter()
    response = llm.invoke(
        [SystemMessage(content=BASELINE_SYSTEM_PROMPT), HumanMessage(content=user)]
    )
    latency = time.perf_counter() - start
    return {"answer": str(response.content), "sources": [], "latency_s": latency}
