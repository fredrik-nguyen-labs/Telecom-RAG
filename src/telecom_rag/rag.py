from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import (
    BGE_QUERY_PREFIX,
    CHUNKING_VERSION,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_GENERATOR_MODEL,
    EMBEDDING_MODEL,
    MAX_OUTPUT_TOKENS,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    RERANKER_MODEL,
    VECTOR_STORE_DIR,
)
from .retrieval import AdvancedRetriever


MANIFEST_NAME = "manifest.json"
CHUNKS_NAME = "chunks.json"


class LocalSentenceTransformerEmbeddings(Embeddings):
    """LangChain adapter with retrieval-specific BGE query handling."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def _query_text(self, text: str) -> str:
        if "bge-" in self.model_name.lower():
            return BGE_QUERY_PREFIX + text
        return text

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            [self._query_text(text)],
            normalize_embeddings=True,
        )[0]
        return np.asarray(vector, dtype=np.float32).tolist()

    def embed_text_for_similarity(self, text: str) -> list[float]:
        """Embed plain text without retrieval query instructions for evaluation."""
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vector, dtype=np.float32).tolist()


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> LocalSentenceTransformerEmbeddings:
    return LocalSentenceTransformerEmbeddings(model_name=model_name)


def _expected_manifest(embedding_model: str = EMBEDDING_MODEL) -> dict[str, Any]:
    return {
        "embedding_model": embedding_model,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunking_version": CHUNKING_VERSION,
    }


def _read_manifest(index_dir: Path) -> dict[str, Any] | None:
    path = index_dir / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_chunks(chunks: list[Document], index_dir: Path) -> None:
    payload = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in chunks
    ]
    (index_dir / CHUNKS_NAME).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def load_index_chunks(index_dir: Path = VECTOR_STORE_DIR) -> list[Document]:
    path = index_dir / CHUNKS_NAME
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in payload
        ]

    # Backward-compatible fallback for indexes created before chunk persistence.
    from .documents import chunk_documents, load_documents

    return chunk_documents(load_documents())


def build_vector_store(
    index_dir: Path = VECTOR_STORE_DIR,
    embedding_model: str = EMBEDDING_MODEL,
) -> Any:
    from langchain_community.vectorstores import FAISS
    from .documents import chunk_documents, load_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    embeddings = get_embeddings(embedding_model)
    store = FAISS.from_documents(chunks, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    _write_chunks(chunks, index_dir)
    (index_dir / MANIFEST_NAME).write_text(
        json.dumps(_expected_manifest(embedding_model), indent=2) + "\n",
        encoding="utf-8",
    )
    return store


def index_is_current(
    index_dir: Path = VECTOR_STORE_DIR,
    embedding_model: str = EMBEDDING_MODEL,
) -> bool:
    required = [
        index_dir / "index.faiss",
        index_dir / "index.pkl",
        index_dir / MANIFEST_NAME,
        index_dir / CHUNKS_NAME,
    ]
    return all(path.exists() for path in required) and (
        _read_manifest(index_dir) == _expected_manifest(embedding_model)
    )


def load_vector_store(
    index_dir: Path = VECTOR_STORE_DIR,
    embedding_model: str = EMBEDDING_MODEL,
    build_if_missing: bool = True,
) -> Any:
    from langchain_community.vectorstores import FAISS

    embeddings = get_embeddings(embedding_model)

    if index_is_current(index_dir, embedding_model):
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if not build_if_missing:
        actual = _read_manifest(index_dir)
        raise FileNotFoundError(
            "FAISS index is missing or stale for the current retrieval configuration. "
            f"Expected {_expected_manifest(embedding_model)}, found {actual}."
        )

    # Important when switching between same-dimensional embedding models: rebuild instead
    # of silently reusing vectors created by a different model.
    return build_vector_store(index_dir=index_dir, embedding_model=embedding_model)


def load_advanced_retriever(
    index_dir: Path = VECTOR_STORE_DIR,
    embedding_model: str = EMBEDDING_MODEL,
    reranker_model: str = RERANKER_MODEL,
    build_if_missing: bool = True,
) -> AdvancedRetriever:
    store = load_vector_store(
        index_dir=index_dir,
        embedding_model=embedding_model,
        build_if_missing=build_if_missing,
    )
    chunks = load_index_chunks(index_dir)
    return AdvancedRetriever(
        dense_store=store,
        chunks=chunks,
        reranker_model=reranker_model,
    )


def _message_role(message: Any) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    return str(getattr(message, "type", "user"))


def _visible_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content or "").strip()


class CloudflareWorkersAIChat:
    """Small LangChain-compatible adapter around Workers AI Chat Completions.

    The hosted path uses Cloudflare's documented request schema directly instead of
    relying on an OpenAI client to translate model-specific fields such as
    chat_template_kwargs and max_completion_tokens.
    """

    def __init__(
        self,
        *,
        account_id: str,
        api_token: str,
        model: str,
        max_output_tokens: int,
        timeout_s: float = 60.0,
    ) -> None:
        self.account_id = account_id
        self.api_token = api_token
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.base_url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/v1/chat/completions"
        )

    def invoke(self, messages: list[Any]) -> AIMessage:
        payload_messages = [
            {
                "role": _message_role(message),
                "content": getattr(message, "content", str(message)),
            }
            for message in messages
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": 0,
            "max_completion_tokens": self.max_output_tokens,
            "stream": False,
        }

        # Both deployed models are reasoning-capable. For an interactive RAG app we
        # want the token budget spent on visible answer text, not hidden reasoning.
        if (
            "gemma-4" in self.model.lower()
            or "glm-4.7" in self.model.lower()
        ):
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_s,
        )
        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[:500]
            raise RuntimeError(
                f"Cloudflare Workers AI HTTP {response.status_code}: {detail}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(
                "Cloudflare Workers AI returned no completion choices."
            )

        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = _visible_message_text(message.get("content"))
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage") or {}

        if not content:
            raise RuntimeError(
                "Cloudflare Workers AI returned an empty assistant content "
                f"(model={self.model}, finish_reason={finish_reason}, usage={usage})."
            )

        return AIMessage(
            content=content,
            response_metadata={
                "provider": "cloudflare",
                "model": data.get("model") or self.model,
                "finish_reason": finish_reason,
                "usage": usage,
            },
        )


def get_llm(
    provider: str = "ollama",
    model: str | None = None,
    max_output_tokens: int | None = None,
) -> Any:
    """Return a deterministic, bounded-output chat model for the application."""
    provider = provider.lower().strip()
    output_limit = max_output_tokens or MAX_OUTPUT_TOKENS
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or OLLAMA_MODEL,
            temperature=0,
            num_predict=output_limit,
        )
    if provider == "cloudflare":
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", CLOUDFLARE_ACCOUNT_ID)
        api_token = os.getenv("CLOUDFLARE_API_TOKEN", CLOUDFLARE_API_TOKEN)
        if not account_id or not api_token:
            raise RuntimeError(
                "Cloudflare Workers AI is not configured. Set "
                "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN."
            )

        return CloudflareWorkersAIChat(
            account_id=account_id,
            api_token=api_token,
            model=model
            or os.getenv("CLOUDFLARE_GENERATOR_MODEL", CLOUDFLARE_GENERATOR_MODEL),
            max_output_tokens=output_limit,
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
    raise ValueError("provider must be 'ollama', 'cloudflare', or 'openai'")


def format_context(docs: list[Document]) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        section = doc.metadata.get("section")
        source_id = doc.metadata.get("source_id", source)
        page_text = f", page {page}" if page else ""
        section_text = f", section {section}" if section else ""
        blocks.append(
            f"[S{idx}] {source}{page_text}{section_text}\n{doc.page_content}"
        )
        sources.append(
            {
                "citation": f"S{idx}",
                "source": source,
                "source_id": source_id,
                "title": doc.metadata.get("title"),
                "page": page,
                "section": section,
                "chunk_id": doc.metadata.get("chunk_id"),
                "retrieval_methods": doc.metadata.get("retrieval_methods"),
                "rerank_score": doc.metadata.get("rerank_score"),
                "reranker_backend": doc.metadata.get("reranker_backend"),
                "reranker_fallback_reason": doc.metadata.get("reranker_fallback_reason"),
                "excerpt": doc.page_content[:650],
                "content": doc.page_content,
            }
        )
    return "\n\n---\n\n".join(blocks), sources


DOCS_RAG_SYSTEM_PROMPT = """You are a telecom technical assistant.
Answer from the retrieved technical context and cite source-supported claims with [S1],
[S2], etc. Do not use or discuss KPI observations, statistical evidence, diagnoses, or
hypotheses. If the retrieved context is insufficient, say what is missing rather than
guessing.

Return Markdown with:
## Answer
A direct answer to the question.

Optionally, when useful:
## Technical interpretation
A concise source-supported explanation of the mechanism.

Give a complete, technically useful answer at the depth the question deserves. For
explanatory or analytical questions, explain the important relationships and mechanisms
instead of reducing the answer to a few sentences. Avoid filler, but do not artificially
shorten the response."""


KPI_RAG_SYSTEM_PROMPT = """You are a telecom network analysis assistant.
Use the retrieved technical context as the factual knowledge source for technical claims.
Treat the supplied KPI context as observation evidence. It may come from a real dataset
row or from user-entered KPI values; preserve that distinction.

The KPI context may include percentiles, rank correlations, nearest-neighbor comparisons,
cross-KPI consistency checks, and anomaly/rarity statistics computed before generation.
Treat these as descriptive statistical evidence, not causal proof. Correlations are
computed across many observations in the reference dataset; never imply that a
correlation was estimated from a single selected/custom row.

Return Markdown with:
## Answer
A direct answer to the user's question.

## Observation evidence
Only the KPI values/statistics relevant to the question. Prefer diagnostic statistical
findings over repeating every number.

Optionally, when useful:
## Technical interpretation
A source-supported technical explanation.

Only when a causal explanation is genuinely uncertain and useful:
## Hypotheses
Clearly qualified possible explanations and what additional evidence would distinguish
them. Never present hypotheses as measured facts.

Cite source-supported technical claims with [S1], [S2], etc. Do not invent universal
thresholds not supported by a source. Give enough detail to connect the statistical
evidence to the technical mechanism. Prioritize the strongest findings, explain why they
matter, and distinguish clearly between evidence and inference. Avoid filler, but do not
artificially shorten the response."""

BASELINE_SYSTEM_PROMPT = """You are a telecom network analysis assistant.
Answer from your pretrained knowledge only. If you are unsure, say so. Do not invent
citations or pretend you consulted documents. Give a complete technical answer at the
depth the question deserves rather than defaulting to a very short response."""


_SECTION_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?"
    r"(Answer|Measured evidence|Observation evidence|Technical interpretation|"
    r"Hypothesis|Hypotheses)\s*:?[ \t]*$"
)


def parse_answer_sections(text: str) -> dict[str, str]:
    """Parse the model's stable Markdown section contract for card-based rendering."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return {"answer": text.strip()} if text.strip() else {}

    key_map = {
        "answer": "answer",
        "measured evidence": "observation_evidence",
        "observation evidence": "observation_evidence",
        "technical interpretation": "technical_interpretation",
        "hypothesis": "hypotheses",
        "hypotheses": "hypotheses",
    }
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections[key_map[match.group(1).lower()]] = content

    # If the model accidentally emitted prose before the first heading, keep it visible.
    preamble = text[: matches[0].start()].strip()
    if preamble and "answer" not in sections:
        sections["answer"] = preamble
    return sections


def answer_with_rag(
    llm: BaseChatModel,
    question: str,
    retrieved_docs: list[Document],
    kpi_context: str = "",
    use_kpi_context: bool = False,
) -> dict[str, Any]:
    context, sources = format_context(retrieved_docs)
    user = f"Question:\n{question}\n\n"
    if kpi_context:
        user += (
            "KPI context (observation values + dataset-relative statistics; "
            "no citation required for the values themselves):\n"
            f"{kpi_context}\n\n"
        )
    user += f"Retrieved technical context:\n{context}"

    start = time.perf_counter()
    system_prompt = (
        KPI_RAG_SYSTEM_PROMPT if use_kpi_context else DOCS_RAG_SYSTEM_PROMPT
    )
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user)]
    )
    latency = time.perf_counter() - start
    answer_text = str(response.content)
    return {
        "answer": answer_text,
        "answer_sections": parse_answer_sections(answer_text),
        "sources": sources,
        "latency_s": latency,
    }


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
    return {
        "answer": str(response.content),
        "sources": [],
        "latency_s": latency,
    }
