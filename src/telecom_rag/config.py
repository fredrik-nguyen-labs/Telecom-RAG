from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "ericsson_5g_nsa"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_KPI_PATH = PROCESSED_DATA_DIR / "kpi_observations.csv"
DOCS_DIR = PROJECT_ROOT / "docs" / "corpus"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
EVAL_PATH = PROJECT_ROOT / "eval" / "questions.json"

# Local retrieval stack.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
)
BGE_QUERY_PREFIX = os.getenv(
    "BGE_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)

# Chunk sizing stays within the embedding model's context window while preserving
# enough surrounding technical context for standards and engineering documents.
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1700"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "250"))
CHUNKING_VERSION = "section-aware-v2"

TOP_K = int(os.getenv("RAG_TOP_K", "4"))
DENSE_CANDIDATES = int(os.getenv("RAG_DENSE_CANDIDATES", "15"))
# BM25/RRF remain available for controlled retrieval ablations, but are not used
# by the default local application path.
BM25_CANDIDATES = int(os.getenv("RAG_BM25_CANDIDATES", "15"))
RERANK_CANDIDATES = int(os.getenv("RAG_RERANK_CANDIDATES", "20"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# Hosted inference. Workers AI exposes an OpenAI-compatible Chat Completions endpoint,
# so the LangChain interface is shared with the rest of the application.
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_GENERATOR_MODEL = os.getenv(
    "CLOUDFLARE_GENERATOR_MODEL", "@cf/google/gemma-4-26b-a4b-it"
)
CLOUDFLARE_ROUTER_MODEL = os.getenv(
    "CLOUDFLARE_ROUTER_MODEL", "@cf/zai-org/glm-4.7-flash"
)
CLOUDFLARE_EMBEDDING_MODEL = os.getenv(
    "CLOUDFLARE_EMBEDDING_MODEL", "@cf/baai/bge-small-en-v1.5"
)
CLOUDFLARE_RERANKER_MODEL = os.getenv(
    "CLOUDFLARE_RERANKER_MODEL", "@cf/baai/bge-reranker-base"
)
USE_CLOUDFLARE_RETRIEVAL = os.getenv(
    "USE_CLOUDFLARE_RETRIEVAL", "true"
).lower() in {"1", "true", "yes", "on"}


# Hosted-application safety defaults.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1400"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "700"))
MAX_REQUEST_UNITS_PER_SESSION = int(os.getenv("MAX_REQUEST_UNITS_PER_SESSION", "12"))


# Hosted persistence/vector sync.
SUPABASE_SYNC_BATCH_SIZE = int(os.getenv("SUPABASE_SYNC_BATCH_SIZE", "50"))
