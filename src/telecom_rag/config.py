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

# Retrieval stack. BGE is retrieval-specific while remaining small enough for CPU demos.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
)
BGE_QUERY_PREFIX = os.getenv(
    "BGE_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)

# ~400-token chunks are a better fit for the 512-token embedding window than the
# previous 1200-character baseline, while still giving technical definitions context.
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1700"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "250"))
CHUNKING_VERSION = "section-aware-v2"

TOP_K = int(os.getenv("RAG_TOP_K", "4"))
DENSE_CANDIDATES = int(os.getenv("RAG_DENSE_CANDIDATES", "15"))
BM25_CANDIDATES = int(os.getenv("RAG_BM25_CANDIDATES", "15"))
RERANK_CANDIDATES = int(os.getenv("RAG_RERANK_CANDIDATES", "20"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

# Free hosted generation for the public demo. Workers AI exposes an
# OpenAI-compatible Chat Completions endpoint, so the LangChain interface stays shared.
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_MODEL = os.getenv(
    "CLOUDFLARE_MODEL", "@cf/meta/llama-3.2-3b-instruct"
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

# Optional paid fallback.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

# Public-demo safety defaults. Environment variables can tighten these further.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "650"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "700"))
MAX_REQUEST_UNITS_PER_SESSION = int(os.getenv("MAX_REQUEST_UNITS_PER_SESSION", "12"))
MAX_TOP_K_PUBLIC = int(os.getenv("MAX_TOP_K_PUBLIC", "5"))


# Optional hosted persistence/vector backend.
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
# New Supabase API key names (legacy anon/service_role are still accepted in backend code).
SUPABASE_PUBLISHABLE_KEY = os.getenv(
    "SUPABASE_PUBLISHABLE_KEY", os.getenv("SUPABASE_ANON_KEY", "")
)
SUPABASE_SECRET_KEY = os.getenv(
    "SUPABASE_SECRET_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
)
USE_SUPABASE = os.getenv("USE_SUPABASE", "").lower() in {"1", "true", "yes", "on"}
SUPABASE_SYNC_BATCH_SIZE = int(os.getenv("SUPABASE_SYNC_BATCH_SIZE", "50"))
