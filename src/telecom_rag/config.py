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

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
