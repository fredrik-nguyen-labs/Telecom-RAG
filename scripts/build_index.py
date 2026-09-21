#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rag.config import VECTOR_STORE_DIR
from telecom_rag.rag import build_vector_store


if __name__ == "__main__":
    store = build_vector_store()
    print(f"FAISS index written to: {VECTOR_STORE_DIR}")
    print(f"Indexed chunks: {store.index.ntotal}")
