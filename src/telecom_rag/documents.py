from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR


def _source_id(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def load_documents(corpus_dir: Path = DOCS_DIR) -> list[Document]:
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Document corpus not found at {corpus_dir}. Run python scripts/download_docs.py."
        )

    docs: list[Document] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        sid = _source_id(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pdf = fitz.open(path)
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source_id": sid,
                            "source": path.name,
                            "page": page_num,
                            "path": str(path),
                        },
                    )
                )
        elif suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source_id": sid,
                            "source": path.name,
                            "page": None,
                            "path": str(path),
                        },
                    )
                )
    if not docs:
        raise RuntimeError(
            f"No readable PDF/TXT/MD documents found under {corpus_dir}. "
            "Run scripts/download_docs.py or add your own public documents."
        )
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i:05d}"
    return chunks
