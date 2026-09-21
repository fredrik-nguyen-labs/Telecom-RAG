from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR


_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+){0,4})\s+(?P<title>[A-Za-z][^\n]{2,110})\s*$"
)


def _source_id(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def _clean_pdf_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\x00", " ").splitlines()]
    # Drop repeated blank lines while preserving headings and paragraphs.
    cleaned: list[str] = []
    last_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and last_blank:
            continue
        cleaned.append(line)
        last_blank = blank
    return "\n".join(cleaned).strip()


def _detect_section(text: str) -> str | None:
    """Best-effort section label from the first part of an ETSI/technical page."""
    for raw in text.splitlines()[:45]:
        line = " ".join(raw.split())
        match = _NUMBERED_HEADING.match(line)
        if match:
            return f"{match.group('number')} {match.group('title').strip()}"
    return None


def _text_title(text: str, fallback: str) -> str:
    for line in text.splitlines()[:8]:
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip()
    return fallback


def load_documents(corpus_dir: Path = DOCS_DIR) -> list[Document]:
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Document corpus not found at {corpus_dir}. Run python scripts/download_docs.py."
        )

    docs: list[Document] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.name.startswith(".") or path.name.startswith("_") or not path.is_file():
            continue
        sid = _source_id(path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            pdf = fitz.open(path)
            title = (pdf.metadata or {}).get("title") or path.stem
            for page_num, page in enumerate(pdf, start=1):
                text = _clean_pdf_text(page.get_text("text"))
                if not text:
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source_id": sid,
                            "source": path.name,
                            "title": title,
                            "page": page_num,
                            "section": _detect_section(text),
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
                            "title": _text_title(text, path.stem),
                            "page": None,
                            "section": _detect_section(text),
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


def _context_header(metadata: dict) -> str:
    bits = [
        f"Document: {metadata.get('title') or metadata.get('source', 'unknown')}",
    ]
    if metadata.get("section"):
        bits.append(f"Section: {metadata['section']}")
    if metadata.get("page"):
        bits.append(f"Page: {metadata['page']}")
    return "\n".join(bits)


def chunk_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Section-aware chunking with contextual headers.

    The old baseline embedded raw fixed-size fragments. This version keeps paragraph /
    heading boundaries where possible and prepends document/section/page context to every
    chunk. The header is embedded as part of the chunk, which improves retrieval for
    standards with repeated terminology.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )

    chunks: list[Document] = []
    for doc in docs:
        bodies = splitter.split_text(doc.page_content)
        header = _context_header(doc.metadata)
        for body in bodies:
            body = body.strip()
            if not body:
                continue
            chunks.append(
                Document(
                    page_content=f"{header}\n\n{body}",
                    metadata=dict(doc.metadata),
                )
            )

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i:05d}"
    return chunks
