#!/usr/bin/env python3
"""Download the public documents used by the RAG corpus.

We store source files locally for reproducibility but do not commit the downloaded PDFs.
The source manifest is intentionally small and high quality rather than a large random dump.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "docs" / "corpus"
METADATA_PATH = CORPUS_DIR / "_sources.json"

SOURCES = [
    {
        "source_id": "etsi_ts_138215_v18_5_0",
        "filename": "etsi_ts_138215_v18_5_0.pdf",
        "kind": "pdf",
        "url": "https://www.etsi.org/deliver/etsi_ts/138200_138299/138215/18.05.00_60/ts_138215v180500p.pdf",
        "description": "3GPP TS 38.215 Release 18: NR physical layer measurements.",
    },
    {
        "source_id": "etsi_ts_138214_v18_10_0",
        "filename": "etsi_ts_138214_v18_10_0.pdf",
        "kind": "pdf",
        "url": "https://www.etsi.org/deliver/etsi_ts/138200_138299/138214/18.10.00_60/ts_138214v181000p.pdf",
        "description": "3GPP TS 38.214 Release 18: NR physical layer procedures for data.",
    },
    {
        "source_id": "aerpaw_ericsson_dataset",
        "filename": "aerpaw_ericsson_dataset.txt",
        "kind": "html_to_text",
        "url": "https://aerpaw.org/dataset/aerpaw-ericsson-5g-uav-experiment/",
        "description": "AERPAW dataset description for the Ericsson 5G NSA UAV experiment.",
    },
    {
        "source_id": "aerpaw_post_processing",
        "filename": "aerpaw_post_processing.txt",
        "kind": "html_to_text",
        "url": "https://sites.google.com/ncsu.edu/aerpaw-user-manual/6-sample-experiments-repository/6-1-radio-software/6-1-6-ericsson-experiments/post-processing-of-experiment-logs",
        "description": "AERPAW user manual: Ericsson experiment log post-processing.",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def html_to_clean_text(html: str, url: str, title: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    lines = [line.strip() for line in main.get_text("\n").splitlines()]
    text = "\n".join(line for line in lines if line)
    return f"Title: {title}\nSource URL: {url}\n\n{text}\n"


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Telecom-RAG/1.0 research project"})

    metadata = []
    failures = []
    for source in SOURCES:
        path = CORPUS_DIR / source["filename"]
        print(f"Downloading: {source['description']}")
        try:
            response = session.get(source["url"], timeout=(15, 120))
            response.raise_for_status()
            if source["kind"] == "pdf":
                if not response.content.startswith(b"%PDF"):
                    raise RuntimeError("Response was not a PDF")
                path.write_bytes(response.content)
            else:
                text = html_to_clean_text(
                    response.text, source["url"], source["description"]
                )
                path.write_text(text, encoding="utf-8")

            metadata.append(
                {
                    **source,
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
            print(f"  saved {path}")
        except Exception as exc:
            failures.append({"source_id": source["source_id"], "error": str(exc)})
            print(f"  FAILED: {exc}")

    METADATA_PATH.write_text(
        json.dumps({"sources": metadata, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote source manifest: {METADATA_PATH}")

    if len(metadata) < 2:
        raise SystemExit(
            "Fewer than two corpus documents downloaded. Check your internet connection "
            "or manually download the URLs listed in docs/SOURCES.md."
        )


if __name__ == "__main__":
    main()
