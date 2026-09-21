#!/usr/bin/env python3
"""Download the public documents used by the RAG corpus.

The corpus intentionally mixes standards (definitions/procedures), exact AERPAW
experiment documentation, and applied Ericsson performance material. Downloaded third-
party content is reproducible but not committed to Git.
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
        "topic": "nr_measurements",
    },
    {
        "source_id": "etsi_ts_138214_v18_10_0",
        "filename": "etsi_ts_138214_v18_10_0.pdf",
        "kind": "pdf",
        "url": "https://www.etsi.org/deliver/etsi_ts/138200_138299/138214/18.10.00_60/ts_138214v181000p.pdf",
        "description": "3GPP TS 38.214 Release 18: NR physical layer procedures for data.",
        "topic": "nr_data_procedures",
    },
    {
        "source_id": "etsi_ts_138300_v18_5_0",
        "filename": "etsi_ts_138300_v18_5_0.pdf",
        "kind": "pdf",
        "url": "https://www.etsi.org/deliver/etsi_ts/138300_138399/138300/18.05.00_60/ts_138300v180500p.pdf",
        "description": "3GPP TS 38.300 Release 18: NR and NG-RAN overall description.",
        "topic": "nr_architecture_and_radio_procedures",
    },
    {
        "source_id": "etsi_ts_136214_v18_0_0",
        "filename": "etsi_ts_136214_v18_0_0.pdf",
        "kind": "pdf",
        "url": "https://www.etsi.org/deliver/etsi_ts/136200_136299/136214/18.00.00_60/ts_136214v180000p.pdf",
        "description": "3GPP TS 36.214 Release 18: LTE physical layer measurements.",
        "topic": "lte_measurements",
    },
    {
        "source_id": "aerpaw_ericsson_dataset",
        "filename": "aerpaw_ericsson_dataset.txt",
        "kind": "html_to_text",
        "url": "https://aerpaw.org/dataset/aerpaw-ericsson-5g-uav-experiment/",
        "description": "AERPAW dataset description for the Ericsson 5G NSA UAV experiment.",
        "topic": "experiment",
    },
    {
        "source_id": "aerpaw_post_processing",
        "filename": "aerpaw_post_processing.txt",
        "kind": "html_to_text",
        "url": "https://sites.google.com/ncsu.edu/aerpaw-user-manual/6-sample-experiments-repository/6-1-radio-software/6-1-6-ericsson-experiments/post-processing-of-experiment-logs",
        "description": "AERPAW user manual: Ericsson experiment log post-processing.",
        "topic": "experiment_processing",
    },
    {
        "source_id": "ericsson_massive_mimo_beamforming",
        "filename": "ericsson_massive_mimo_beamforming.txt",
        "kind": "html_to_text",
        "url": "https://www.ericsson.com/en/reports-and-papers/ericsson-technology-review/articles/beamforming-in-massive-mimo",
        "description": "Ericsson Technology Review: beamforming in Massive MIMO.",
        "topic": "nr_performance",
    },
    {
        "source_id": "ericsson_traffic_patterns",
        "filename": "ericsson_traffic_patterns.txt",
        "kind": "html_to_text",
        "url": "https://www.ericsson.com/en/reports-and-papers/mobility-report/articles/traffic-patterns-drive-network-evolution",
        "description": "Ericsson Mobility Report: traffic patterns and network evolution.",
        "topic": "coverage_capacity_throughput",
    },
    {
        "source_id": "ericsson_ai_network_performance",
        "filename": "ericsson_ai_network_performance.txt",
        "kind": "html_to_text",
        "url": "https://www.ericsson.com/en/reports-and-papers/mobility-report/articles/reinforcement-learning",
        "description": "Ericsson Mobility Report: AI and network performance optimization.",
        "topic": "network_optimization",
    },
    {
        "source_id": "ericsson_mobility_report_june_2025",
        "filename": "ericsson_mobility_report_june_2025.pdf",
        "kind": "pdf",
        "url": "https://www.ericsson.com/49e9b6/assets/local/reports-papers/mobility-report/documents/2025/ericsson-mobility-report-june-2025.pdf",
        "description": "Ericsson Mobility Report June 2025.",
        "topic": "real_world_5g_performance",
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
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    lines = [line.strip() for line in main.get_text("\n").splitlines()]
    lines = [line for line in lines if line]

    # Remove adjacent duplicate lines common in CMS-rendered pages.
    deduped: list[str] = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    text = "\n".join(deduped)
    return f"Title: {title}\nSource URL: {url}\n\n{text}\n"


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Telecom-RAG/2.0 research portfolio project",
            "Accept": "text/html,application/pdf,application/octet-stream,*/*",
        }
    )

    metadata = []
    failures = []

    for source in SOURCES:
        path = CORPUS_DIR / source["filename"]
        print(f"Downloading: {source['description']}")
        try:
            response = session.get(source["url"], timeout=(15, 120), allow_redirects=True)
            response.raise_for_status()

            if source["kind"] == "pdf":
                if not response.content.startswith(b"%PDF"):
                    raise RuntimeError("Response was not a PDF")
                path.write_bytes(response.content)
            else:
                text = html_to_clean_text(
                    response.text, source["url"], source["description"]
                )
                if len(text) < 500:
                    raise RuntimeError("Extracted webpage text was unexpectedly short")
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
            failures.append(
                {
                    "source_id": source["source_id"],
                    "url": source["url"],
                    "error": str(exc),
                }
            )
            print(f"  FAILED: {exc}")

    METADATA_PATH.write_text(
        json.dumps(
            {
                "configured_source_count": len(SOURCES),
                "successful_source_count": len(metadata),
                "sources": metadata,
                "failures": failures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote source manifest: {METADATA_PATH}")
    print(f"Downloaded {len(metadata)}/{len(SOURCES)} configured sources.")

    # Keep the app usable even if one external publisher temporarily rejects downloads.
    if len(metadata) < 4:
        raise SystemExit(
            "Fewer than four corpus documents downloaded. Check your internet connection "
            "or manually download the URLs listed in docs/SOURCES.md."
        )


if __name__ == "__main__":
    main()
