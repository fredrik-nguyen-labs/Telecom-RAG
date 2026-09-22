from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import DOCS_DIR, PROCESSED_KPI_PATH, PROJECT_ROOT, VECTOR_STORE_DIR
from .data import build_kpi_table, save_kpi_table
from .kpi import add_anomaly_scores
from .rag import build_vector_store, index_is_current


@dataclass
class BootstrapReport:
    kpi_ready: bool = False
    docs_ready: bool = False
    vector_ready: bool = False
    kpi_message: str = ""
    docs_message: str = ""
    vector_message: str = ""


def _run_script(path: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    text = "\n".join(x for x in [proc.stdout.strip(), proc.stderr.strip()] if x)
    return proc.returncode == 0, text[-5000:]


def ensure_kpi_data() -> tuple[bool, str]:
    if PROCESSED_KPI_PATH.exists():
        return True, f"Using existing processed KPI table: {PROCESSED_KPI_PATH.name}"

    ok, log = _run_script(PROJECT_ROOT / "scripts" / "download_data.py")
    if not ok:
        return False, (
            "Automatic KPI download failed. Documentation-only RAG can still run. "
            "Details: " + log
        )

    try:
        df = build_kpi_table()
        scored = add_anomaly_scores(df).dataframe
        save_kpi_table(scored)
        return True, f"Built processed KPI table with {len(scored):,} observations."
    except Exception as exc:
        return False, f"KPI preprocessing failed: {exc}"


def ensure_docs() -> tuple[bool, str]:
    readable = [
        p for p in DOCS_DIR.glob("*")
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"} and not p.name.startswith("_")
    ] if DOCS_DIR.exists() else []
    # The configured corpus has 10 sources. Refresh obviously incomplete local corpora,
    # while tolerating a small number of temporary publisher download failures.
    if len(readable) >= 8:
        return True, f"Using {len(readable)} downloaded RAG source files."

    ok, log = _run_script(PROJECT_ROOT / "scripts" / "download_docs.py")
    readable = [
        p for p in DOCS_DIR.glob("*")
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"} and not p.name.startswith("_")
    ] if DOCS_DIR.exists() else []
    if ok and len(readable) >= 4:
        return True, f"Downloaded {len(readable)} RAG source files."
    return False, "RAG document bootstrap failed. " + log


def ensure_vector_index() -> tuple[bool, str]:
    if index_is_current(VECTOR_STORE_DIR):
        return True, "Using current FAISS/BGE index."
    try:
        store = build_vector_store()
        return True, (
            f"Built current FAISS/BGE index with {store.index.ntotal:,} contextual chunks."
        )
    except Exception as exc:
        return False, f"FAISS bootstrap failed: {exc}"


def ensure_local_assets() -> BootstrapReport:
    """Prepare reproducible assets for the local Streamlit development path.

    KPI data is optional: if Dryad is temporarily unavailable, documentation-only RAG
    can still run. The document corpus and local vector index are required.
    """
    report = BootstrapReport()

    report.kpi_ready, report.kpi_message = ensure_kpi_data()
    report.docs_ready, report.docs_message = ensure_docs()

    if report.docs_ready:
        report.vector_ready, report.vector_message = ensure_vector_index()
    else:
        report.vector_ready = False
        report.vector_message = "Skipped because the RAG corpus is unavailable."

    return report
