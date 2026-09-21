#!/usr/bin/env python3
"""Download the exact KPI dataset used by Telecom-RAG.

Dataset:
    Ericsson 5G NSA network RF and throughput measurements on AERPAW network
    DOI: 10.5061/dryad.wh70rxx06
    Dryad version: May 21, 2025
    File: Ericsson_Amir.zip
    Dryad file id: 4078259

The raw dataset is kept out of Git and reproduced locally with this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests


DOI = "10.5061/dryad.wh70rxx06"
LANDING_PAGE = "https://datadryad.org/dataset/doi:10.5061/dryad.wh70rxx06"
ARCHIVE_NAME = "Ericsson_Amir.zip"
DRYAD_FILE_ID = "4078259"
DOWNLOAD_URL = f"https://datadryad.org/downloads/file_stream/{DRYAD_FILE_ID}"
# Dryad's legacy file-stream endpoint may be blocked by its anti-bot layer.
# This API endpoint assembles the same pinned May 21, 2025 release and returns
# a short-lived signed ZIP URL without requiring an API token.
API_DOWNLOAD_URL = "https://datadryad.org/api/v2/versions/365672/download"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "downloads"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ericsson_5g_nsa"
ARCHIVE_PATH = DOWNLOAD_DIR / ARCHIVE_NAME
METADATA_PATH = RAW_DIR / "_download_metadata.json"

# Basenames documented by the dataset authors and required for our first analysis.
EXPECTED_BASENAMES = {
    "input_throughput_with_header.csv",
    "inputf1_sinr_with_header.csv",
    "inputf2_sinr_with_header.csv",
    "inputf1_rsrp_with_header.csv",
    "inputf2_rsrp_with_header.csv",
    "inputf1_cellid_with_header.csv",
    "inputf2_cellid_with_header.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(force: bool = False) -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists() and not force:
        if zipfile.is_zipfile(ARCHIVE_PATH):
            print(f"Using existing archive: {ARCHIVE_PATH}")
            return
        print("Existing archive is not a valid ZIP; downloading it again.")

    tmp_path = ARCHIVE_PATH.with_suffix(".zip.part")
    tmp_path.unlink(missing_ok=True)

    headers = {
        "User-Agent": "Telecom-RAG/1.0 dataset downloader",
        "Accept": "application/zip,application/octet-stream,*/*",
        "Referer": LANDING_PAGE,
    }

    print(f"Downloading {ARCHIVE_NAME}")
    print(f"Source: {LANDING_PAGE}")

    last_error: Exception | None = None
    urls = (DOWNLOAD_URL, API_DOWNLOAD_URL)
    for index, url in enumerate(urls):
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(15, 120),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)

            if zipfile.is_zipfile(tmp_path):
                tmp_path.replace(ARCHIVE_PATH)
                print(f"Saved: {ARCHIVE_PATH}")
                return

            last_error = RuntimeError(f"Response from {url} was not a ZIP archive")
            tmp_path.unlink(missing_ok=True)
            if index + 1 < len(urls):
                print(f"Download endpoint did not return a ZIP; trying fallback: {urls[index + 1]}")
        except requests.RequestException as exc:
            last_error = exc
            tmp_path.unlink(missing_ok=True)

    raise RuntimeError(
        "Dryad download failed. Open the dataset landing page shown above, "
        f"download '{ARCHIVE_NAME}' from the May 21, 2025 version, and place "
        f"it at: {ARCHIVE_PATH}"
    ) from last_error


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise RuntimeError(f"Unsafe path in ZIP archive: {member.filename}")
        zf.extractall(destination)


def validate_extracted_data() -> list[str]:
    files = [path for path in RAW_DIR.rglob("*") if path.is_file()]
    basenames = {path.name for path in files}

    missing = sorted(EXPECTED_BASENAMES - basenames)
    if missing:
        raise RuntimeError(
            "Dataset extracted, but expected files were not found: "
            + ", ".join(missing)
        )

    return sorted(str(path.relative_to(RAW_DIR)) for path in files)


def extract_archive(force: bool = False) -> None:
    if force and RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)

    if RAW_DIR.exists() and any(RAW_DIR.iterdir()) and not force:
        try:
            validate_extracted_data()
            print(f"Using existing extracted dataset: {RAW_DIR}")
            return
        except RuntimeError:
            print("Existing extracted data is incomplete; re-extracting.")
            shutil.rmtree(RAW_DIR)

    print(f"Extracting to: {RAW_DIR}")
    safe_extract_zip(ARCHIVE_PATH, RAW_DIR)

    # The Dryad API version-download endpoint returns a release bundle.  The
    # requested data archive is one level inside that bundle, whereas the
    # legacy file-stream endpoint returns it directly.
    try:
        validate_extracted_data()
    except RuntimeError:
        nested_archives = sorted(RAW_DIR.rglob("*.zip"))
        if not nested_archives:
            raise
        nested_archive = nested_archives[0]
        print(f"Extracting nested data archive: {nested_archive}")
        safe_extract_zip(nested_archive, RAW_DIR)


def write_metadata(extracted_files: list[str]) -> None:
    metadata = {
        "dataset_title": (
            "Ericsson 5G NSA network RF and throughput measurements "
            "on AERPAW network"
        ),
        "doi": DOI,
        "landing_page": LANDING_PAGE,
        "dryad_version": "2025-05-21",
        "archive_name": ARCHIVE_NAME,
        "dryad_file_id": DRYAD_FILE_ID,
        "download_url": DOWNLOAD_URL,
        "archive_sha256": sha256(ARCHIVE_PATH),
        "archive_size_bytes": ARCHIVE_PATH.stat().st_size,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "extracted_files": extracted_files,
    }

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote metadata: {METADATA_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the pinned Ericsson 5G NSA KPI dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the archive and replace existing extracted data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        download_archive(force=args.force)
        extract_archive(force=args.force)
        extracted_files = validate_extracted_data()
        write_metadata(extracted_files)
    except (RuntimeError, OSError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print()
    print("Dataset ready.")
    print(f"Raw data: {RAW_DIR}")
    print("Next step: build a timestamp-aligned KPI observation table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
