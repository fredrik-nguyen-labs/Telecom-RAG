#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rag.bootstrap import ensure_docs, ensure_kpi_data
from telecom_rag.data import load_processed_kpis
from telecom_rag.supabase_backend import (
    get_supabase_client,
    get_supabase_status,
    supabase_admin_configured,
    sync_document_chunks,
    sync_kpi_observations,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed Supabase with Telecom-RAG document chunks and KPI observations."
    )
    parser.add_argument("--skip-docs", action="store_true")
    parser.add_argument("--skip-kpis", action="store_true")
    args = parser.parse_args()

    if not supabase_admin_configured():
        raise SystemExit(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY before running this script. "
            "The service-role key is only for this trusted sync step; do not put it in "
            "the public Streamlit app."
        )

    client = get_supabase_client(admin=True)

    if not args.skip_docs:
        ok, message = ensure_docs()
        print("Documents:", message)
        if not ok:
            raise SystemExit("Could not prepare the RAG corpus.")
        count = sync_document_chunks(client=client)
        print(f"Synced {count:,} document chunks to Supabase.")

    if not args.skip_kpis:
        ok, message = ensure_kpi_data()
        print("KPI data:", message)
        if not ok:
            raise SystemExit("Could not prepare the KPI table.")
        df = load_processed_kpis()
        count = sync_kpi_observations(df, client=client)
        print(f"Synced {count:,} KPI observations to Supabase.")

    status = get_supabase_status(client)
    print("\nSupabase status:")
    for key, value in status.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
