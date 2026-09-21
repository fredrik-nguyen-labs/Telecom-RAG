#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rag.config import PROCESSED_KPI_PATH, RAW_DATA_DIR
from telecom_rag.data import build_kpi_table, data_quality_report, save_kpi_table
from telecom_rag.kpi import add_anomaly_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the timestamp-aligned KPI table.")
    parser.add_argument("--merge-tolerance", default="2s")
    args = parser.parse_args()

    print(f"Reading raw data from: {RAW_DATA_DIR}")
    df = build_kpi_table(merge_tolerance=args.merge_tolerance)
    result = add_anomaly_scores(df)
    path = save_kpi_table(result.dataframe, PROCESSED_KPI_PATH)

    print(f"Saved {len(result.dataframe):,} observations to: {path}")
    print(f"Anomaly features: {result.features}")
    print("\nHighest-missingness columns:")
    print(data_quality_report(result.dataframe).head(10).to_string())


if __name__ == "__main__":
    main()
