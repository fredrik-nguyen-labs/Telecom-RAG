from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PROCESSED_KPI_PATH, RAW_DATA_DIR


METRIC_FILES: dict[str, tuple[str, bool]] = {
    "throughput_mbps": ("input_throughput_with_header.csv", True),
    "lte_sinr_db": ("inputf1_sinr_with_header.csv", True),
    "nr_sinr_db": ("inputf2_sinr_with_header.csv", True),
    "lte_rsrp_dbm": ("inputf1_rsrp_with_header.csv", True),
    "nr_rsrp_dbm": ("inputf2_rsrp_with_header.csv", True),
    "lte_cell_id": ("inputf1_cellid_with_header.csv", False),
    "nr_cell_id": ("inputf2_cellid_with_header.csv", False),
    "nr_cqi": ("inputf2_cqi_with_header.csv", True),
    "nr_mcs": ("inputf2_mcs_with_header.csv", True),
    "nr_ri": ("inputf2_ri_with_header.csv", True),
}

GEO_ALIASES = {
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon", "lng", "long"),
    "altitude": ("altitude", "alt", "height"),
}


def _snake(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            if len(df.columns) == 1:
                df2 = pd.read_csv(path, encoding=encoding, sep=";")
                if len(df2.columns) > 1:
                    df = df2
            return df
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Could not parse {path}. Attempts: {errors}")


def _parse_timestamp(df: pd.DataFrame) -> pd.Series:
    cols = list(df.columns)
    strong = [
        c for c in cols
        if c in {"timestamp", "datetime", "date_time", "time_date", "time_and_date"}
        or ("time" in c and "date" in c)
    ]
    for col in strong:
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.5:
            return parsed

    date_cols = [c for c in cols if "date" in c]
    time_cols = [c for c in cols if "time" in c and "date" not in c]
    for dcol in date_cols:
        for tcol in time_cols:
            parsed = pd.to_datetime(
                df[dcol].astype(str).str.strip() + " " + df[tcol].astype(str).str.strip(),
                errors="coerce",
                format="mixed",
            )
            if parsed.notna().mean() >= 0.5:
                return parsed

    for col in cols[:3]:
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.7:
            return parsed

    raise ValueError("Could not identify a timestamp column. Columns were: " + ", ".join(cols))


def _find_geo_col(columns: Iterable[str], aliases: tuple[str, ...]) -> str | None:
    columns = list(columns)
    for alias in aliases:
        exact = [c for c in columns if c == alias]
        if exact:
            return exact[0]
    for alias in aliases:
        partial = [c for c in columns if alias in c]
        if partial:
            return partial[0]
    return None


def _metric_col(df: pd.DataFrame, metric_name: str) -> str:
    tokens = [t for t in metric_name.split("_") if t not in {"lte", "nr", "db", "dbm", "mbps"}]
    excluded = {
        "timestamp", "datetime", "date", "time", "latitude", "longitude",
        "altitude", "lat", "lon", "long", "lng", "alt",
    }
    scored: list[tuple[int, str]] = []
    for col in df.columns:
        if col in excluded or any(col == x for x in excluded):
            continue
        score = sum(token in col for token in tokens)
        if score:
            scored.append((score, col))
    if scored:
        return sorted(scored, key=lambda x: (-x[0], x[1]))[0][1]

    candidate_cols = []
    for col in df.columns:
        if any(key in col for key in ("time", "date", "lat", "lon", "long", "alt")):
            continue
        candidate_cols.append(col)
    if candidate_cols:
        return candidate_cols[-1]
    raise ValueError(f"Could not identify KPI column for {metric_name}: {list(df.columns)}")


def load_metric_file(path: Path, metric_name: str, numeric: bool = True) -> pd.DataFrame:
    """Load one author-provided KPI CSV into a normalized table."""
    raw = _read_csv_flexible(path)
    raw.columns = [_snake(c) for c in raw.columns]
    out = pd.DataFrame({"timestamp": _parse_timestamp(raw)})
    for canonical, aliases in GEO_ALIASES.items():
        col = _find_geo_col(raw.columns, aliases)
        if col is not None:
            out[canonical] = pd.to_numeric(raw[col], errors="coerce")

    value_col = _metric_col(raw, metric_name)
    if numeric:
        out[metric_name] = pd.to_numeric(raw[value_col], errors="coerce")
    else:
        out[metric_name] = raw[value_col].astype("string").str.strip()
        out.loc[out[metric_name].isin(["", "nan", "None", "<NA>"]), metric_name] = pd.NA

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out.reset_index(drop=True)


def _orientation_for(path: Path, root: Path) -> str:
    try:
        rel_parts = path.relative_to(root).parts[:-1]
    except ValueError:
        rel_parts = path.parts[:-1]
    for part in reversed(rel_parts):
        match = re.search(r"yaw\s*([0-9]+)", part.lower().replace("_", ""))
        if match:
            return f"yaw{match.group(1)}"
    return path.parent.name or "unknown"


def discover_metric_files(raw_root: Path = RAW_DATA_DIR) -> dict[str, dict[str, Path]]:
    """Return {orientation: {metric_name: csv_path}} for the released data tree."""
    if not raw_root.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_root}. Run python scripts/download_data.py first."
        )
    by_orientation: dict[str, dict[str, Path]] = {}
    for metric_name, (basename, _numeric) in METRIC_FILES.items():
        matches = sorted(raw_root.rglob(basename))
        for path in matches:
            orientation = _orientation_for(path, raw_root)
            slot = by_orientation.setdefault(orientation, {})
            if metric_name in slot:
                warnings.warn(
                    f"Multiple {basename} files mapped to {orientation}; using {slot[metric_name]}"
                )
                continue
            slot[metric_name] = path

    if not by_orientation:
        raise FileNotFoundError(
            f"No documented KPI CSV files were found under {raw_root}. "
            "Check that Ericsson_Amir.zip was extracted correctly."
        )
    return by_orientation


def _median_interval_seconds(series: pd.Series) -> float | None:
    if len(series) < 3:
        return None
    diffs = series.sort_values().diff().dt.total_seconds().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None
    return float(diffs.median())


def merge_orientation(
    files: dict[str, Path],
    orientation: str,
    merge_tolerance: str | pd.Timedelta = "2s",
) -> pd.DataFrame:
    """Timestamp-align the separate KPI streams for one UAV orientation."""
    streams: dict[str, pd.DataFrame] = {}
    for metric_name, path in files.items():
        numeric = METRIC_FILES[metric_name][1]
        streams[metric_name] = load_metric_file(path, metric_name, numeric=numeric)

    if "throughput_mbps" in streams:
        base_name = "throughput_mbps"
    else:
        base_name = max(streams, key=lambda name: len(streams[name]))
        warnings.warn(f"{orientation}: throughput stream missing; using {base_name} as merge base.")

    base = streams[base_name].copy().sort_values("timestamp")
    tolerance = pd.Timedelta(merge_tolerance)
    for metric_name, stream in streams.items():
        if metric_name == base_name:
            continue
        right = stream[["timestamp", metric_name]].sort_values("timestamp")
        base = pd.merge_asof(base, right, on="timestamp", direction="nearest", tolerance=tolerance)

    base["orientation"] = orientation
    base["source_sampling_interval_s"] = _median_interval_seconds(streams[base_name]["timestamp"])
    return base


def build_kpi_table(
    raw_root: Path = RAW_DATA_DIR,
    merge_tolerance: str | pd.Timedelta = "2s",
) -> pd.DataFrame:
    """Create one clean observation table from the released separate KPI CSVs."""
    discovered = discover_metric_files(raw_root)
    frames = [
        merge_orientation(files, orientation, merge_tolerance=merge_tolerance)
        for orientation, files in sorted(discovered.items())
    ]
    df = pd.concat(frames, ignore_index=True).sort_values(["orientation", "timestamp"])
    preferred = [
        "timestamp", "orientation", "latitude", "longitude", "altitude",
        "lte_cell_id", "nr_cell_id", "lte_rsrp_dbm", "nr_rsrp_dbm",
        "lte_sinr_db", "nr_sinr_db", "nr_cqi", "nr_mcs", "nr_ri",
        "throughput_mbps", "source_sampling_interval_s",
    ]
    ordered = [c for c in preferred if c in df.columns]
    remainder = [c for c in df.columns if c not in ordered]
    df = df[ordered + remainder].reset_index(drop=True)
    df.insert(0, "observation_id", [f"obs_{i:05d}" for i in range(len(df))])

    kpi_cols = [c for c in METRIC_FILES if c in df.columns]
    if kpi_cols:
        df = df.loc[~df[kpi_cols].isna().all(axis=1)].reset_index(drop=True)
    return df


def save_kpi_table(df: pd.DataFrame, path: Path = PROCESSED_KPI_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_processed_kpis(path: Path = PROCESSED_KPI_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed KPI table not found at {path}. Run notebook 01 or scripts/prepare_kpi_data.py."
        )
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_pct": (100 * df.isna().mean()).round(2),
            "n_unique": df.nunique(dropna=True),
        }
    ).sort_values("missing_pct", ascending=False)
