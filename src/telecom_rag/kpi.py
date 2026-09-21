from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_KPI_COLUMNS = [
    "lte_rsrp_dbm",
    "nr_rsrp_dbm",
    "lte_sinr_db",
    "nr_sinr_db",
    "nr_cqi",
    "nr_mcs",
    "nr_ri",
    "throughput_mbps",
]


@dataclass
class AnomalyResult:
    dataframe: pd.DataFrame
    features: list[str]


def add_anomaly_scores(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> AnomalyResult:
    """Add an unsupervised anomaly score without inventing telecom thresholds."""
    out = df.copy()
    features = [
        c
        for c in DEFAULT_KPI_COLUMNS
        if c in out.columns and pd.to_numeric(out[c], errors="coerce").notna().sum() >= 10
    ]
    if len(out) < 20 or len(features) < 2:
        out["anomaly_score"] = np.nan
        out["anomaly_flag"] = False
        return AnomalyResult(out, features)

    X = out[features].apply(pd.to_numeric, errors="coerce")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    contamination=contamination,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X)
    model: IsolationForest = pipeline.named_steps["model"]
    Xt = pipeline[:-1].transform(X)
    out["anomaly_score"] = -model.score_samples(Xt)
    out["anomaly_flag"] = model.predict(Xt) == -1
    return AnomalyResult(out, features)


def _percentile(series: pd.Series, value: float) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty or pd.isna(value):
        return None
    return float((numeric <= value).mean() * 100.0)


def summarize_observation(row: pd.Series, reference_df: pd.DataFrame) -> str:
    """Create purely data-relative KPI context for the LLM.

    This intentionally avoids hard-coded 'good/bad' radio thresholds. Statements are
    relative to the measured dataset, making the analysis reproducible and defensible.
    """
    lines = ["Selected network observation (measured data):"]
    if "observation_id" in row and pd.notna(row["observation_id"]):
        lines.append(f"- observation_id: {row['observation_id']}")
    if "timestamp" in row and pd.notna(row["timestamp"]):
        lines.append(f"- timestamp: {row['timestamp']}")
    if "orientation" in row and pd.notna(row["orientation"]):
        lines.append(f"- UAV orientation group: {row['orientation']}")

    units = {
        "lte_rsrp_dbm": "dBm",
        "nr_rsrp_dbm": "dBm",
        "lte_sinr_db": "dB",
        "nr_sinr_db": "dB",
        "throughput_mbps": "Mbps",
    }
    labels = {
        "lte_rsrp_dbm": "LTE RSRP",
        "nr_rsrp_dbm": "NR RSRP",
        "lte_sinr_db": "LTE SINR",
        "nr_sinr_db": "NR SINR",
        "nr_cqi": "NR CQI",
        "nr_mcs": "NR MCS",
        "nr_ri": "NR RI",
        "throughput_mbps": "Downlink throughput",
    }

    for col in DEFAULT_KPI_COLUMNS:
        if col not in row.index or col not in reference_df.columns:
            continue
        value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        pct = _percentile(reference_df[col], float(value))
        unit = units.get(col, "")
        suffix = f" {unit}" if unit else ""
        if pct is None:
            lines.append(f"- {labels[col]}: {value:.3g}{suffix}")
        else:
            lines.append(
                f"- {labels[col]}: {value:.3g}{suffix} "
                f"(about the {pct:.0f}th percentile of this dataset)"
            )

    if "anomaly_score" in row.index and pd.notna(row.get("anomaly_score")):
        score_pct = _percentile(reference_df["anomaly_score"], float(row["anomaly_score"]))
        if score_pct is not None:
            lines.append(
                f"- Unsupervised anomaly score is around the {score_pct:.0f}th percentile "
                "(higher means more unusual within this dataset)."
            )

    lines.append(
        "Interpret percentiles as dataset-relative evidence, not universal telecom quality thresholds."
    )
    return "\n".join(lines)
