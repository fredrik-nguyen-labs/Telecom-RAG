from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


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

KPI_LABELS = {
    "lte_rsrp_dbm": "LTE RSRP",
    "nr_rsrp_dbm": "NR RSRP",
    "lte_sinr_db": "LTE SINR",
    "nr_sinr_db": "NR SINR",
    "nr_cqi": "NR CQI",
    "nr_mcs": "NR MCS",
    "nr_ri": "NR RI",
    "throughput_mbps": "Downlink throughput",
}

KPI_UNITS = {
    "lte_rsrp_dbm": "dBm",
    "nr_rsrp_dbm": "dBm",
    "lte_sinr_db": "dB",
    "nr_sinr_db": "dB",
    "throughput_mbps": "Mbps",
}


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
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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


def _numeric_value(row: pd.Series, column: str) -> float | None:
    if column not in row.index:
        return None
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _numeric_series(reference_df: pd.DataFrame, column: str) -> pd.Series:
    if column not in reference_df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(reference_df[column], errors="coerce")


def _percentile(series: pd.Series, value: float) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty or pd.isna(value):
        return None
    return float((numeric <= value).mean() * 100.0)


def _percentile_rows(row: pd.Series, reference_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in DEFAULT_KPI_COLUMNS:
        value = _numeric_value(row, column)
        if value is None:
            continue
        pct = _percentile(_numeric_series(reference_df, column), value)
        rows.append(
            {
                "metric": column,
                "label": KPI_LABELS[column],
                "value": value,
                "unit": KPI_UNITS.get(column, ""),
                "percentile": pct,
            }
        )
    return rows


def _throughput_correlations(
    reference_df: pd.DataFrame,
    max_items: int = 5,
) -> list[dict[str, Any]]:
    """Rank-based (Spearman-style) KPI associations with throughput.

    Implemented as Pearson correlation of ranks so the hosted app needs only pandas/numpy.
    These are descriptive dataset associations, not causal effects.
    """
    throughput = _numeric_series(reference_df, "throughput_mbps")
    results: list[dict[str, Any]] = []

    for column in DEFAULT_KPI_COLUMNS:
        if column == "throughput_mbps":
            continue
        metric = _numeric_series(reference_df, column)
        pair = pd.DataFrame({"x": metric, "y": throughput}).dropna()
        if len(pair) < 20 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
            continue

        rho = pair["x"].rank(method="average").corr(
            pair["y"].rank(method="average"),
            method="pearson",
        )
        if pd.isna(rho):
            continue
        results.append(
            {
                "metric": column,
                "label": KPI_LABELS[column],
                "spearman_rho": float(rho),
                "n": int(len(pair)),
            }
        )

    results.sort(key=lambda item: abs(item["spearman_rho"]), reverse=True)
    return results[:max_items]


def _nearest_neighbor_analysis(
    row: pd.Series,
    reference_df: pd.DataFrame,
    max_neighbors: int = 25,
) -> dict[str, Any] | None:
    """Compare an observation with nearby measured radio conditions.

    Distance uses z-scored available radio/link KPIs and intentionally excludes throughput
    from the matching features so throughput can be compared out-of-sample rather than
    leaking into the neighbor selection.
    """
    predictor_columns = [
        column
        for column in DEFAULT_KPI_COLUMNS
        if column != "throughput_mbps"
        and _numeric_value(row, column) is not None
        and column in reference_df.columns
        and _numeric_series(reference_df, column).notna().sum() >= 20
    ]
    if not predictor_columns:
        return None

    candidate_df = reference_df.copy()
    if (
        str(row.get("observation_source", "")) != "user-entered"
        and "observation_id" in candidate_df.columns
        and "observation_id" in row.index
        and pd.notna(row.get("observation_id"))
    ):
        candidate_df = candidate_df.loc[
            candidate_df["observation_id"].astype(str)
            != str(row.get("observation_id"))
        ]

    if candidate_df.empty:
        return None

    matrix = candidate_df[predictor_columns].apply(pd.to_numeric, errors="coerce")
    medians = matrix.median(axis=0)
    scales = matrix.std(axis=0, ddof=0).replace(0, np.nan)
    valid_features = [
        column
        for column in predictor_columns
        if pd.notna(medians[column]) and pd.notna(scales[column]) and scales[column] > 0
    ]
    if not valid_features:
        return None

    matrix = matrix[valid_features].fillna(medians[valid_features])
    row_vector = np.array(
        [_numeric_value(row, column) for column in valid_features],
        dtype=float,
    )
    med = medians[valid_features].to_numpy(dtype=float)
    scale = scales[valid_features].to_numpy(dtype=float)

    z_matrix = (matrix.to_numpy(dtype=float) - med) / scale
    z_row = (row_vector - med) / scale
    distances = np.sqrt(np.mean((z_matrix - z_row) ** 2, axis=1))

    n_neighbors = min(max_neighbors, len(candidate_df))
    if n_neighbors < 3:
        return None
    neighbor_positions = np.argsort(distances)[:n_neighbors]
    neighbors = candidate_df.iloc[neighbor_positions]

    throughput = pd.to_numeric(neighbors.get("throughput_mbps"), errors="coerce").dropna()
    result: dict[str, Any] = {
        "features": valid_features,
        "feature_labels": [KPI_LABELS[column] for column in valid_features],
        "k": int(n_neighbors),
        "median_distance_z": float(np.median(distances[neighbor_positions])),
    }

    if not throughput.empty:
        q25 = float(throughput.quantile(0.25))
        median = float(throughput.median())
        q75 = float(throughput.quantile(0.75))
        result.update(
            {
                "throughput_median_mbps": median,
                "throughput_q25_mbps": q25,
                "throughput_q75_mbps": q75,
            }
        )

        actual = _numeric_value(row, "throughput_mbps")
        if actual is not None:
            result["actual_throughput_mbps"] = actual
            result["throughput_gap_mbps"] = actual - median
            result["local_throughput_percentile"] = float(
                (throughput <= actual).mean() * 100.0
            )

    return result


def _multivariate_profile(
    row: pd.Series,
    reference_df: pd.DataFrame,
) -> dict[str, Any] | None:
    """Estimate how unusual the KPI combination is using regularized Mahalanobis distance.

    The percentile is relative to the empirical reference distribution. It is a
    multivariate rarity indicator, not a telecom fault threshold or causal diagnosis.
    """
    features = [
        column
        for column in DEFAULT_KPI_COLUMNS
        if _numeric_value(row, column) is not None
        and column in reference_df.columns
        and _numeric_series(reference_df, column).notna().sum() >= 20
    ]
    if len(features) < 2:
        return None

    matrix = reference_df[features].apply(pd.to_numeric, errors="coerce")
    medians = matrix.median(axis=0)
    scales = matrix.std(axis=0, ddof=0).replace(0, np.nan)
    valid_features = [
        column
        for column in features
        if pd.notna(medians[column]) and pd.notna(scales[column]) and scales[column] > 0
    ]
    if len(valid_features) < 2:
        return None

    matrix = matrix[valid_features].fillna(medians[valid_features])
    med = medians[valid_features].to_numpy(dtype=float)
    scale = scales[valid_features].to_numpy(dtype=float)
    z_matrix = (matrix.to_numpy(dtype=float) - med) / scale

    # Regularization makes the inverse stable when radio KPIs are highly correlated.
    covariance = np.cov(z_matrix, rowvar=False)
    covariance = np.atleast_2d(covariance)
    covariance += np.eye(covariance.shape[0]) * 1e-6
    inverse_covariance = np.linalg.pinv(covariance)

    row_vector = np.array(
        [_numeric_value(row, column) for column in valid_features],
        dtype=float,
    )
    z_row = (row_vector - med) / scale

    reference_dist2 = np.einsum(
        "ij,jk,ik->i",
        z_matrix,
        inverse_covariance,
        z_matrix,
    )
    row_dist2 = float(z_row @ inverse_covariance @ z_row)
    percentile = float((reference_dist2 <= row_dist2).mean() * 100.0)

    return {
        "features": valid_features,
        "feature_labels": [KPI_LABELS[column] for column in valid_features],
        "distance": float(np.sqrt(max(row_dist2, 0.0))),
        "percentile": percentile,
    }


def analyze_observation(row: pd.Series, reference_df: pd.DataFrame) -> dict[str, Any]:
    """Build reproducible statistical evidence for one KPI observation."""
    is_user_entered = str(row.get("observation_source", "")) == "user-entered"
    percentiles = _percentile_rows(row, reference_df)
    correlations = _throughput_correlations(reference_df)
    neighbors = _nearest_neighbor_analysis(row, reference_df)
    multivariate = _multivariate_profile(row, reference_df)

    anomaly: dict[str, Any] | None = None
    anomaly_score = _numeric_value(row, "anomaly_score")
    if anomaly_score is not None and "anomaly_score" in reference_df.columns:
        score_pct = _percentile(reference_df["anomaly_score"], anomaly_score)
        anomaly = {
            "score": anomaly_score,
            "percentile": score_pct,
            "flag": bool(row.get("anomaly_flag", False)),
            "method": "Isolation Forest",
        }

    lines = [
        "Selected KPI observation (user-entered values):"
        if is_user_entered
        else "Selected network observation (measured data):"
    ]
    if "observation_id" in row and pd.notna(row.get("observation_id")):
        lines.append(f"- observation_id: {row['observation_id']}")
    if "timestamp" in row and pd.notna(row.get("timestamp")):
        lines.append(f"- timestamp: {row['timestamp']}")
    if "orientation" in row and pd.notna(row.get("orientation")):
        lines.append(f"- UAV orientation group: {row['orientation']}")

    lines.append("\nMarginal dataset comparison:")
    for item in percentiles:
        suffix = f" {item['unit']}" if item["unit"] else ""
        if item["percentile"] is None:
            lines.append(f"- {item['label']}: {item['value']:.3g}{suffix}")
        else:
            lines.append(
                f"- {item['label']}: {item['value']:.3g}{suffix} "
                f"(about the {item['percentile']:.0f}th percentile)"
            )

    if correlations:
        lines.append("\nReference-dataset relationships with throughput:")
        for item in correlations:
            lines.append(
                f"- {item['label']} vs throughput: rank correlation "
                f"rho={item['spearman_rho']:+.2f} across n={item['n']} observations."
            )
        lines.append(
            "- These are descriptive associations in this dataset, not causal effects."
        )

    if neighbors:
        feature_names = ", ".join(neighbors["feature_labels"])
        lines.append("\nLocal similar-observation comparison:")
        lines.append(
            f"- Matched the {neighbors['k']} closest reference observations using "
            f"standardized {feature_names}; throughput was not used for matching."
        )
        if "throughput_median_mbps" in neighbors:
            lines.append(
                f"- Similar observations had median throughput "
                f"{neighbors['throughput_median_mbps']:.3g} Mbps "
                f"(IQR {neighbors['throughput_q25_mbps']:.3g}–"
                f"{neighbors['throughput_q75_mbps']:.3g} Mbps)."
            )
        if "throughput_gap_mbps" in neighbors:
            gap = neighbors["throughput_gap_mbps"]
            direction = "above" if gap >= 0 else "below"
            lines.append(
                f"- This observation's throughput is {abs(gap):.3g} Mbps {direction} "
                f"the local median and around the "
                f"{neighbors['local_throughput_percentile']:.0f}th percentile among "
                f"those neighbors."
            )

    if multivariate:
        lines.append("\nMultivariate profile:")
        lines.append(
            f"- The combination of available KPI values has multivariate-distance "
            f"percentile {multivariate['percentile']:.0f} across the reference dataset "
            f"using {len(multivariate['features'])} KPIs."
        )
        lines.append(
            "- Higher percentile means the combination is farther from the dataset's "
            "typical multivariate center; this is a rarity signal, not a fault threshold."
        )

    if anomaly and anomaly.get("percentile") is not None:
        lines.append(
            f"- Existing Isolation Forest anomaly score is around the "
            f"{anomaly['percentile']:.0f}th percentile of the dataset "
            f"(flagged={anomaly['flag']})."
        )

    lines.append(
        "\nUse these statistics as dataset-relative evidence only. Do not infer causality "
        "from percentile, correlation, neighbor, or anomaly statistics alone."
    )

    return {
        "source": "user-entered" if is_user_entered else "dataset",
        "percentiles": percentiles,
        "throughput_correlations": correlations,
        "neighbors": neighbors,
        "multivariate": multivariate,
        "anomaly": anomaly,
        "context": "\n".join(lines),
    }


def summarize_observation(row: pd.Series, reference_df: pd.DataFrame) -> str:
    """Backward-compatible text summary used by older notebook code."""
    return str(analyze_observation(row, reference_df)["context"])
