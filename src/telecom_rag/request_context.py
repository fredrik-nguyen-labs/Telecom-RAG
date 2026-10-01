from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

NUMERIC_OBSERVATION_FIELDS = frozenset(
    {
        "lte_rsrp_dbm",
        "nr_rsrp_dbm",
        "lte_sinr_db",
        "nr_sinr_db",
        "nr_cqi",
        "nr_mcs",
        "nr_ri",
        "throughput_mbps",
        "anomaly_score",
    }
)

TEXT_OBSERVATION_FIELDS = frozenset(
    {
        "observation_id",
        "observation_source",
        "timestamp",
        "orientation",
        "lte_cell_id",
        "nr_cell_id",
    }
)


def sanitize_observation(
    values: Mapping[str, Any] | None,
    *,
    default_id: str = "custom",
    default_source: str = "user-entered",
) -> dict[str, Any] | None:
    """Keep only supported KPI fields and coerce user-provided values safely."""
    if not values:
        return None

    cleaned: dict[str, Any] = {}

    for key in NUMERIC_OBSERVATION_FIELDS:
        value = values.get(key)
        if value is None or value == "":
            continue
        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            continue

    for key in TEXT_OBSERVATION_FIELDS:
        value = values.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            cleaned[key] = text[:120]

    if "anomaly_flag" in values and values.get("anomaly_flag") is not None:
        cleaned["anomaly_flag"] = bool(values["anomaly_flag"])

    measurement_keys = NUMERIC_OBSERVATION_FIELDS | {"anomaly_flag"}
    if not any(key in cleaned for key in measurement_keys):
        return None

    cleaned.setdefault("observation_id", default_id)
    cleaned.setdefault("observation_source", default_source)
    return cleaned


def conversation_question(
    question: str,
    history: Sequence[Mapping[str, Any]],
    *,
    max_messages: int = 8,
    max_chars_per_message: int = 600,
) -> str:
    """Attach a bounded recent-history window for follow-up interpretation."""
    lines: list[str] = []
    for message in history[-max_messages:]:
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        role = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {content[:max_chars_per_message]}")

    if not lines:
        return question

    return (
        "Recent conversation for follow-up context:\n"
        + "\n".join(lines)
        + f"\n\nCurrent question:\n{question}"
    )
