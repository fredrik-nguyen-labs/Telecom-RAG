from __future__ import annotations

SOURCE_CATALOG: dict[str, dict[str, str]] = {
    "etsi_ts_138215_v18_5_0": {
        "title": "ETSI / 3GPP TS 38.215 V18.5.0 — NR physical layer measurements",
        "url": "https://www.etsi.org/deliver/etsi_ts/138200_138299/138215/18.05.00_60/ts_138215v180500p.pdf",
    },
    "etsi_ts_138214_v18_10_0": {
        "title": "ETSI / 3GPP TS 38.214 V18.10.0 — NR physical layer procedures for data",
        "url": "https://www.etsi.org/deliver/etsi_ts/138200_138299/138214/18.10.00_60/ts_138214v181000p.pdf",
    },
    "etsi_ts_138300_v18_5_0": {
        "title": "ETSI / 3GPP TS 38.300 V18.5.0 — NR and NG-RAN overall description",
        "url": "https://www.etsi.org/deliver/etsi_ts/138300_138399/138300/18.05.00_60/ts_138300v180500p.pdf",
    },
    "etsi_ts_136214_v18_0_0": {
        "title": "ETSI / 3GPP TS 36.214 V18.0.0 — LTE physical layer measurements",
        "url": "https://www.etsi.org/deliver/etsi_ts/136200_136299/136214/18.00.00_60/ts_136214v180000p.pdf",
    },
    "aerpaw_ericsson_dataset": {
        "title": "AERPAW — Ericsson 5G NSA UAV experiment dataset description",
        "url": "https://aerpaw.org/dataset/aerpaw-ericsson-5g-uav-experiment/",
    },
    "aerpaw_post_processing": {
        "title": "AERPAW user manual — Ericsson experiment post-processing",
        "url": "https://sites.google.com/ncsu.edu/aerpaw-user-manual/6-sample-experiments-repository/6-1-radio-software/6-1-6-ericsson-experiments/post-processing-of-experiment-logs",
    },
    "ericsson_massive_mimo_beamforming": {
        "title": "Ericsson Technology Review — beamforming in Massive MIMO",
        "url": "https://www.ericsson.com/en/reports-and-papers/ericsson-technology-review/articles/beamforming-in-massive-mimo",
    },
    "ericsson_traffic_patterns": {
        "title": "Ericsson Mobility Report — traffic patterns and network evolution",
        "url": "https://www.ericsson.com/en/reports-and-papers/mobility-report/articles/traffic-patterns-drive-network-evolution",
    },
    "ericsson_ai_network_performance": {
        "title": "Ericsson Mobility Report — AI and network performance optimization",
        "url": "https://www.ericsson.com/en/reports-and-papers/mobility-report/articles/reinforcement-learning",
    },
    "ericsson_mobility_report_june_2025": {
        "title": "Ericsson Mobility Report, June 2025",
        "url": "https://www.ericsson.com/49e9b6/assets/local/reports-papers/mobility-report/documents/2025/ericsson-mobility-report-june-2025.pdf",
    },
}


def source_info(source_id: str) -> dict[str, str]:
    return SOURCE_CATALOG.get(source_id, {})
