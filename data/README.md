# Data

Telecom-RAG uses the public **Ericsson 5G NSA network RF and throughput measurements on AERPAW network** dataset for KPI analysis.

- Dryad DOI: https://doi.org/10.5061/dryad.wh70rxx06
- release used: May 21, 2025
- archive: `Ericsson_Amir.zip`
- Dryad file ID: `4078259`

The dataset contains LTE/NR radio measurements, throughput, serving-cell information, and UAV telemetry. The processing pipeline aligns the separate timestamped KPI streams into a single observation table.

## Prepare the data

```bash
uv run python scripts/download_data.py
uv run python scripts/prepare_kpi_data.py
```

The processed table is written to:

```text
data/processed/kpi_observations.csv
```

Primary fields include:

- LTE / NR RSRP
- LTE / NR SINR
- NR CQI
- NR MCS
- NR RI
- downlink throughput
- serving-cell IDs
- timestamp and UAV position/orientation metadata

Raw downloads and processed outputs are excluded from Git. The downloader and preprocessing code are versioned so the dataset can be reconstructed reproducibly.
