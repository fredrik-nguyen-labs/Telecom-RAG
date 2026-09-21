# Data

## Exact KPI dataset

This project uses one pinned public dataset for the KPI-analysis side:

**Ericsson 5G NSA network RF and throughput measurements on AERPAW network**

- Repository: Dryad
- DOI: https://doi.org/10.5061/dryad.wh70rxx06
- Version used: **May 21, 2025**
- Archive used: **Ericsson_Amir.zip**
- Dryad file ID: **4078259**
- Approximate archive size: **1.28 MB**
- Measurement setting: Ericsson 5G NSA network, AERPAW UAV zig-zag flight, 30 m altitude, yaw orientations 45° and 315°

Run:

```bash
python scripts/download_data.py
```

The script records the downloaded archive's SHA-256 hash in
`data/raw/ericsson_5g_nsa/_download_metadata.json`.

## What the raw data represents

The KPI data is the **network observation we want to diagnose**.

It is not the RAG knowledge base.

For example, one eventual observation might contain:

```text
timestamp
LTE RSRP
LTE SINR
NR RSRP
NR SINR
NR CQI
NR MCS
NR RI
throughput
cell ID
latitude / longitude / altitude
```

The RAG corpus will be a separate set of technical documents. The application will combine the measured KPI state with retrieved documentation to produce a grounded explanation.

## Files we will use first

The archive contains measurements for two UAV yaw orientations. The first preprocessing pass should use these author-provided CSVs from each orientation where available:

| File | Role in the project |
| --- | --- |
| `input_throughput_with_header.csv` | Observed throughput; primary performance outcome |
| `inputf1_rsrp_with_header.csv` | LTE-side received reference signal power |
| `inputf2_rsrp_with_header.csv` | NR-side received reference signal power |
| `inputf1_sinr_with_header.csv` | LTE-side signal-to-interference-plus-noise ratio |
| `inputf2_sinr_with_header.csv` | NR-side signal-to-interference-plus-noise ratio |
| `inputf1_cellid_with_header.csv` | LTE serving cell identity |
| `inputf2_cellid_with_header.csv` | NR serving cell identity |
| `inputf2_cqi_with_header.csv` | NR channel quality indicator |
| `inputf2_mcs_with_header.csv` | NR modulation and coding scheme |
| `inputf2_ri_with_header.csv` | NR rank indicator / number of spatial layers |

The files include time/date plus position information, which gives us a way to align measurements into a single observation table.

## Useful secondary files

### `Serving_cell_Params_ENDC.csv`

Contains richer LTE and NR5G-NSA serving-cell parameters, including:

- LTE: RSRP, RSRQ, RSSI, SINR, CQI, band/channel information and cell identifiers
- NR: NR5G_RSRP, NR5G_RSRQ, NR5G_SINR, NR5G band, ARFCN, bandwidth and subcarrier spacing

We should add these fields after the core timestamp alignment is working rather than making the first preprocessing step unnecessarily complex.

### `Basic_and_Other_Params.csv`

Contains LTE/NR CSI-style parameters such as MCS, RI, CQI and PMI.

### vehicle telemetry files

The `*_vehicleOut.txt` files contain location, orientation, velocity and timestamps. These are useful for later mobility/geospatial analysis but are not required for the first RAG demo.

## Planned processed table

We will turn the separate measurement files into a clean table roughly like:

```text
timestamp
orientation
latitude
longitude
altitude
lte_cell_id
nr_cell_id
lte_rsrp_dbm
nr_rsrp_dbm
lte_sinr_db
nr_sinr_db
nr_cqi
nr_mcs
nr_ri
throughput_mbps
```

The exact column set will be finalized after inspecting the downloaded CSV headers and timestamp sampling rates.

## What we will analyze

The first DS layer should answer questions such as:

- How strongly are SINR and RSRP associated with throughput?
- Which observations have unusually low throughput relative to their radio conditions?
- Are there obvious regimes/clusters of good and poor radio performance?
- Which KPI combinations are common among the worst-performing observations?

This data-derived context can then be supplied to the LLM together with retrieved technical documentation.

## Git policy

Raw downloaded data and generated processed data are ignored by Git. The repository contains the downloader and processing code so the dataset remains reproducible without checking third-party data into the repo.
