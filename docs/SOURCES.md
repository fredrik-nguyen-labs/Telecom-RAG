# RAG corpus sources

The project intentionally starts with a **small, high-quality corpus**. The point is to evaluate retrieval quality, not to inflate the document count.

Run:

```bash
python scripts/download_docs.py
```

The script downloads these public sources into `docs/corpus/`:

1. **ETSI TS 138 215 V18.5.0 / 3GPP TS 38.215 Release 18 — NR physical layer measurements**  
   https://www.etsi.org/deliver/etsi_ts/138200_138299/138215/18.05.00_60/ts_138215v180500p.pdf

2. **ETSI TS 138 214 V18.10.0 / 3GPP TS 38.214 Release 18 — NR physical layer procedures for data**  
   https://www.etsi.org/deliver/etsi_ts/138200_138299/138214/18.10.00_60/ts_138214v181000p.pdf

3. **AERPAW — Ericsson 5G NSA UAV experiment dataset description**  
   https://aerpaw.org/dataset/aerpaw-ericsson-5g-uav-experiment/

4. **AERPAW user manual — post-processing Ericsson experiment logs**  
   https://sites.google.com/ncsu.edu/aerpaw-user-manual/6-sample-experiments-repository/6-1-radio-software/6-1-6-ericsson-experiments/post-processing-of-experiment-logs

## Why these documents?

- TS 38.215 gives authoritative definitions/context for NR physical-layer measurements.
- TS 38.214 provides physical-layer data procedures relevant to CQI/MCS-style questions.
- AERPAW describes the exact experiment behind our KPI data.
- The AERPAW post-processing manual explains the real radio/PHY/traffic logs and measurement workflow.

Downloaded third-party documents are ignored by Git; the source list and downloader are committed instead.
