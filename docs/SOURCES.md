# RAG corpus sources

The project uses a focused corpus rather than a random PDF dump. The goal is to cover:

- authoritative NR/LTE measurement definitions,
- radio/data procedures and architecture,
- the exact Ericsson/AERPAW experiment,
- applied Ericsson material on coverage, capacity, throughput and radio performance.

Run:

```bash
python scripts/download_docs.py
```

The downloader currently configures **10 sources**.

## Standards

1. **ETSI / 3GPP TS 38.215 V18.5.0 — NR physical layer measurements**  
   https://www.etsi.org/deliver/etsi_ts/138200_138299/138215/18.05.00_60/ts_138215v180500p.pdf

2. **ETSI / 3GPP TS 38.214 V18.10.0 — NR physical layer procedures for data**  
   https://www.etsi.org/deliver/etsi_ts/138200_138299/138214/18.10.00_60/ts_138214v181000p.pdf

3. **ETSI / 3GPP TS 38.300 V18.5.0 — NR and NG-RAN overall description**  
   https://www.etsi.org/deliver/etsi_ts/138300_138399/138300/18.05.00_60/ts_138300v180500p.pdf

4. **ETSI / 3GPP TS 36.214 V18.0.0 — LTE physical layer measurements**  
   https://www.etsi.org/deliver/etsi_ts/136200_136299/136214/18.00.00_60/ts_136214v180000p.pdf

## Exact experiment documentation

5. **AERPAW — Ericsson 5G NSA UAV experiment dataset description**  
   https://aerpaw.org/dataset/aerpaw-ericsson-5g-uav-experiment/

6. **AERPAW user manual — Ericsson experiment post-processing**  
   https://sites.google.com/ncsu.edu/aerpaw-user-manual/6-sample-experiments-repository/6-1-radio-software/6-1-6-ericsson-experiments/post-processing-of-experiment-logs

## Applied Ericsson performance material

7. **Ericsson Technology Review — beamforming in Massive MIMO**  
   https://www.ericsson.com/en/reports-and-papers/ericsson-technology-review/articles/beamforming-in-massive-mimo

8. **Ericsson Mobility Report — traffic patterns and network evolution**  
   https://www.ericsson.com/en/reports-and-papers/mobility-report/articles/traffic-patterns-drive-network-evolution

9. **Ericsson Mobility Report — AI and network performance optimization**  
   https://www.ericsson.com/en/reports-and-papers/mobility-report/articles/reinforcement-learning

10. **Ericsson Mobility Report, June 2025**  
    https://www.ericsson.com/49e9b6/assets/local/reports-papers/mobility-report/documents/2025/ericsson-mobility-report-june-2025.pdf

## Why the corpus was expanded

The original four-source corpus was good for definitions but weak for diagnosis. Questions such as “why might throughput be poor despite reasonable signal strength?” benefit from material about traffic, coverage/capacity, beamforming, and real-world network performance.

The retrieval system still remains intentionally small enough to inspect manually.

## Reproducibility

Downloaded third-party files are ignored by Git. The repository stores:

- this source list,
- `scripts/download_docs.py`,
- a generated local `docs/corpus/_sources.json` manifest containing successful downloads and SHA-256 hashes.

If one external site is temporarily unavailable, the downloader continues with the other sources. The app requires at least four successfully downloaded documents to build the RAG index.
