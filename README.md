# Telecom-RAG

A telecom-focused RAG project for diagnosing 5G network observations using real network KPIs and retrieved technical documentation.

## KPI dataset

The project is pinned to:

**Ericsson 5G NSA network RF and throughput measurements on AERPAW network**  
Dryad DOI: https://doi.org/10.5061/dryad.wh70rxx06  
Dataset version: **May 21, 2025**  
Pinned archive: **Ericsson_Amir.zip**

The measurements were captured on an Ericsson 5G Non-Standalone (NSA) network during a UAV zig-zag flight on the AERPAW platform. The release contains LTE/NR radio measurements and throughput data, including RSRP, RSRQ, SINR, throughput, cell IDs, CQI, MCS, RI, position and vehicle telemetry.

The raw dataset is intentionally not committed to Git. Reproduce it with:

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py
```

The script downloads the pinned Dryad file and extracts it under:

```text
data/raw/ericsson_5g_nsa/
```

See [data/README.md](data/README.md) for the exact files we will use and what each contributes.

## Planned system

```text
Real 5G KPI data
      |
      +--> pandas / scikit-learn analysis
      |        - distributions and relationships
      |        - identify poor-performing / unusual observations
      |
User question
      |
      +--> LangGraph workflow
                |
                +--> KPI analysis when needed
                |
                +--> LangChain retriever
                         |
                         +--> embedded telecom documentation
                         +--> FAISS vector store
                |
                +--> LLM answer with citations
```

The main experiment will compare the **same LLM without RAG vs with RAG**, then evaluate retrieval and answer quality separately.

## Roadmap

1. Download and document the real Ericsson KPI dataset.
2. Build preprocessing + EDA and create a clean observation table.
3. Assemble a small high-quality telecom document corpus.
4. Build embeddings + FAISS retrieval with LangChain.
5. Add a small LangGraph workflow for KPI queries vs documentation queries.
6. Evaluate LLM-only vs RAG and tune retrieval.
7. Deploy a Streamlit demo.
