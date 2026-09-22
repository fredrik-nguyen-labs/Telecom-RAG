# Telecom-RAG

**Live application:** https://telecom-rag-3yyu7bhzs5ebhutyyserwy.streamlit.app/

Telecom-RAG is a 5G network diagnostics system that combines structured KPI analysis with Retrieval-Augmented Generation over telecom standards and technical documentation.

## Architecture

```text
User question
     |
Semantic router
   /      \
docs    KPI + docs
 |          |
 |     statistical analysis
 |          |
 +---- hybrid retrieval ----+
            |
        reranking
            |
     grounded answer
```

The system keeps three evidence layers separate:

- measured or user-entered KPI values,
- statistics computed from the Ericsson/AERPAW reference dataset,
- technical facts retrieved from standards and documentation.

## KPI diagnostics

For KPI-aware questions, the system computes:

- empirical KPI percentiles,
- rank correlations,
- nearest-neighbor comparisons,
- expected-vs-actual throughput,
- cross-KPI consistency,
- multivariate rarity,
- Isolation Forest anomaly scores for dataset observations.

These statistics are descriptive and dataset-relative; they are not treated as universal telecom thresholds or causal proof.

## Retrieval

### Hosted

```text
Cloudflare BGE embedding
        +
PostgreSQL full-text search
        |
       RRF
        |
Cloudflare BGE reranker
```

Hosted stack:

- Supabase Postgres + pgvector/HNSW
- PostgreSQL full-text search
- `@cf/baai/bge-small-en-v1.5`
- `@cf/baai/bge-reranker-base`
- semantic router: `@cf/zai-org/glm-4.7-flash`
- generator: `@cf/google/gemma-4-26b-a4b-it`

If reranking is unavailable, the system falls back to the fused RRF ranking.

### Local

```text
BGE + FAISS
   +
 BM25
   |
  RRF
   |
MiniLM cross-encoder
```

The local path is used for reproducible retrieval experiments and ablations.

## Data and corpus

The KPI pipeline uses the public **Ericsson 5G NSA network RF and throughput measurements on AERPAW network** dataset:

- Dryad DOI: https://doi.org/10.5061/dryad.wh70rxx06
- release: May 21, 2025
- KPIs include RSRP, SINR, CQI, MCS, RI, throughput and cell metadata.

The RAG corpus includes ETSI/3GPP NR/LTE standards, AERPAW experiment documentation, and Ericsson technical material.

See [docs/SOURCES.md](docs/SOURCES.md) for the configured sources.

## Evaluation

Retrieval is evaluated with:

- source recall / precision,
- MRR,
- evidence-term recall,
- latency.

Generation compares the same LLM with and without retrieved context and tracks citation and grounding proxies.

The benchmark contains 20 hand-auditable telecom questions spanning standards, corpus-specific facts, diagnostics and cross-source reasoning.

See [RESULTS.md](RESULTS.md) for the current benchmark results.

## Quick start

Requires Python 3.12 and `uv`.

```bash
uv sync

uv run python scripts/download_data.py
uv run python scripts/prepare_kpi_data.py
uv run python scripts/download_docs.py
uv run python scripts/build_index.py

uv run streamlit run app.py
```

For notebooks:

```bash
uv run jupyter lab
```

Run:

1. [01_data_processing_and_eda.ipynb](notebooks/01_data_processing_and_eda.ipynb)
2. [02_rag_evaluation.ipynb](notebooks/02_rag_evaluation.ipynb)

## Deployment

The hosted application uses:

- Streamlit Community Cloud,
- Supabase Postgres + pgvector,
- Cloudflare Workers AI.

Detailed setup:

- [DEPLOYMENT.md](DEPLOYMENT.md)
- [SUPABASE.md](SUPABASE.md)
- [CLOUDFLARE.md](CLOUDFLARE.md)
