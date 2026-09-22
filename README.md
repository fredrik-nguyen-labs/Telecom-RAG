# Telecom-RAG

**Live application:** [telecom-rag-3yyu7bhzs5ebhutyyserwy.streamlit.app](https://telecom-rag-3yyu7bhzs5ebhutyyserwy.streamlit.app/)

**Telecom-RAG** is a retrieval-augmented diagnostics system for 5G radio-network measurements. It combines structured KPI analysis with standards and technical documentation so that questions about network behavior can be answered from both **measured evidence** and **retrieved domain knowledge**.

The system is built around a simple principle: numerical observations, statistical relationships, retrieved technical facts, and model-generated hypotheses should remain distinguishable throughout the reasoning pipeline.

## Overview

Telecom-RAG accepts either a measurement from the Ericsson/AERPAW 5G NSA dataset or a user-supplied set of radio KPIs. A semantic router determines whether the question requires technical-document retrieval alone or a combined KPI-analysis and document-retrieval workflow.

For KPI-aware questions, the system first computes deterministic statistical evidence from the reference dataset. Retrieved standards and technical documentation are then used to interpret those findings. The language model receives both sources of context and produces a grounded response with citations.

```text
                              User question
                                   |
                                   v
                         Semantic intent router
                          /                 \
                         /                   \
                technical-docs          KPI + technical-docs
                     |                         |
                     |                 statistical analysis
                     |                         |
                     +-----------+-------------+
                                 |
                          hybrid retrieval
                                 |
                         grounded generation
                                 |
                                 v
                    answer + evidence + citations
```

### Core capabilities

- semantic routing between documentation-only and KPI-aware workflows,
- statistical diagnostics over measured or user-entered KPI observations,
- hybrid dense + lexical retrieval with reciprocal-rank fusion,
- reranking before generation,
- source-level citations and evidence inspection,
- separate local and hosted retrieval backends,
- retrieval and generation evaluation with reproducible benchmarks.

---

## System design

### 1. KPI analysis

The structured-data branch operates on LTE/NR radio and throughput measurements and produces deterministic dataset-relative evidence.

Current analyses include:

- per-KPI empirical percentiles,
- Spearman-style rank correlations over the reference dataset,
- nearest-neighbor comparison under similar radio conditions,
- expected-vs-actual throughput relative to local neighbors,
- cross-KPI consistency checks,
- multivariate rarity using regularized distance,
- Isolation Forest anomaly scores for processed dataset observations.

Throughput is deliberately excluded from nearest-neighbor matching when throughput itself is being evaluated, preventing target leakage.

These statistics are treated as **descriptive evidence**, not universal quality thresholds or causal conclusions.

### 2. Semantic orchestration

LangGraph controls the workflow.

```text
START
  |
semantic route
  |----------------------------|
  |                            |
docs-only                   KPI + docs
  |                            |
  |                       analyze KPIs
  |                            |
  +-----------> retrieve <------+
                  |
               generate
                  |
                 END
```

Routing is performed by an LLM-based semantic classifier rather than hard-coded keyword rules. The selected observation is therefore not automatically injected into every question.

Examples:

| Question | Route |
|---|---|
| What does RSRP measure? | Technical documents |
| How are SINR and CQI related conceptually? | Technical documents |
| What stands out in these measurements? | KPI + documents |
| What relationships are visible in the supplied values? | KPI + documents |
| Is this throughput unusual given the other KPIs? | KPI + documents |

The two routes use separate generation prompts. Documentation-only answers do not request KPI evidence or diagnostic hypotheses.

### 3. Hybrid retrieval

The hosted and local implementations use the same retrieval structure but different infrastructure.

#### Hosted path

```text
query
  |
Cloudflare BGE embedding
  |
  +--------------------------------+
  |                                |
Supabase pgvector/HNSW       PostgreSQL full-text search
  |                                |
  +---------------+----------------+
                  |
                 RRF
                  |
        Cloudflare BGE reranker
                  |
             final top-k
```

Hosted components:

- embeddings: `@cf/baai/bge-small-en-v1.5`
- lexical retrieval: PostgreSQL full-text search
- fusion: reciprocal-rank fusion (RRF)
- reranker: `@cf/baai/bge-reranker-base`
- semantic router: `@cf/zai-org/glm-4.7-flash`
- generator: `@cf/google/gemma-4-26b-a4b-it`

If hosted reranking is temporarily unavailable, retrieval falls back to the already-fused RRF ranking instead of failing the complete request.

#### Local / research path

```text
BGE + FAISS dense retrieval
          +
      BM25 lexical retrieval
          |
         RRF
          |
cross-encoder/ms-marco-MiniLM-L6-v2
          |
       final top-k
```

The local path is retained for reproducible retrieval ablations and offline experimentation.

---

## Data

The structured-data pipeline is pinned to the public Dryad release:

**Ericsson 5G NSA network RF and throughput measurements on AERPAW network**  
DOI: https://doi.org/10.5061/dryad.wh70rxx06  
Release: **May 21, 2025**  
Archive: **Ericsson_Amir.zip**

The release contains measurements including:

- LTE / NR RSRP,
- LTE / NR SINR,
- CQI,
- MCS,
- rank indicator,
- throughput,
- serving-cell identifiers,
- UAV position and orientation metadata.

The raw release stores KPI streams separately. `src/telecom_rag/data.py` normalizes each stream and performs timestamp-aware alignment rather than assuming row-wise correspondence.

Download and prepare the data with:

```bash
uv run python scripts/download_data.py
uv run python scripts/prepare_kpi_data.py
```

The resulting observation table is written to:

```text
data/processed/kpi_observations.csv
```

See [data/README.md](data/README.md) for the source-file layout and processing assumptions.

---

## Technical corpus

The retrieval corpus is built from public standards, experiment documentation, and applied network-engineering material.

Configured sources include:

- ETSI / 3GPP TS 38.215 — NR physical-layer measurements,
- ETSI / 3GPP TS 38.214 — NR physical-layer procedures for data,
- ETSI / 3GPP TS 38.300 — NR and NG-RAN overall description,
- ETSI / 3GPP TS 36.214 — LTE physical-layer measurements,
- AERPAW Ericsson experiment documentation,
- Ericsson material on Massive MIMO and beamforming,
- Ericsson material on traffic, network performance and optimization,
- Ericsson Mobility Report material.

Build the local corpus with:

```bash
uv run python scripts/download_docs.py
```

Downloaded third-party documents are excluded from Git. Source definitions and acquisition logic are committed so the corpus can be reconstructed.

See [docs/SOURCES.md](docs/SOURCES.md).

---

## Evaluation

Retrieval and generation are evaluated independently.

### Retrieval

The benchmark compares:

```text
dense
vs
dense + lexical + RRF
vs
dense + lexical + RRF + reranker
```

Metrics include:

- source hit rate,
- source recall,
- source precision,
- mean reciprocal rank,
- evidence-term recall,
- retrieval latency.

### Generation

Generation compares the **same language model** with and without retrieved evidence.

```text
LLM only
vs
LLM + retrieved context
```

Tracked diagnostics include:

- required-fact recall,
- semantic similarity to reference answers,
- citation presence,
- citation-reference validity,
- citation-support similarity,
- answer-context similarity,
- retrieval and generation latency.

The benchmark contains 20 hand-auditable questions across general telecom knowledge, standards, corpus-specific facts, applied diagnostics, and cross-source reasoning.

The evaluation code intentionally treats embedding-similarity scores as proxies rather than factuality or entailment metrics.

---

## Reproducibility

### Requirements

- Python 3.12
- `uv`

Install the locked environment:

```bash
uv sync
```

The exact dependency resolution is committed in `uv.lock`.

### Prepare the full local pipeline

```bash
uv run python scripts/download_data.py
uv run python scripts/prepare_kpi_data.py
uv run python scripts/download_docs.py
uv run python scripts/build_index.py
```

Run the application:

```bash
uv run streamlit run app.py
```

Run the notebooks:

```bash
uv run jupyter lab
```

Recommended order:

1. [01_data_processing_and_eda.ipynb](notebooks/01_data_processing_and_eda.ipynb)
2. [02_rag_demo_and_evaluation.ipynb](notebooks/02_rag_demo_and_evaluation.ipynb)

Notebook 01 covers data preparation and exploratory analysis. Notebook 02 covers retrieval, orchestration, RAG generation, and evaluation.

---

## Inference backends

### Local generation

The default local generator is Qwen3 4B through Ollama:

```bash
ollama pull qwen3:4b
ollama serve
```

### Cloudflare Workers AI

Hosted inference uses:

```text
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
CLOUDFLARE_GENERATOR_MODEL=@cf/google/gemma-4-26b-a4b-it
CLOUDFLARE_ROUTER_MODEL=@cf/zai-org/glm-4.7-flash
CLOUDFLARE_EMBEDDING_MODEL=@cf/baai/bge-small-en-v1.5
CLOUDFLARE_RERANKER_MODEL=@cf/baai/bge-reranker-base
USE_CLOUDFLARE_RETRIEVAL=true
```

See [CLOUDFLARE.md](CLOUDFLARE.md) for setup details.

---

## Hosted deployment

The hosted architecture uses:

- Streamlit Community Cloud,
- Supabase Postgres + pgvector,
- Cloudflare Workers AI.

Supabase is used for:

- persisted KPI observations,
- persisted document chunks,
- pgvector/HNSW dense retrieval,
- PostgreSQL full-text retrieval,
- constrained retrieval RPCs.

The public application uses only the Supabase publishable key. The secret/admin key is required only for trusted synchronization.

Seed the hosted backend with:

```bash
uv run python scripts/sync_supabase.py
```

The lightweight Streamlit entrypoint is:

```text
deploy/app.py
```

Detailed setup:

- [DEPLOYMENT.md](DEPLOYMENT.md)
- [SUPABASE.md](SUPABASE.md)
- [CLOUDFLARE.md](CLOUDFLARE.md)

---

## Repository structure

```text
Telecom-RAG/
├── app.py
├── deploy/
│   ├── app.py
│   └── requirements.txt
├── notebooks/
│   ├── 01_data_processing_and_eda.ipynb
│   └── 02_rag_demo_and_evaluation.ipynb
├── scripts/
│   ├── build_index.py
│   ├── download_data.py
│   ├── download_docs.py
│   ├── prepare_kpi_data.py
│   └── sync_supabase.py
├── src/telecom_rag/
│   ├── bootstrap.py
│   ├── config.py
│   ├── data.py
│   ├── documents.py
│   ├── evaluation.py
│   ├── graph.py
│   ├── kpi.py
│   ├── rag.py
│   ├── retrieval.py
│   └── supabase_backend.py
├── supabase/migrations/
├── data/
├── docs/
├── eval/
├── pyproject.toml
└── uv.lock
```

---

## Engineering considerations

Several implementation choices are deliberate:

- **No fixed telecom quality thresholds.** Radio measurements are interpreted relative to the reference dataset unless a retrieved source provides an explicit threshold.
- **No causal claims from correlation alone.** Statistical associations and model-generated hypotheses are clearly separated.
- **No target leakage in local throughput comparisons.** Throughput is excluded when selecting similar observations used to benchmark throughput itself.
- **No automatic KPI contamination of general questions.** The semantic router decides whether observation analysis is relevant.
- **No full numeric KPI dump into retrieval embeddings.** Retrieval queries remain focused on the user's question and relevant radio concepts.
- **Graceful degradation of optional reranking.** Hybrid RRF results remain usable if the hosted reranker is unavailable.
- **Separate evidence layers.** Dataset statistics, retrieved documentation, and model inference remain independently inspectable.

---

## Limitations

- The KPI analysis is relative to one public measurement campaign and should not be interpreted as a universal model of cellular-network behavior.
- The system identifies statistical relationships and diagnostic signals; it does not establish causal root cause from the available measurements alone.
- Retrieval quality depends on the configured technical corpus and chunking strategy.
- Citation-support and semantic-similarity metrics are evaluation proxies, not formal entailment guarantees.
- The hosted implementation relies on external inference and database services and therefore inherits their latency and availability characteristics.

---

## Project status

The repository contains the complete data-processing, retrieval, orchestration, evaluation, and hosted-deployment paths used by the current system. Development is focused on improving routing reliability, statistical diagnostics, retrieval quality, and grounded technical reasoning while keeping each stage independently inspectable and measurable.
