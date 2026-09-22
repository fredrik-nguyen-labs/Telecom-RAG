# Telecom-RAG

A portfolio project for **5G network diagnostics with Retrieval-Augmented Generation (RAG)**.

The project combines real Ericsson/AERPAW 5G NSA measurements with reproducible statistical analysis, hybrid document retrieval, LangGraph orchestration, and grounded LLM explanations.

## What the project does

A user can either select a real measurement from the Ericsson/AERPAW dataset or enter their own KPI values, then ask a natural-language question.

The workflow decides semantically whether the question needs:

- **technical documents only**, or
- **KPI/data analysis + technical documents**.

For KPI-aware questions, the application computes statistical evidence first and gives that evidence to the LLM alongside retrieved telecom documentation.

```text
                           User question
                                |
                                v
                      Semantic intent router
                        /               \
                       /                 \
              technical docs        KPI + technical docs
                    |                     |
                    |             statistical KPI analysis
                    |                     |
                    +----------+----------+
                               |
                         hybrid retrieval
                               |
                      grounded generation
                               |
                               v
                  answer + evidence + citations
```

The application intentionally separates:

1. **measured/user-entered observations**,
2. **statistics computed from the reference dataset**,
3. **technical facts retrieved from documentation**, and
4. **qualified hypotheses**.

That separation makes it easier to inspect which claims came from data versus documentation versus model reasoning.

---

## Data

The KPI side is pinned to the public Dryad release:

**Ericsson 5G NSA network RF and throughput measurements on AERPAW network**  
DOI: https://doi.org/10.5061/dryad.wh70rxx06  
Version: **May 21, 2025**  
Archive: **Ericsson_Amir.zip**

The experiment includes LTE/NR measurements such as:

- RSRP
- SINR
- CQI
- MCS
- rank indicator (RI)
- throughput
- cell IDs
- UAV position/orientation metadata

Download and extract the dataset with:

```bash
uv run python scripts/download_data.py
```

The processed observation table is written to:

```text
data/processed/kpi_observations.csv
```

See [data/README.md](data/README.md) for the expected source files.

---

## KPI analytics

The KPI branch does more than display raw values.

For a selected or user-entered observation, it can compute:

- per-KPI **dataset percentiles**,
- **rank correlations** across the reference dataset,
- **nearest-neighbor comparisons** using similar radio conditions,
- **expected-vs-actual throughput** relative to similar observations,
- **cross-KPI consistency** checks,
- a **multivariate rarity** score,
- existing **Isolation Forest anomaly** information for processed dataset rows.

These statistics are descriptive evidence, not universal telecom thresholds and not proof of causality.

For example, the system can distinguish between:

> throughput is low overall

and the more informative:

> throughput is also unusually low compared with measurements that have similar RSRP, SINR, CQI, MCS and RI.

The deterministic statistics are shown separately in the Streamlit UI so they can be inspected independently of the LLM answer.

---

## RAG corpus

Run:

```bash
uv run python scripts/download_docs.py
```

The configured corpus contains 10 focused public sources, including:

- ETSI / 3GPP TS 38.215 — NR physical-layer measurements
- ETSI / 3GPP TS 38.214 — NR physical-layer procedures for data
- ETSI / 3GPP TS 38.300 — NR / NG-RAN overall description
- ETSI / 3GPP TS 36.214 — LTE physical-layer measurements
- AERPAW Ericsson dataset material
- Ericsson material on beamforming, coverage/capacity and network performance
- Ericsson Mobility Report material

Downloaded third-party files are excluded from Git. Source URLs and download logic are committed for reproducibility.

See [docs/SOURCES.md](docs/SOURCES.md).

---

## Retrieval architecture

The project deliberately keeps a reproducible local retrieval path and a lightweight hosted path.

### Hosted Streamlit deployment

```text
query
  |
Cloudflare BGE query embedding
  |
  +-------------------------------+
  |                               |
Supabase pgvector/HNSW      PostgreSQL full-text search
  |                               |
  +---------------+---------------+
                  |
                 RRF
                  |
      Cloudflare BGE reranker
                  |
             final top-k
```

Hosted models:

- query embeddings: `@cf/baai/bge-small-en-v1.5`
- reranker: `@cf/baai/bge-reranker-base`
- semantic router: `@cf/zai-org/glm-4.7-flash`
- generation default: `@cf/google/gemma-4-26b-a4b-it`

If the hosted reranker is busy or times out, the application falls back to the already-fused RRF ranking instead of failing the request.

The hosted lexical retriever is **PostgreSQL full-text search, not BM25**.

### Local / notebook retrieval

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

This path is useful for local development and controlled retrieval ablations.

---

## LangGraph orchestration

The graph contains a real conditional workflow rather than always injecting the selected KPI row.

```text
START
  |
semantic route
  |---------------------------|
  |                           |
docs-only                  KPI + docs
  |                           |
  |                      analyze KPIs
  |                           |
  +----------> retrieve <------+
                 |
              generate
                 |
                END
```

The router uses semantic intent classification rather than hard-coded keyword routing.

Examples:

- `What does RSRP mean?` → technical documents only
- `How are SINR and CQI related in theory?` → technical documents only
- `What stands out in these values?` → KPI + documents
- `What relationships do you see in the supplied data?` → KPI + documents

The two routes also use separate generation prompts. Documentation-only answers do not request KPI evidence or hypotheses.

---

## Environment

Recommended: **Python 3.12**.

Install the locked environment with:

```bash
uv sync
```

The exact dependency resolution is stored in `uv.lock`.

### Local LLM

The default local development model is Qwen3 4B through Ollama.

```bash
ollama pull qwen3:4b
ollama serve
```

### Cloudflare Workers AI

For hosted inference:

```text
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
CLOUDFLARE_GENERATOR_MODEL=@cf/google/gemma-4-26b-a4b-it
CLOUDFLARE_ROUTER_MODEL=@cf/zai-org/glm-4.7-flash
CLOUDFLARE_EMBEDDING_MODEL=@cf/baai/bge-small-en-v1.5
CLOUDFLARE_RERANKER_MODEL=@cf/baai/bge-reranker-base
USE_CLOUDFLARE_RETRIEVAL=true
```

OpenAI is optional and is not required for the public demo.

---

## Run locally

Prepare data and retrieval assets:

```bash
uv run python scripts/download_data.py
uv run python scripts/prepare_kpi_data.py
uv run python scripts/download_docs.py
uv run python scripts/build_index.py
```

Launch the app:

```bash
uv run streamlit run app.py
```

Or launch Jupyter:

```bash
uv run jupyter lab
```

Run the notebooks in order:

1. [notebooks/01_data_processing_and_eda.ipynb](notebooks/01_data_processing_and_eda.ipynb)
2. [notebooks/02_rag_demo_and_evaluation.ipynb](notebooks/02_rag_demo_and_evaluation.ipynb)

The first notebook covers data preparation/EDA. The second covers retrieval, RAG, LangGraph and evaluation.

---

## Hosted deployment

The public deployment uses:

- Streamlit Community Cloud
- Supabase Postgres + pgvector
- Cloudflare Workers AI

Required Streamlit secrets:

```toml
CLOUDFLARE_ACCOUNT_ID = "..."
CLOUDFLARE_API_TOKEN = "..."
CLOUDFLARE_GENERATOR_MODEL = "@cf/google/gemma-4-26b-a4b-it"
CLOUDFLARE_ROUTER_MODEL = "@cf/zai-org/glm-4.7-flash"

USE_SUPABASE = "true"
SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "sb_publishable_..."
```

The public app does **not** need the Supabase secret/service-role key.

Seed Supabase once from a trusted environment:

```bash
uv run python scripts/sync_supabase.py
```

The lightweight Streamlit entrypoint is:

```text
deploy/app.py
```

See [DEPLOYMENT.md](DEPLOYMENT.md) and [SUPABASE.md](SUPABASE.md) for setup details.

---

## Evaluation

Retrieval and generation are evaluated separately.

The retrieval evaluation includes metrics such as:

- source hit/recall/precision
- MRR
- evidence-term recall
- latency

Generation compares the **same LLM on the same questions**:

```text
LLM only
vs
LLM + retrieved context
```

Additional diagnostics include citation validity/support proxies and answer-context similarity.

The benchmark is intentionally small and hand-auditable. It demonstrates experimental design rather than claiming to be a production telecom benchmark.

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
│   ├── download_data.py
│   ├── download_docs.py
│   ├── prepare_kpi_data.py
│   ├── build_index.py
│   └── sync_supabase.py
├── src/telecom_rag/
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

## Design principles

This project intentionally avoids:

- hard-coded universal “good/bad” radio thresholds,
- treating correlation as causality,
- silently mixing user-entered values with measured data,
- dumping the full numeric KPI summary into the retrieval query,
- unnecessary multi-agent complexity,
- failing the whole RAG request when an optional reranker is unavailable.

The goal is a small, inspectable system in which the data analysis, retrieval, orchestration and generation stages can each be evaluated independently.
