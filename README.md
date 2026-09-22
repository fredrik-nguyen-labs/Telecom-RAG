# Telecom-RAG

**Live demo:** https://telecom-rag-3yyu7bhzs5ebhutyyserwy.streamlit.app/

Telecom-RAG is a 5G network diagnostics assistant that combines structured radio KPI
analysis with retrieval-augmented generation over telecom standards and technical
documentation. It supports both measurement-aware troubleshooting and general telecom
questions in one conversational interface.

## What the app does

- Accepts either a real Ericsson/AERPAW observation or editable example KPI values.
- Routes each question to either documentation-only reasoning or KPI + documentation.
- Computes deterministic dataset-relative KPI context before generation when measurements
  are relevant.
- Retrieves supporting telecom evidence and generates a concise answer with inline
  citations such as `[S1]`.
- Keeps a bounded session conversation so follow-up questions can refer to earlier turns.
- Shows the exact retrieved source chunks behind cited claims.

The public UI intentionally keeps the answer simple: one concise response followed by the
cited sources. Detailed statistics are used as model context when relevant rather than
rendered as a separate diagnostics dashboard.

## Architecture

```text
User question + optional KPI observation
                 |
          semantic router
          /             \
     docs-only        KPI + docs
                         |
                 deterministic KPI analysis
          \             /
           retrieval query
                 |
     Supabase hybrid retrieval
   pgvector + PostgreSQL FTS
                 |
                RRF
                 |
       Cloudflare BGE reranker
                 |
     retrieved technical context
                 |
       Gemma grounded answer
                 |
       answer + cited sources
```

Hosted model roles:

- router: `@cf/zai-org/glm-4.7-flash`
- generator: `@cf/google/gemma-4-26b-a4b-it`
- query embeddings: `@cf/baai/bge-small-en-v1.5`
- reranker: `@cf/baai/bge-reranker-base`

The generator and router are called through a small direct Workers AI Chat Completions
adapter. Reasoning/thinking is disabled for the interactive hosted path so the completion
budget is spent on visible output.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the detailed request flow.

## Conversation memory

Conversation memory is deliberately session-scoped.

- Recent user/assistant turns are stored in Streamlit `session_state`.
- Up to the latest eight messages are supplied as bounded follow-up context.
- **New chat** clears the conversation and restores the default starter question.
- Chat history is **not persisted to Supabase** and is not remembered in a later browser
  session or after the Streamlit session is lost.
- KPI controls are separate UI state; clearing the conversation does not intentionally
  clear the selected/entered measurement.

This keeps the demo conversational without introducing user accounts or persistent chat
storage.

## KPI context

The reference data is the public **Ericsson 5G NSA network RF and throughput measurements
on AERPAW network** dataset.

- Dryad DOI: https://doi.org/10.5061/dryad.wh70rxx06
- release: May 21, 2025
- examples of available features: LTE/NR RSRP, LTE/NR SINR, CQI, MCS, RI, throughput,
  orientation and cell metadata

For KPI-aware questions the data layer can compute:

- empirical KPI percentiles
- Spearman/rank relationships
- nearest-neighbor comparisons
- expected-vs-actual throughput context
- cross-KPI consistency
- multivariate rarity
- Isolation Forest anomaly scores for dataset observations

These are descriptive, dataset-relative signals. They are not universal quality
thresholds or causal proof.

## Retrieval

### Hosted path

```text
Cloudflare BGE query embedding
            |
Supabase pgvector/HNSW + PostgreSQL FTS
            |
   reciprocal-rank fusion
            |
 Cloudflare BGE reranking
            |
         top-k chunks
```

If the optional hosted reranker is unavailable, the system falls back to the already
fused RRF ranking instead of failing the request.

### Local/evaluation path

The reproducible local stack uses BGE embeddings with FAISS and a MiniLM cross-encoder.
BM25/RRF remains available for controlled retrieval ablations in the evaluation code even
though it is not the default local application path.

## Corpus

The configured corpus covers:

- ETSI/3GPP NR and LTE standards
- the AERPAW/Ericsson experiment
- Ericsson technical material on radio performance, beamforming, coverage and capacity

See [docs/SOURCES.md](docs/SOURCES.md) for the source inventory and provenance.

## Repository layout

```text
app.py                       Streamlit UI and session conversation
src/telecom_rag/
  graph.py                   LangGraph routing/retrieval/generation workflow
  kpi.py                     deterministic KPI analysis
  rag.py                     model clients, prompts and answer generation
  retrieval.py               local retrieval implementations
  supabase_backend.py        hosted retrieval, embeddings, reranking and sync
  data.py                    KPI processing
  documents.py               corpus loading/chunking
  evaluation.py              local evaluation utilities
scripts/
  download_data.py           fetch public KPI data
  prepare_kpi_data.py        build processed KPI table
  download_docs.py           fetch configured corpus
  build_index.py             build local vector index
  sync_supabase.py           seed hosted data
  evaluate_hosted.py         hosted retrieval/routing/generation benchmark
notebooks/
  01_data_processing_and_eda.ipynb
  02_rag_evaluation.ipynb
supabase/migrations/         hosted database schema/RPCs
eval/                        benchmark questions and generated-result directory
deploy/                      lightweight Streamlit Community Cloud entrypoint
```

## Local quick start

Requires Python 3.12 and `uv`.

```bash
uv sync

uv run python scripts/download_data.py
uv run python scripts/prepare_kpi_data.py
uv run python scripts/download_docs.py
uv run python scripts/build_index.py

uv run streamlit run app.py
```

The local application uses Ollama by default:

```bash
ollama pull qwen2.5:3b
```

Environment options are documented in [.env.example](.env.example).

## Hosted deployment

The public deployment uses:

- Streamlit Community Cloud
- Supabase Postgres + pgvector
- Cloudflare Workers AI

Setup guides:

- [DEPLOYMENT.md](DEPLOYMENT.md)
- [SUPABASE.md](SUPABASE.md)
- [CLOUDFLARE.md](CLOUDFLARE.md)

The public runtime uses only the low-privilege Supabase publishable key. The secret/admin
key is required only for trusted database synchronization and must never be exposed in
Streamlit.

## Evaluation

The repository includes separate local and hosted evaluation paths.

Retrieval metrics include source hit/recall/precision, MRR, evidence-term recall and
latency. Generation evaluation compares the same generator with and without retrieval and
tracks answer-term recall, citation validity and grounding proxies. Routing has its own
labeled benchmark.

Current recorded results and their limitations are in [RESULTS.md](RESULTS.md).

Run the local evaluation notebook:

```bash
uv run jupyter lab
```

Run the production-style hosted benchmark after configuring Supabase and Cloudflare:

```bash
uv run python scripts/evaluate_hosted.py
```

Generated evaluation artifacts under `eval/results/` are intentionally not committed.

## Testing and repository hygiene

Every push/PR runs:

- Python compilation
- notebook/JSON validation
- unit tests
- checks preventing local secrets and generated evaluation artifacts from being committed

```bash
pytest -q tests
```

## Design choices and limitations

- The assistant is a diagnostics/research demo, not a network-operations control system.
- KPI statistics describe the included reference dataset; they are not universal operator
  thresholds.
- Retrieved documentation grounds technical claims, but generated answers can still be
  imperfect and the source chunks should be inspected for important conclusions.
- Chat memory is session-only by design.
- The hosted app requires a seeded Supabase backend and valid Workers AI credentials.
- Local and hosted retrieval stacks differ, so their benchmark numbers should be reported
  separately.

## License

[MIT License](LICENSE)
