# How to Run Telecom-RAG

This guide shows the exact order for running the project locally.

## 1. Clone the repository

```bash
git clone https://github.com/fredrik-nguyen-labs/Telecom-RAG.git
cd Telecom-RAG
```

## 2. Create the locked Python environment

Python **3.12** is recommended.

```bash
uv sync
```

`uv.lock` records the exact resolution. The requirements files remain available
for non-uv environments.

---

## 3. Download the Ericsson/AERPAW KPI dataset

Run:

```bash
uv run python scripts/download_data.py
```

The project is pinned to:

- Dataset: **Ericsson 5G NSA network RF and throughput measurements on AERPAW network**
- Dryad DOI: https://doi.org/10.5061/dryad.wh70rxx06
- Version: **May 21, 2025**
- Archive: **Ericsson_Amir.zip**

The script downloads and extracts the dataset into:

```text
data/raw/ericsson_5g_nsa/
```

### If Dryad blocks the automatic download

Manually download `Ericsson_Amir.zip` from the Dryad DOI page and place it at:

```text
data/downloads/Ericsson_Amir.zip
```

Then rerun:

```bash
uv run python scripts/download_data.py
```

---

## 4. Process the KPI data

### Recommended: use the notebook

Start Jupyter:

```bash
uv run jupyter lab
```

Open and run:

```text
notebooks/01_data_processing_and_eda.ipynb
```

Run all cells from top to bottom.

This notebook:

1. discovers the released KPI files,
2. parses timestamps,
3. aligns the separate KPI streams by nearest timestamp,
4. checks missing values,
5. performs exploratory data analysis,
6. computes an Isolation Forest anomaly score,
7. saves the final observation table.

The resulting file is:

```text
data/processed/kpi_observations.csv
```

### Script-only alternative

You can create the same processed table without Jupyter:

```bash
uv run python scripts/prepare_kpi_data.py
```

---

## 5. Download the RAG documents

Run:

```bash
uv run python scripts/download_docs.py
```

The RAG corpus now contains **10 configured sources** covering:

1. NR physical-layer measurements (TS 38.215)
2. NR data procedures (TS 38.214)
3. NR / NG-RAN architecture (TS 38.300)
4. LTE physical-layer measurements (TS 36.214)
5. the exact AERPAW/Ericsson dataset
6. its post-processing workflow
7. Ericsson Massive MIMO / beamforming material
8. Ericsson coverage/capacity and traffic analysis
9. Ericsson network-performance optimization material
10. Ericsson Mobility Report June 2025

They are downloaded into:

```text
docs/corpus/
```

The downloaded third-party files are intentionally ignored by Git. The repository stores the source URLs and download script instead.

See:

```text
docs/SOURCES.md
```

for the exact URLs.

---

## 6. Install and start the local LLM

The default free/local setup uses **Ollama + Qwen3 4B**.

Install Ollama from:

https://ollama.com/

Then pull the model:

```bash
ollama pull qwen3:4b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

Leave Ollama running while using the notebook or Streamlit app.

No API key is needed for the local setup.

---

## 7. Run the RAG demo and evaluation notebook

Start Jupyter if it is not already running:

```bash
uv run jupyter lab
```

Open:

```text
notebooks/02_rag_demo_and_evaluation.ipynb
```

Run the cells from top to bottom.

The notebook demonstrates:

1. document loading,
2. text extraction,
3. section-aware contextual chunking,
4. BGE retrieval embeddings,
5. FAISS dense retrieval,
6. BM25 lexical retrieval,
7. reciprocal-rank fusion,
8. cross-encoder reranking,
9. dense vs hybrid vs reranked retrieval ablation,
10. LangGraph routing,
11. documentation-only and KPI-aware RAG,
12. a 20-question source-aware evaluation,
13. **same LLM without RAG vs final reranked RAG** evaluation.

The default generation evaluation runs the first 10 questions to keep local inference reasonably fast.

Change:

```python
limit=10
```

to:

```python
limit=None
```

to run all 20 evaluation questions.

---

## 8. Build the FAISS index from the command line

This is optional if Notebook 02 has already built and saved the index.

Run:

```bash
uv run python scripts/build_index.py
```

The generated BGE/FAISS index, its chunk snapshot, and a retrieval-configuration manifest are stored in:

```text
vector_store/
```

and is ignored by Git because it can be reproduced from the document corpus.

---

## 9. Run the Streamlit application

Make sure:

- the Python environment is active,
- KPI processing has been run,
- RAG documents have been downloaded,
- the FAISS index exists or can be built,
- Ollama is running.

Then run:

```bash
uv run streamlit run app.py
```

Streamlit will print a local URL, normally similar to:

```text
http://localhost:8501
```

The app lets you:

- select a real measured KPI observation,
- inspect its KPI values,
- ask a telecom question,
- turn RAG on or off,
- compare against the same LLM without RAG,
- inspect the LangGraph route,
- inspect the data-derived KPI context,
- switch between dense, hybrid, and reranked retrieval,
- inspect the clean retrieval query,
- inspect retrieved chunks, retrieval method/reranker metadata, and source citations.

---

## 10. Optional: use Cloudflare Workers AI instead of Ollama

For hosted/free inference, set:

```bash
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_API_TOKEN="your-workers-ai-token"
export CLOUDFLARE_MODEL="@cf/meta/llama-3.2-3b-instruct"
```

Then run:

```bash
uv run streamlit run app.py
```

If the Cloudflare variables are present, Streamlit automatically prefers Workers AI.
OpenAI remains an optional fallback but is not required.

See `CLOUDFLARE.md` for the full setup and Streamlit Secrets configuration.

---

## Fastest complete run

If you just want the shortest path from a fresh clone to the working demo:

```bash
uv sync

uv run python scripts/download_data.py
uv run python scripts/prepare_kpi_data.py

uv run python scripts/download_docs.py
uv run python scripts/build_index.py

ollama pull qwen3:4b
ollama serve
```

Then, in another terminal with the same virtual environment:

```bash
uv run streamlit run app.py
```

---

## Recommended first-time workflow

For learning and for understanding the project well enough to explain it in an interview, use this order:

```text
01_data_processing_and_eda.ipynb
        ↓
02_rag_demo_and_evaluation.ipynb
        ↓
Streamlit app
```

The notebooks explain the reasoning behind the implementation; the scripts and Streamlit app are the reusable application version.


---

## Public deployment

For the Streamlit Community Cloud deployment, including API billing protection, secrets, self-bootstrap behavior, and how to make the app public while keeping the GitHub repository private, see:

```text
DEPLOYMENT.md
```


## After upgrading from the original RAG baseline

If you ran the older MiniLM/four-source version, simply rerun:

```bash
uv sync
uv run python scripts/download_docs.py
uv run python scripts/build_index.py
```

The index manifest prevents the application from silently reusing the old MiniLM vectors. Streamlit/bootstrap also refreshes a legacy four-document corpus and rebuilds stale indexes automatically.


---

## Optional: Supabase hosted backend

For notebooks and pure local development, no database is required: the project still uses local FAISS/BM25.

For a persistent hosted deployment, configure Supabase. Full instructions are in `SUPABASE.md`.

The short version:

1. Create a Supabase project.
2. Run `supabase/migrations/20260921130000_init_telecom_rag.sql` in the Supabase SQL Editor.
3. In a trusted local/admin terminal set the project URL and Supabase secret key.
4. Run `uv run python scripts/sync_supabase.py`.
5. For runtime, set `USE_SUPABASE=true`, the project URL, and the Supabase publishable key, then run `uv run streamlit run app.py`.

When Supabase is seeded, the Streamlit sidebar shows `Storage: Supabase Postgres + pgvector`.

If Supabase is unavailable or not seeded, the app retains the local FAISS/BM25 fallback.
