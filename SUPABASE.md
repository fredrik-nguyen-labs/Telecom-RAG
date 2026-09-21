# Supabase setup for Telecom-RAG

Supabase is the **recommended hosted persistence/vector backend** for the deployed app.

The project keeps the local FAISS/BM25 path for notebooks and offline development, while the deployed path can use:

```text
Streamlit
   |
   | BGE query embedding
   v
Supabase Postgres
├── kpi_observations
└── document_chunks
       ├── pgvector / HNSW dense search
       └── Postgres full-text search
              |
              v
       reciprocal-rank fusion
              |
              v
     local cross-encoder reranker
              |
              v
             LLM
```

The Supabase SQL schema assumes the default embedding model:

```text
BAAI/bge-small-en-v1.5
dimension = 384
```

If you change to an embedding model with a different dimension, update the vector dimensions in the migration.

---

## 1. Create a Supabase project

Create a project at:

https://supabase.com/dashboard

After the project is ready, open its **Connect** dialog or **Settings -> API Keys**.

You need:

- Project URL
- Publishable key: `sb_publishable_...`
- Secret key: `sb_secret_...`

The publishable key is used by the deployed Streamlit app.

The secret key is used **only** from your trusted local/admin environment to seed the database. Never commit it and do not put it in a public client.

Legacy `anon` / `service_role` keys are also accepted by the code, but the current Supabase publishable/secret keys are preferred.

---

## 2. Apply the database migration

Open the Supabase **SQL Editor**.

Copy the complete contents of:

```text
supabase/migrations/20260921130000_init_telecom_rag.sql
```

into the editor and run it.

The migration creates:

### `document_chunks`

Stores:

- chunk ID
- source/document metadata
- page and section
- chunk content
- BGE embedding `vector(384)`
- embedding/chunking version
- generated Postgres full-text-search vector

Indexes:

- HNSW cosine index for pgvector
- GIN full-text index
- embedding/chunk-version index

### `kpi_observations`

Stores the processed Ericsson/AERPAW observation table, including:

- timestamp/orientation/location
- LTE/NR RSRP
- LTE/NR SINR
- CQI/MCS/RI
- throughput
- anomaly score/flag
- full row payload

### RPC functions

The migration also creates:

```text
match_document_chunks(...)
hybrid_search_document_chunks(...)
telecom_rag_status(...)
```

The hybrid function combines:

```text
pgvector semantic ranking
        +
Postgres full-text ranking
        |
        v
 reciprocal-rank fusion
```

The Python application then applies the existing cross-encoder reranker.

---

## 3. Configure your local admin environment

Create a local `.env` or export these variables.

### Linux/macOS

```bash
export SUPABASE_URL="https://YOUR_PROJECT_REF.supabase.co"
export SUPABASE_SECRET_KEY="sb_secret_..."
```

### Windows PowerShell

```powershell
$env:SUPABASE_URL="https://YOUR_PROJECT_REF.supabase.co"
$env:SUPABASE_SECRET_KEY="sb_secret_..."
```

Do **not** commit the secret key.

The repository's `.gitignore` already ignores `.env`.

---

## 4. Install dependencies

```bash
python -m pip install -r requirements-dev.txt
```

The pinned runtime dependency includes:

```text
supabase==2.31.0
```

---

## 5. Seed Supabase

From the repository root:

```bash
python scripts/sync_supabase.py
```

The command automatically:

1. prepares/downloads the document corpus if needed,
2. chunks the documents,
3. creates BGE embeddings,
4. upserts all document chunks/embeddings into Supabase,
5. prepares the processed KPI table if needed,
6. upserts the KPI observations,
7. prints the resulting database status.

Typical final output looks like:

```text
Supabase status:
  document_chunks_total: ...
  document_chunks_current: ...
  kpi_observations: ...
  embedding_model: BAAI/bge-small-en-v1.5
  chunking_version: section-aware-v2
```

You can sync only one side:

```bash
python scripts/sync_supabase.py --skip-kpis
python scripts/sync_supabase.py --skip-docs
```

Rerun the sync command whenever the corpus, chunking, embeddings, or KPI table changes.

---

## 6. Test Supabase locally

For the application runtime, set the **publishable** key:

### Linux/macOS

```bash
export USE_SUPABASE=true
export SUPABASE_URL="https://YOUR_PROJECT_REF.supabase.co"
export SUPABASE_PUBLISHABLE_KEY="sb_publishable_..."
streamlit run app.py
```

### Windows PowerShell

```powershell
$env:USE_SUPABASE="true"
$env:SUPABASE_URL="https://YOUR_PROJECT_REF.supabase.co"
$env:SUPABASE_PUBLISHABLE_KEY="sb_publishable_..."
streamlit run app.py
```

The Streamlit sidebar should show:

```text
Storage: Supabase Postgres + pgvector
Current document chunks: ...
KPI observations: ...
Vector index: ✅ HNSW
```

If Supabase is configured but not seeded or temporarily unavailable, the app falls back to the local reproducible FAISS/BM25 backend.

---

## 7. Configure Streamlit Community Cloud

In Streamlit **App settings -> Secrets**, add:

```toml
OPENAI_API_KEY = "..."
OPENAI_MODEL = "gpt-5.6-luna"

USE_SUPABASE = "true"
SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "sb_publishable_..."
```

Do **not** add `SUPABASE_SECRET_KEY` to the public Streamlit app.

The deployed application only needs read/search access.

---

## Security model

The SQL migration enables Row Level Security.

The publishable-key application can:

- read the public KPI observations,
- call the constrained dense/hybrid search RPCs,
- call the status RPC.

It cannot directly read the raw `document_chunks` table through the Data API.

The secret key maps to the elevated `service_role` and is used only by the trusted sync script.

---

## Local vs hosted retrieval

### Local/notebook

```text
BGE
 + FAISS
 + rank-bm25
 + RRF
 + cross-encoder
```

### Hosted/Supabase

```text
BGE
 + pgvector/HNSW
 + Postgres full-text search
 + RRF
 + cross-encoder
```

The LangGraph and generation layers are unchanged.

This gives the project both:

- a simple reproducible ML/RAG baseline,
- a persistent production-style hosted architecture.
