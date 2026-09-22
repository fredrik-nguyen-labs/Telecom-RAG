# Supabase backend

Supabase is the hosted persistence and retrieval backend for the public Streamlit demo.

The local/notebook path does not require Supabase.

## Architecture

```text
Cloudflare BGE query embedding
          |
          v
Supabase Postgres
├── KPI observations
├── pgvector / HNSW dense retrieval
└── PostgreSQL full-text retrieval
          |
          v
reciprocal-rank fusion
          |
          v
Cloudflare BGE reranker
```

The schema uses `BAAI/bge-small-en-v1.5` embeddings with **384 dimensions**.

## 1. Create a project

Create a Supabase project and obtain:

- Project URL
- publishable key
- secret key

The public Streamlit app uses only the publishable key.

The secret key is used only from a trusted environment when seeding the database.

## 2. Apply the migration

Open the Supabase SQL Editor and run:

```text
supabase/migrations/20260921130000_init_telecom_rag.sql
```

It creates:

- `document_chunks`
- `kpi_observations`
- a pgvector HNSW index
- a PostgreSQL full-text GIN index
- constrained dense/hybrid retrieval RPCs
- a status RPC
- Row Level Security policies

## 3. Seed the database

Set admin credentials in a trusted shell:

```bash
export SUPABASE_URL="https://YOUR_PROJECT_REF.supabase.co"
export SUPABASE_SECRET_KEY="..."
```

Then:

```bash
uv sync
uv run python scripts/sync_supabase.py
```

Useful partial syncs:

```bash
uv run python scripts/sync_supabase.py --skip-kpis
uv run python scripts/sync_supabase.py --skip-docs
```

Rerun the sync whenever the corpus, chunking/embedding configuration, or KPI table changes.

## 4. Test the hosted backend locally

Use the low-privilege runtime key:

```bash
export USE_SUPABASE=true
export SUPABASE_URL="https://YOUR_PROJECT_REF.supabase.co"
export SUPABASE_PUBLISHABLE_KEY="..."
uv run streamlit run app.py
```

For the full hosted retrieval path, also configure Cloudflare as described in
[CLOUDFLARE.md](CLOUDFLARE.md).

## Security model

The public runtime can:

- read the public KPI observations,
- call constrained document-search RPCs,
- call the backend status RPC.

It cannot directly read the raw `document_chunks` table through the public Data API.

Do not expose `SUPABASE_SECRET_KEY` in Streamlit secrets or client-side code.
