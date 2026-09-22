# Supabase backend

Supabase is the persistence and retrieval backend for the hosted Streamlit application.
The local/notebook path does not require Supabase.

## Architecture

```text
Cloudflare BGE query embedding
          |
          v
Supabase Postgres
├── KPI observations
├── pgvector/HNSW dense retrieval
└── PostgreSQL full-text retrieval
          |
          v
reciprocal-rank fusion
          |
          v
Cloudflare BGE reranker
```

The document vectors use BGE-small embeddings with 384 dimensions.

## 1. Create a project

Create a Supabase project and obtain:

- project URL
- publishable key
- secret/admin key for trusted synchronization only

The public Streamlit app uses only the publishable key.

## 2. Apply the migration

Run this file in the Supabase SQL Editor:

```text
supabase/migrations/20260921130000_init_telecom_rag.sql
```

It creates:

- `document_chunks`
- `kpi_observations`
- pgvector HNSW index
- PostgreSQL FTS GIN index
- constrained dense/hybrid search RPCs
- backend status RPC
- Row Level Security policies

## 3. Seed the database

From a trusted environment:

```bash
export SUPABASE_URL="https://YOUR_PROJECT_REF.supabase.co"
export SUPABASE_SECRET_KEY="..."

uv sync
uv run python scripts/sync_supabase.py
```

Partial syncs:

```bash
uv run python scripts/sync_supabase.py --skip-kpis
uv run python scripts/sync_supabase.py --skip-docs
```

Rerun synchronization whenever the corpus, embedding/chunking configuration or KPI table
changes.

## 4. Runtime configuration

Use the low-privilege key:

```bash
export USE_SUPABASE=true
export SUPABASE_URL="https://YOUR_PROJECT_REF.supabase.co"
export SUPABASE_PUBLISHABLE_KEY="..."
```

Configure Cloudflare as described in [CLOUDFLARE.md](CLOUDFLARE.md) for the full hosted
retrieval path.

## Security model

The public runtime can:

- read public KPI observations;
- call constrained search RPCs;
- call the backend status RPC.

It cannot use the admin synchronization path. Do not expose `SUPABASE_SECRET_KEY` in
Streamlit secrets or client-side code.

## Conversation storage

Chat history is **not stored in Supabase**. The conversation exists only in Streamlit
session state. Supabase stores the reference KPI observations and RAG document chunks, not
user conversations.

This is intentional for the public portfolio demo: it enables follow-up questions without
requiring user accounts or persistent chat storage.
