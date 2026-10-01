# Supabase backend

Supabase stores hosted KPI observations and the RAG document index.

## Hosted retrieval

```text
Cloudflare BGE query embedding
          |
Supabase Postgres
  +--> pgvector/HNSW
  +--> PostgreSQL FTS
          |
          RRF
          |
Cloudflare BGE reranker
```

The document vectors use 384-dimensional BGE-small embeddings.

## Schema

Migrations live under `supabase/migrations/` and create:

- document chunks and KPI observation tables;
- pgvector HNSW and PostgreSQL FTS indexes;
- constrained dense/hybrid search RPCs;
- status RPCs and Row Level Security policies.

## Runtime

The public runtime uses only:

```text
USE_SUPABASE=true
SUPABASE_URL=...
SUPABASE_PUBLISHABLE_KEY=...
```

The admin/secret key is reserved for trusted synchronization via
`scripts/sync_supabase.py` and must never be exposed to the frontend or public API
configuration.

Chat history is not stored in Supabase.
