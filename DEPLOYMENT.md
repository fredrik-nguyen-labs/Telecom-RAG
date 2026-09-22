# Streamlit Community Cloud deployment

The public demo is designed for a lightweight hosted runtime:

```text
Streamlit
   |
   +--> Supabase KPI rows
   |
   +--> Cloudflare BGE query embedding
            |
            v
       Supabase hybrid retrieval
       pgvector + PostgreSQL FTS
            |
           RRF
            |
       Cloudflare BGE reranker
            |
            v
       Cloudflare LLM
```

No local PyTorch, SentenceTransformers, FAISS or scikit-learn model is loaded in the
hosted Streamlit process.

## 1. Prepare Supabase

Follow [SUPABASE.md](SUPABASE.md):

1. create the project,
2. run the SQL migration,
3. seed documents/KPI rows with `scripts/sync_supabase.py`.

## 2. Prepare Cloudflare Workers AI

Follow [CLOUDFLARE.md](CLOUDFLARE.md) and obtain:

- Account ID
- Workers AI API token

## 3. Create the Streamlit app

Use:

```text
Repository:  fredrik-nguyen-labs/Telecom-RAG
Branch:      main
Main file:   deploy/app.py
Python:      3.12
```

The `deploy/app.py` entrypoint sets the lightweight hosted mode and runs the root app.

## 4. Add secrets

In **App settings → Secrets**:

```toml
CLOUDFLARE_ACCOUNT_ID = "..."
CLOUDFLARE_API_TOKEN = "..."
CLOUDFLARE_MODEL = "@cf/meta/llama-3.2-3b-instruct"
CLOUDFLARE_EMBEDDING_MODEL = "@cf/baai/bge-small-en-v1.5"
CLOUDFLARE_RERANKER_MODEL = "@cf/baai/bge-reranker-base"
USE_CLOUDFLARE_RETRIEVAL = "true"

USE_SUPABASE = "true"
SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "..."
```

Never put the Supabase secret/admin key in the public deployment.

## 5. Verify

The sidebar should show:

```text
Hosted retrieval: Supabase + Cloudflare
Dense retrieval:   pgvector HNSW
Lexical retrieval: PostgreSQL FTS
Fusion:            RRF
Reranker:          Cloudflare BGE
```

Test both routes:

```text
What does RSRP measure?
```

should use the technical-doc route, while an observation-aware diagnostic question should
use KPI analysis plus documents.

## 6. Updates

Streamlit Community Cloud watches the configured branch. Normal pushes to `main` should
update the app automatically.

If you change the corpus, chunking, embedding model or KPI dataset, also rerun:

```bash
uv run python scripts/sync_supabase.py
```

so Supabase matches the new repository configuration.

## Hosted failure behavior

The lightweight hosted deployment requires a working seeded Supabase backend.

If Cloudflare reranking alone is slow/unavailable, retrieval falls back to the fused RRF
ranking. Provider/network failures outside that optional rerank step are shown as request
errors rather than silently changing the architecture.
