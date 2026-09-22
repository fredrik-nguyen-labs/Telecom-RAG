# Deployment

The public application is designed for a lightweight Streamlit Community Cloud runtime.

## Hosted architecture

```text
Streamlit
   |
   +--> Supabase KPI observations
   |
   +--> Cloudflare BGE query embedding
            |
       Supabase hybrid retrieval
       pgvector + PostgreSQL FTS
            |
           RRF
            |
       Cloudflare BGE reranker
            |
      Cloudflare GLM router
            |
      Cloudflare Gemma answer
```

The hosted process does not load local PyTorch, SentenceTransformers, FAISS or
scikit-learn models.

## 1. Prepare Supabase

Follow [SUPABASE.md](SUPABASE.md):

1. create the Supabase project;
2. apply `supabase/migrations/20260921130000_init_telecom_rag.sql`;
3. seed KPI rows and document chunks with `scripts/sync_supabase.py`.

## 2. Prepare Cloudflare Workers AI

Follow [CLOUDFLARE.md](CLOUDFLARE.md) and create a Workers AI API token plus Account ID.

## 3. Configure Streamlit Community Cloud

Use:

```text
Repository:  fredrik-nguyen-labs/Telecom-RAG
Branch:      main
Main file:   deploy/app.py
Python:      3.12
```

`deploy/app.py` enables hosted-lightweight mode and runs the root `app.py`.

## 4. Add Streamlit secrets

In **App settings -> Secrets**:

```toml
CLOUDFLARE_ACCOUNT_ID = "..."
CLOUDFLARE_API_TOKEN = "..."
CLOUDFLARE_GENERATOR_MODEL = "@cf/google/gemma-4-26b-a4b-it"
CLOUDFLARE_ROUTER_MODEL = "@cf/zai-org/glm-4.7-flash"
CLOUDFLARE_EMBEDDING_MODEL = "@cf/baai/bge-small-en-v1.5"
CLOUDFLARE_RERANKER_MODEL = "@cf/baai/bge-reranker-base"
USE_CLOUDFLARE_RETRIEVAL = "true"

USE_SUPABASE = "true"
SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "..."
```

Never put `SUPABASE_SECRET_KEY` in the public Streamlit deployment.

## 5. Verify the UI

The current public layout contains:

1. a highlighted usage guide;
2. a full-width scrollable conversation history;
3. KPI controls and the question composer in a two-column layout;
4. suggested questions above the message box;
5. a highlighted analysis/loading status directly below the conversation;
6. concise answers with expandable cited source chunks.

Test a docs-only question:

```text
What is the difference between RSRP and SINR?
```

Then test an observation-aware question:

```text
Why might this measurement have this throughput?
```

The first should not require KPI analysis merely because values are selected; the second
should route through KPI + docs.

## 6. Conversation behavior

Chat memory is session-only. **New chat** clears the conversation and restores the default
starter question. Conversation history is not stored in Supabase and is not available in a
new Streamlit/browser session.

## 7. Updates and data synchronization

Streamlit Community Cloud watches `main`, so normal pushes redeploy automatically.

When the corpus, chunking configuration, embedding model or KPI dataset changes, rerun:

```bash
uv run python scripts/sync_supabase.py
```

so the hosted database matches the repository configuration.

## Failure behavior

- Hosted-lightweight mode requires a working seeded Supabase backend.
- Router failures fall back to docs-only.
- Cloudflare reranker failures fall back to the fused RRF ranking.
- Cloudflare generation/provider failures are displayed rather than silently changing the
  architecture.
- The direct Workers AI client reports empty-completion finish reason and usage metadata in
  the technical exception.

The public runtime intentionally uses only the low-privilege Supabase publishable key.
