# Deploy Telecom-RAG to Streamlit Community Cloud

The recommended hosted architecture is:

```text
Streamlit
   ↓
BGE query embedding
   ↓
Supabase Postgres
├── KPI observations
├── pgvector/HNSW semantic search
└── Postgres full-text search
   ↓
reciprocal-rank fusion
   ↓
cross-encoder reranker
   ↓
Cloudflare Workers AI
```

The local FAISS/BM25 backend remains available as a fallback.

## 1. Configure Cloudflare Workers AI

Create or sign into a Cloudflare account and open **Workers AI -> Use REST API**.

Create a Workers AI API token and copy:

```text
CLOUDFLARE_ACCOUNT_ID
CLOUDFLARE_API_TOKEN
```

The default model is:

```text
@cf/meta/llama-3.2-3b-instruct
```

No OpenAI API key is required. See [`CLOUDFLARE.md`](CLOUDFLARE.md) for the full setup.

## 2. Create and seed Supabase

Create a Supabase project.

Then follow [`SUPABASE.md`](SUPABASE.md). The one-time setup is:

1. Open the Supabase SQL Editor.
2. Run:
   ```text
   supabase/migrations/20260921130000_init_telecom_rag.sql
   ```
3. In a trusted local/admin environment set the project URL and Supabase secret key.
4. Run:
   ```bash
   python scripts/sync_supabase.py
   ```
5. Keep the Supabase Project URL and publishable key for Streamlit.

Do **not** add the Supabase secret/admin key to the public Streamlit deployment.

## 3. Connect Streamlit Community Cloud to GitHub

Go to:

https://share.streamlit.io

Sign in with GitHub and authorize access to:

```text
fredrik-nguyen-labs/Telecom-RAG
```

Because the repository is organization-owned and private, make sure Streamlit has access to that organization/repository.

## 4. Create the app

Use:

```text
Repository:     fredrik-nguyen-labs/Telecom-RAG
Branch:         main
Main file:      app.py
Python version: 3.12
```

Choose an available Streamlit subdomain.

## 5. Add Streamlit secrets

In **App settings -> Secrets**, configure:

```toml
CLOUDFLARE_ACCOUNT_ID = "..."
CLOUDFLARE_API_TOKEN = "..."
CLOUDFLARE_MODEL = "@cf/meta/llama-3.2-3b-instruct"

USE_SUPABASE = "true"
SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "..."
```

The checked-in template is:

```text
.streamlit/secrets.toml.example
```

The public app needs only the low-privilege Supabase publishable key.

## 6. Deploy

Click **Deploy**.

When Supabase is configured and seeded, a new Streamlit container does **not** need to redownload the telecom corpus or rebuild the FAISS index. The persistent document chunks, embeddings, and KPI rows are read from Supabase.

The first retrieval request still loads the small local BGE query-embedding model and cross-encoder reranker.

If Supabase is unavailable or not seeded, the app retains the reproducible local FAISS/BM25 fallback.

## 7. Make the app public

After deployment, use the Streamlit sharing/privacy settings to make the app public if desired.

You can keep the GitHub repository private while sharing the deployed app.

## 8. Verify the deployment

The sidebar should show something similar to:

```text
Hosted LLM configured
Storage: Supabase Postgres + pgvector
Current document chunks: <non-zero>
KPI observations: <non-zero>
Vector index: ✅ HNSW
```

Test a documentation question:

```text
What do RSRP and SINR measure in a 5G NR network?
```

Then select a KPI observation and test:

```text
Why might this observation have this throughput, and which radio measurements are most relevant to investigate?
```

The default retrieval mode should be `reranked`.

## 9. Troubleshooting

### Cloudflare Workers AI error

Check that:

- the Account ID is correct,
- the API token has Workers AI permissions,
- the configured model name is valid,
- the daily free Workers AI allocation has not been exhausted,
- Cloudflare currently has inference capacity for the model.

### Supabase configured but app uses local storage

Check that:

- the SQL migration ran successfully,
- `python scripts/sync_supabase.py` completed,
- `USE_SUPABASE` is true in Streamlit secrets,
- the Supabase Project URL is correct,
- the publishable key is correct.

When the app falls back, the sidebar exposes the Supabase status/error.

### Local fallback bootstrap fails

If Supabase is unavailable, the app may try the local corpus/FAISS path. Inspect the bootstrap details and Streamlit logs.

## 10. Updating the deployed app

Streamlit Community Cloud watches the connected `main` branch.

Code changes pushed to `main` update the app automatically.

If the **corpus, chunking, embedding model, or KPI table changes**, rerun:

```bash
python scripts/sync_supabase.py
```

so the persistent database matches the new version.

## Recommended CV links

Once public:

```text
Live demo: https://<your-subdomain>.streamlit.app
GitHub:    https://github.com/fredrik-nguyen-labs/Telecom-RAG
```

If the GitHub repository stays private, use the live demo link on the CV.
