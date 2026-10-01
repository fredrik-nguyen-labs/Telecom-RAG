# Deployment

Telecom-RAG currently keeps two hosted entry points in the repository:

1. the existing Streamlit Community Cloud app, which remains live during migration;
2. the new React/Vite + FastAPI web stack.

Both hosted paths reuse the same Supabase data, LangGraph workflow and Cloudflare Workers
AI models.

## Target web architecture

```text
Browser
  |
React + Vite static frontend
Cloudflare Pages
  |
  | HTTPS /api requests
  v
FastAPI
Render
  |
  +--> LangGraph semantic routing
  +--> deterministic KPI analysis
  +--> Cloudflare BGE query embeddings
  +--> Supabase pgvector/HNSW + PostgreSQL FTS + RRF
  +--> Cloudflare BGE reranking
  +--> Cloudflare Gemma generation
```

The FastAPI service imports the existing `src/telecom_rag` package. It does not
reimplement the RAG pipeline.

The hosted API intentionally excludes local PyTorch, SentenceTransformers, FAISS and
scikit-learn. Those remain part of the local/evaluation stack only.

## Current Streamlit deployment

The existing public demo stays available at:

```text
https://telecom-rag-3yyu7bhzs5ebhutyyserwy.streamlit.app/
```

Streamlit Community Cloud continues to use:

```text
Repository:  fredrik-nguyen-labs/Telecom-RAG
Branch:      main
Main file:   deploy/app.py
Python:      3.12
```

No Streamlit files need to be removed to deploy the new site.

## FastAPI backend

The API is in `backend/app/main.py`. The Render blueprint is `render.yaml`.

Endpoints:

- `GET /api/health` — verifies Supabase/model runtime configuration;
- `GET /api/observations` — returns a bounded list of public KPI samples;
- `POST /api/chat` — runs routing, KPI analysis, retrieval and generation.

The API keeps recent conversation context client-supplied and bounded to the latest eight
messages. It does not persist chat history.

### Required backend environment variables

```text
USE_SUPABASE=true
USE_CLOUDFLARE_RETRIEVAL=true
SUPABASE_URL=...
SUPABASE_PUBLISHABLE_KEY=...
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
CORS_ORIGINS=https://YOUR_FRONTEND_HOST
API_RATE_LIMIT_PER_HOUR=30
```

Optional model overrides use the same variables as Streamlit:

```text
CLOUDFLARE_GENERATOR_MODEL=@cf/google/gemma-4-26b-a4b-it
CLOUDFLARE_ROUTER_MODEL=@cf/zai-org/glm-4.7-flash
CLOUDFLARE_EMBEDDING_MODEL=@cf/baai/bge-small-en-v1.5
CLOUDFLARE_RERANKER_MODEL=@cf/baai/bge-reranker-base
```

Never put `SUPABASE_SECRET_KEY` in the public web runtime.

The backend applies an in-memory per-IP chat limit as a basic abuse guard. This is a
portfolio-demo safeguard, not a substitute for edge rate limiting if the service becomes a
high-traffic product.

## React frontend

The frontend lives under `frontend/`.

Local development:

```bash
cd frontend
npm install
npm run dev
```

Set:

```text
VITE_API_BASE_URL=http://localhost:8000
```

For a static production deployment:

```text
Root directory: frontend
Build command:  npm install && npm run build
Output:         dist
```

Set `VITE_API_BASE_URL` to the deployed FastAPI origin before building.

The frontend contains no Cloudflare or Supabase secret credentials. It only talks to the
FastAPI API.

## Local FastAPI development

From the repository root:

```bash
python -m pip install -r backend/requirements.txt

USE_SUPABASE=true \
SUPABASE_URL=... \
SUPABASE_PUBLISHABLE_KEY=... \
CLOUDFLARE_ACCOUNT_ID=... \
CLOUDFLARE_API_TOKEN=... \
uvicorn backend.app.main:app --reload --port 8000
```

The React dev server defaults to `http://localhost:5173`, which is allowed by the
default backend CORS configuration.

## Deployment order

Use this order so no public demo is interrupted:

1. keep Streamlit live;
2. deploy FastAPI to Render and verify `/api/health`;
3. deploy the React frontend using the Render API URL;
4. set `CORS_ORIGINS` on Render to the exact frontend origin;
5. test docs-only, KPI-aware, follow-up and citation flows;
6. attach a custom domain only after those checks pass;
7. keep the Streamlit URL as a fallback until the new site has been stable.

## Verification

Test a docs-only question:

```text
What is the difference between RSRP and SINR?
```

Test a KPI-aware question:

```text
Why might this measurement have this throughput?
```

Verify:

- the first route does not use KPI analysis merely because values are present;
- the second route uses KPI + docs;
- every retrieved source is displayed;
- cited sources map to the claims they support;
- source links point to the canonical site/PDF and page when available;
- `New chat` clears conversation state but keeps the selected KPI context.

## Data synchronization

When corpus, chunking, embedding configuration or KPI data changes, rerun:

```bash
uv run python scripts/sync_supabase.py
```

The production web API reads only the low-privilege Supabase publishable key. Admin/service
credentials remain restricted to trusted synchronization tasks.

## Failure behavior

- Missing or unseeded Supabase causes `/api/health` to return 503.
- Router failure falls back to docs-only.
- Cloudflare reranker failure falls back to fused RRF candidates.
- Generation/provider failures return an API error instead of silently changing models.
- The Streamlit fallback remains independent of the new frontend/backend deployment.
