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

Production hostname:

```text
https://api.telecom.fnsystems.dev
```

The Blueprint pins the service to Frankfurt, uses the Free compute plan, sets the health
check to `/api/health`, and allows browser requests only from
`https://telecom.fnsystems.dev`.

Endpoints:

- `GET /api/health` — verifies Supabase/model runtime configuration;
- `GET /api/observations` — returns a bounded list of public KPI samples;
- `POST /api/chat` — runs routing, KPI analysis, retrieval and generation.

The API keeps recent conversation context client-supplied and bounded to the latest eight
messages. It does not persist chat history.

### Required backend environment variables

The Blueprint already defines the non-secret production settings. During initial Render
creation, provide only these four values:

```text
SUPABASE_URL=...
SUPABASE_PUBLISHABLE_KEY=...
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
```

`SUPABASE_PUBLISHABLE_KEY` is the low-privilege public/runtime key protected by RLS.
Never use `SUPABASE_SECRET_KEY` in the web service.

Create the Cloudflare credential from **Workers AI -> Use REST API -> Create a Workers AI
API Token** and copy the Account ID from the same page. The token is used only for model
inference; it does not need DNS or zone-management permissions.

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

Local development defaults to `http://localhost:8000`. You can override it with:

```text
VITE_API_BASE_URL=http://localhost:8000
```

Production builds default to `https://api.telecom.fnsystems.dev`, so Cloudflare Pages
does not require an API-base environment variable.

For Cloudflare Pages:

```text
Repository:       fredrik-nguyen-labs/Telecom-RAG
Production branch: main
Root directory:   frontend
Build command:    npm run build
Build output:     dist
Custom domain:    telecom.fnsystems.dev
```

`frontend/public/_headers` supplies basic browser security headers and
`frontend/public/_redirects` keeps the React SPA fallback working.

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
2. create the Workers AI REST token and copy the Cloudflare Account ID;
3. deploy the Render Blueprint and provide the four required runtime values;
4. verify the Render `/api/health` endpoint;
5. add `api.telecom.fnsystems.dev` as a Render custom domain;
6. in Cloudflare DNS, create a DNS-only CNAME for `api.telecom` pointing to the
   Render `.onrender.com` hostname, then verify the custom domain in Render;
7. create the Cloudflare Pages project from the same GitHub repository using
   `frontend` as the root directory and attach `telecom.fnsystems.dev`;
8. test docs-only, KPI-aware, follow-up and citation flows;
9. keep the Streamlit URL as a fallback until the new site has been stable.

Render Free web services can spin down after inactivity, so the first request after an
idle period can have a noticeable cold start. That is acceptable for this portfolio demo
but should be upgraded if the project becomes a production service.

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
