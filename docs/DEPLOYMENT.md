# Deployment

The production web app uses Render for both the static React frontend and the FastAPI API.
The existing Streamlit Community Cloud app remains available as a fallback.

## Production architecture

```text
telecom.fnsystems.dev
        |
React + Vite static site (Render CDN)
        |
api.telecom.fnsystems.dev
        |
FastAPI (Render, Frankfurt)
        |
LangGraph
   +--> Supabase pgvector + PostgreSQL FTS + RRF
   +--> Cloudflare Workers AI
```

The API reuses the same `src/telecom_rag` core as Streamlit.

## Render services

Both services are described in the root `render.yaml`.

### API

```text
Service:       telecom-rag-api
Runtime:       Python
Region:        Frankfurt
Plan:          Free
Build:         pip install -r backend/requirements.txt
Start:         uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
Health:        /api/health
Custom domain: api.telecom.fnsystems.dev
```

Required runtime values:

```text
SUPABASE_URL
SUPABASE_PUBLISHABLE_KEY
CLOUDFLARE_ACCOUNT_ID
CLOUDFLARE_API_TOKEN
```

Never place `SUPABASE_SECRET_KEY` in the public web service.

### Frontend

```text
Service:       telecom-rag-web
Runtime:       Static
Build:         npm install --prefix frontend && npm run build --prefix frontend
Publish:       frontend/dist
Custom domain: telecom.fnsystems.dev
```

Production builds call `https://api.telecom.fnsystems.dev` by default. Local Vite
development calls `http://localhost:8000`.

## DNS

For the API:

```text
Type:   CNAME
Name:   api.telecom
Target: telecom-rag-api.onrender.com
Proxy:  DNS only
```

For the frontend:

```text
Type:   CNAME
Name:   telecom
Target: telecom-rag-web.onrender.com
Proxy:  DNS only
```

Render provisions and renews TLS after domain verification. Remove conflicting AAAA
records for these hostnames.

## Verification

Health check:

```text
https://api.telecom.fnsystems.dev/api/health
```

Functional smoke tests:

```text
What is the difference between RSRP and SINR?
Why might this measurement have this throughput?
```

Verify docs-only routing for the first question, KPI + docs for the second, valid inline
citations, canonical source links, and follow-up conversation context.

Render Free web services may cold-start after inactivity. The static frontend remains CDN
served.
