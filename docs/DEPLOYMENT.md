# Deployment

The production web app uses Render for both the static React frontend and the FastAPI API.
The existing Streamlit Community Cloud app remains available as a fallback.

## Production architecture

```text
telecom.fnsystems.dev
        |
React + Vite static site (Render CDN)
        |
telecom-rag-api.onrender.com
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

Production builds call `https://telecom-rag-api.onrender.com` by default.
The Render frontend service also sets `VITE_API_BASE_URL` to this origin. The optional
`api.telecom.fnsystems.dev` hostname is not used by the live frontend while its custom
domain routing is unreliable. Local Vite development calls `http://localhost:8000`.

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
https://telecom-rag-api.onrender.com/api/health
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


## Security and privacy operations

The API implements an origin allowlist, per-client chat limits, bounded JSON chat bodies,
bounded Pydantic inputs, no-store response headers and generic public error messages.
The proxy-derived client address is used for rate limiting rather than trusting a
caller-supplied first `X-Forwarded-For` entry. Input prompts and KPI payloads are not
deliberately written to application logs. Render may log IPs, routes and timestamps.

The production index.html includes a restrictive Content Security Policy through the
Vite build. The version-controlled `render.yaml` also declares HTTP security headers,
including frame protections, HSTS and CSP, but **services created directly through
the Render API do not automatically import new Blueprint header settings**.

In Render > `telecom-rag-web` > Settings > **Headers**, apply the security header
names/values specified in `render.yaml` if they are not already present. For
`telecom.fnsystems.dev`, do not add `includeSubDomains` to HSTS without also
verifying all other `fnsystems.dev` hostnames. Add an SPA rewrite `/*` to
`/index.html` if future client-side routes are introduced. The current privacy
and terms pages are real static HTML files and do not depend on that rewrite.

For abuse protection beyond this portfolio demo, use an edge WAF/rate limiter with
well-defined proxy trust, shared rate-limit storage across instances, structured
security monitoring, and a provider-specific log retention and privacy review.
DNS-only routing does not activate Cloudflare's HTTP-layer WAF.

## Release checklist

1. GitHub CI: unit tests, syntax validation, frontend TypeScript/build and hygiene.
2. Verify both Render services are deployed from the latest intended `main` commit.
   Auto-deploy may need re-authorization if it stops following repository pushes.
3. Check `/api/health` on the native Render origin.
4. Check `https://telecom.fnsystems.dev/`, `/privacy.html` and `/terms.html`.
5. Verify general docs-only and measurement-aware questions, follow-up context,
   source links and retry behavior.
6. Review current Render environment variables; revoke and rotate exposed tokens.
7. Confirm database RLS and hosted credentials; never commit secrets or provider
   diagnostic output containing tokens.
