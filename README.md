# Telecom RAG

**A 5G network diagnostics assistant grounded in real measurements and telecom documentation.**

[**Live web app ↗**](https://telecom.fnsystems.dev) · [Streamlit fallback](https://telecom-rag-3yyu7bhzs5ebhutyyserwy.streamlit.app/) · [Evaluation](docs/RESULTS.md) · [Sources](docs/SOURCES.md)

Telecom RAG combines network KPI interpretation with retrieval-augmented generation. Ask
about a real Ericsson/AERPAW 5G measurement, enter your own radio KPIs, or explore a
general telecom concept. Answers include inspectable technical source references.

## What it does

- **Measurement-aware answers:** analyze NR/LTE signal strength, SINR, CQI, MCS,
  throughput and other available radio measurements.
- **Semantic routing:** distinguish general documentation questions from questions
  that benefit from KPI analysis.
- **Grounded retrieval:** hybrid semantic + full-text search over standards and
  technical documents, followed by reranking.
- **Transparent evidence:** inline citations, links to original sources, and
  expandable retrieved passages.
- **Conversational context:** bounded follow-up context within the active browser
  session; chat content is not persisted in the app database.

## Technology

| Layer | Implementation |
|---|---|
| Web | React, TypeScript, Vite, Render static CDN |
| API | Python, FastAPI, Render (Frankfurt) |
| Orchestration | LangGraph, LangChain |
| Search | Supabase PostgreSQL, pgvector, FTS, RRF |
| AI | Cloudflare Workers AI (routing, generation, embeddings, reranking) |
| Reference data | Public Ericsson/AERPAW 5G measurements; ETSI/3GPP and other technical sources |

```text
Question + optional measurement
               |
       Semantic intent routing
       /                    \
   Docs only            KPI + docs
       \                    /
          Hybrid retrieval
                 |
             Reranking
                 |
         Cited AI answer
```

KPI statistics are dataset-relative descriptive evidence, not universal thresholds
or causal guarantees. The assistant is a research and portfolio demonstration, not
an automated network-management system.

## Evaluation and provenance

The repo includes deterministic KPI processing, an evaluation notebook, benchmark
question sets, and a hosted evaluation runner.

- [Evaluation results and caveats](docs/RESULTS.md)
- [Data and document provenance](docs/SOURCES.md)
- [Architecture](docs/ARCHITECTURE.md)

The hosted RAG path is distinct from local FAISS-based retrieval, so their evaluation
metrics should not be conflated.

## Get started locally

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and (for the React site) Node 22.

```bash
uv sync
cp .env.example .env
# Set required provider values in .env, never in committed files.
uv run python scripts/download_data.py
uv run python scripts/prepare_kpi_data.py
uv run python scripts/download_docs.py
uv run python scripts/build_index.py
uv run streamlit run app.py
```

For the web stack:

```bash
python -m pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

The hosted stack requires seeded Supabase data and server-side Cloudflare Workers AI
credentials; the optional local generation path supports Ollama. Details are in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Repository organization

- `frontend/` — public React web interface and static information pages
- `backend/` — FastAPI gateway and request safeguards
- `src/telecom_rag/` — reusable LangGraph, KPI and retrieval core
- `supabase/migrations/` — versioned database schema and RLS policies
- `scripts/`, `notebooks/`, `eval/` — data preparation and evaluation
- `deploy/`, `app.py` — retained Streamlit fallback
- `docs/` — deployment, source inventory, architecture and experiment results

`.env.example`, `.python-version`, `uv.lock`, CI configuration and
`render.yaml` are intentional public, non-secret reproducibility files.
Downloaded data, indexes, generated results, runtime credentials and local
environments are excluded via `.gitignore`.

## Security, privacy and limitations

The public API enforces bounded inputs, origin restrictions and request limits; the
frontend uses a restrictive production Content Security Policy. Credentials stay
server-side. Conversations are held in browser memory for the current visit and
are not stored in the app database, but question data passes through Render and
Cloudflare Workers AI for processing.

**Do not submit confidential, subscriber-identifying or proprietary network data.**

See [Privacy notice](https://telecom.fnsystems.dev/privacy.html),
[Usage terms](https://telecom.fnsystems.dev/terms.html) and [Security policy](SECURITY.md).
Infrastructure-level DDoS controls, provider retention and service availability
remain subject to the hosting platforms.

## License

MIT — see [LICENSE](LICENSE).
