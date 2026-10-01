# Architecture

This document describes the production, staged web and local execution paths for Telecom-RAG.

## Presentation and API layers

The existing Streamlit UI remains live while the new web stack is validated. The new React
frontend calls a thin FastAPI layer, and FastAPI imports the same `src/telecom_rag` core
used by Streamlit. There is only one routing/retrieval/generation implementation.

```text
Streamlit ------------------\
                             > shared telecom_rag core -> Supabase / Cloudflare
React -> FastAPI -----------/
```

## Request lifecycle

Both UIs accept an optional KPI observation plus a natural-language question. Recent
conversation context is added only for follow-up interpretation; chat history is not stored
in Supabase.

```text
question + optional recent chat
            |
     optional KPI context
            |
         LangGraph
            |
       semantic router
       /            \
   docs-only       KPI + docs
                       |
             deterministic KPI context
       \               /
          retrieval query
               |
      grounded generation
               |
      answer + cited sources
```

## Routing

If no KPI observation exists, the graph skips the model router and uses the docs-only
route. When an observation exists, the GLM router decides whether the question actually
requires measurement analysis.

Examples:

- `What does RSRP mean?` -> docs-only
- `Why might this measurement have low throughput?` -> KPI + docs
- `How does CQI influence MCS in general?` -> docs-only

Invalid/unavailable router output falls back to docs-only.

## KPI analysis

The KPI route calls `analyze_observation()` before retrieval. The model does not compute
these statistics itself. The data layer produces a compact textual context containing only
the relevant dataset-relative evidence. The public UI no longer renders the former full
statistical diagnostics panel.

## Retrieval query construction

The retrieval query is built from the user question and, on KPI-aware routes, useful KPI
names. Full numeric statistics remain in the generation context instead of being embedded
into the search query. This keeps retrieval focused on telecom concepts rather than raw
numbers and timestamps.

## Hosted retrieval

The hosted retriever lives in `supabase_backend.py`.

```text
question
  |
Cloudflare BGE embedding
  |
  +--> pgvector/HNSW dense search
  |
  +--> PostgreSQL full-text search
  |
Supabase reciprocal-rank fusion
  |
Cloudflare BGE reranking
  |
top-k documents
```

The public runtime calls constrained Supabase RPCs. It does not expose unrestricted
document-table reads. If Cloudflare reranking is unavailable, the already-fused RRF
ranking is used as a fail-open fallback.

## Local retrieval

The reproducible local path uses BGE embeddings with FAISS followed by a MiniLM
cross-encoder. BM25 and RRF remain available as evaluation ablations because they are part
of the recorded retrieval comparison.

## Generation

The final generator receives:

- the current question
- bounded recent conversation context when applicable
- compact KPI context on KPI-aware routes
- top-k retrieved chunks labeled `[S1]`, `[S2]`, ...

The prompt requests one concise answer with inline citations. The UI renders the answer and
the expandable cited chunks underneath it.

Hosted inference uses a direct Workers AI Chat Completions adapter in `rag.py`. Gemma and
GLM requests set `chat_template_kwargs.enable_thinking=false` and use
`max_completion_tokens`. This avoids spending the visible-output budget on hidden
reasoning.

## Conversation state

Streamlit stores the current thread in `st.session_state.chat_messages`. The React
frontend stores it in browser memory and sends only a bounded recent window to FastAPI.
FastAPI again limits follow-up context to the latest eight messages.

`New chat` clears the thread but does not intentionally reset the selected KPI context.
Conversations are never written to Supabase, so a new browser session starts without
previous chat memory.

## Hosted vs local components

| Component | Hosted | Local |
| --- | --- | --- |
| UI | Streamlit fallback + React web frontend | Streamlit or React dev server |
| Router | Cloudflare GLM | selected local model |
| Generator | Cloudflare Gemma | Ollama |
| Embeddings | Cloudflare BGE | local BGE |
| Dense store | Supabase pgvector/HNSW | FAISS |
| Lexical retrieval | PostgreSQL FTS | BM25 ablation |
| Fusion | RRF | evaluation ablation |
| Reranker | Cloudflare BGE reranker | MiniLM cross-encoder |
| KPI storage | Supabase | processed CSV |

## Failure behavior

- Missing or unseeded Supabase in hosted-lightweight mode stops the app with a
  configuration error.
- Router failure falls back to docs-only.
- Hosted reranker failure falls back to fused RRF candidates.
- Generator/provider failures are surfaced instead of silently switching models.
- Empty Cloudflare completions include finish-reason/token metadata in the raised error.

The goal is explicit degradation: optional ranking can fail open, but generation/provider
failures remain visible.
