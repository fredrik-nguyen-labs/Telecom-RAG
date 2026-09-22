# Architecture

This document describes the current production and local execution paths for Telecom-RAG.

## Request lifecycle

The Streamlit app accepts an optional KPI observation plus a natural-language question.
Recent conversation context is added only for follow-up interpretation; chat history is not
stored in Supabase.

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

`st.session_state.chat_messages` stores the current browser-session thread. Up to the
latest eight messages are converted into follow-up context.

`New chat` clears the thread, pending request state, suggested-question selection and
message draft. The starter question is restored. Conversations are never written to
Supabase, so a new Streamlit/browser session starts without previous chat memory.

KPI widgets are separate state, so clearing the conversation does not intentionally reset
the selected or entered measurement.

## Hosted vs local components

| Component | Hosted | Local |
| --- | --- | --- |
| UI | Streamlit | Streamlit |
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
