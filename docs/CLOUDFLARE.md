# Cloudflare Workers AI

Cloudflare Workers AI provides hosted routing, generation, query embeddings and reranking.

## Models

```text
router:      @cf/zai-org/glm-4.7-flash
generation:  @cf/google/gemma-4-26b-a4b-it
embeddings:  @cf/baai/bge-small-en-v1.5
reranker:    @cf/baai/bge-reranker-base
```

## Credentials

Create a Workers AI REST API token and copy the Cloudflare Account ID. Configure them only
in the server runtime:

```text
CLOUDFLARE_ACCOUNT_ID
CLOUDFLARE_API_TOKEN
```

The browser frontend never receives either value.

## Hosted inference

The router and generator use the Workers AI Chat Completions endpoint directly. The
adapter in `src/telecom_rag/rag.py` disables model thinking for the interactive path and
sets a bounded visible completion budget.

Hosted retrieval uses:

```text
Cloudflare BGE query embedding
        |
Supabase pgvector + PostgreSQL FTS
        |
       RRF
        |
Cloudflare BGE reranker
```

If reranking is unavailable, the application falls back to the fused RRF ordering.
