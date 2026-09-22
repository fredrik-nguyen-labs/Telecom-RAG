# Cloudflare Workers AI

Telecom-RAG uses Cloudflare Workers AI for hosted routing, generation, query embeddings and
reranking.

## Models

```text
router:      @cf/zai-org/glm-4.7-flash
generation:  @cf/google/gemma-4-26b-a4b-it
embeddings:  @cf/baai/bge-small-en-v1.5
reranker:    @cf/baai/bge-reranker-base
```

BGE-small produces 384-dimensional vectors, matching the Supabase schema.

## 1. Credentials

In the Cloudflare dashboard:

1. open **Workers AI**;
2. choose the REST/API setup;
3. create a Workers AI API token;
4. copy the API token and Account ID.

Keep the token private and never commit it.

## 2. Environment

Linux/macOS:

```bash
export CLOUDFLARE_ACCOUNT_ID="..."
export CLOUDFLARE_API_TOKEN="..."
export CLOUDFLARE_GENERATOR_MODEL="@cf/google/gemma-4-26b-a4b-it"
export CLOUDFLARE_ROUTER_MODEL="@cf/zai-org/glm-4.7-flash"
export CLOUDFLARE_EMBEDDING_MODEL="@cf/baai/bge-small-en-v1.5"
export CLOUDFLARE_RERANKER_MODEL="@cf/baai/bge-reranker-base"
export USE_CLOUDFLARE_RETRIEVAL=true
```

PowerShell:

```powershell
$env:CLOUDFLARE_ACCOUNT_ID="..."
$env:CLOUDFLARE_API_TOKEN="..."
$env:CLOUDFLARE_GENERATOR_MODEL="@cf/google/gemma-4-26b-a4b-it"
$env:CLOUDFLARE_ROUTER_MODEL="@cf/zai-org/glm-4.7-flash"
$env:CLOUDFLARE_EMBEDDING_MODEL="@cf/baai/bge-small-en-v1.5"
$env:CLOUDFLARE_RERANKER_MODEL="@cf/baai/bge-reranker-base"
$env:USE_CLOUDFLARE_RETRIEVAL="true"
```

## 3. Chat Completions adapter

The hosted router and generator use the direct Workers AI Chat Completions endpoint:

```text
https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1/chat/completions
```

The small adapter in `src/telecom_rag/rag.py` sends explicit JSON rather than routing the
request through `ChatOpenAI`.

For Gemma 4 and GLM 4.7 the request includes:

```json
{
  "stream": false,
  "chat_template_kwargs": {"enable_thinking": false},
  "max_completion_tokens": 650
}
```

The router uses a much smaller completion limit because it only returns `KPI` or
`DOCS`.

Disabling thinking is important for the interactive app: an earlier client path allowed
Gemma to consume the full completion budget in hidden reasoning and return an empty visible
answer. The current adapter also surfaces `finish_reason` and usage metadata when a
completion is empty.

## 4. Embeddings and reranking

Embeddings and reranking use the Workers AI model execution API.

Hosted retrieval:

```text
Cloudflare BGE embedding
        |
Supabase pgvector + PostgreSQL FTS
        |
       RRF
        |
Cloudflare BGE reranker
```

If reranking is unavailable or times out, the application returns the already-fused RRF
ranking instead of failing the entire RAG request.

## 5. Notebook and hosted evaluation

Notebook 02 can use Cloudflare generation when the Account ID and API token are set.
Production-style hosted retrieval/routing/generation evaluation is handled by:

```bash
uv run python scripts/evaluate_hosted.py
```

That script uses the same `get_llm(provider="cloudflare")` path as the application.

## 6. Streamlit Community Cloud

Put the same values in **App settings -> Secrets**. See
`.streamlit/secrets.toml.example`.

The hosted Streamlit runtime does not require local ML model packages for generation,
embedding or reranking.
