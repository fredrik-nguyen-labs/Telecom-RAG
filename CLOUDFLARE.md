# Cloudflare Workers AI

Telecom-RAG can use Cloudflare Workers AI for hosted generation, query embeddings and reranking.

The default hosted models are:

```text
generation:  @cf/meta/llama-3.2-3b-instruct
embeddings:  @cf/baai/bge-small-en-v1.5
reranker:    @cf/baai/bge-reranker-base
```

Cloudflare's BGE-small model returns 384-dimensional embeddings, which matches the
Supabase vector schema used by this project.

## 1. Create credentials

In the Cloudflare dashboard:

1. Open **Workers AI**.
2. Choose **Use REST API**.
3. Create a Workers AI API token.
4. Copy the API token and Account ID.

If you create a custom token instead of the template, give it the Workers AI permissions
required by Cloudflare.

Keep the token private and never commit it.

## 2. Configure the environment

Linux/macOS:

```bash
export CLOUDFLARE_ACCOUNT_ID="..."
export CLOUDFLARE_API_TOKEN="..."
export CLOUDFLARE_MODEL="@cf/meta/llama-3.2-3b-instruct"
export CLOUDFLARE_EMBEDDING_MODEL="@cf/baai/bge-small-en-v1.5"
export CLOUDFLARE_RERANKER_MODEL="@cf/baai/bge-reranker-base"
export USE_CLOUDFLARE_RETRIEVAL=true
```

PowerShell:

```powershell
$env:CLOUDFLARE_ACCOUNT_ID="..."
$env:CLOUDFLARE_API_TOKEN="..."
$env:CLOUDFLARE_MODEL="@cf/meta/llama-3.2-3b-instruct"
$env:CLOUDFLARE_EMBEDDING_MODEL="@cf/baai/bge-small-en-v1.5"
$env:CLOUDFLARE_RERANKER_MODEL="@cf/baai/bge-reranker-base"
$env:USE_CLOUDFLARE_RETRIEVAL="true"
```

The generation client uses Cloudflare's OpenAI-compatible endpoint:

```text
https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1
```

Embeddings and reranking use the Workers AI model execution API.

## 3. Use Cloudflare in Notebook 02

Start Jupyter **from the same terminal where the environment variables are set**:

```bash
uv run jupyter lab
```

Notebook 02 already detects the two required variables:

```text
CLOUDFLARE_ACCOUNT_ID
CLOUDFLARE_API_TOKEN
```

When both exist, its generation evaluation selects the Cloudflare provider instead of
local Ollama.

You can verify it in the notebook cell that prints:

```text
Evaluation provider: cloudflare
Evaluation model: ...
```

The default notebook retrieval experiments remain local FAISS + BM25 + local reranking;
Cloudflare is used for the generator unless you explicitly construct the hosted Supabase
retriever.

## 4. Streamlit Community Cloud

Add the same values to **App settings → Secrets**:

```toml
CLOUDFLARE_ACCOUNT_ID = "..."
CLOUDFLARE_API_TOKEN = "..."
CLOUDFLARE_MODEL = "@cf/meta/llama-3.2-3b-instruct"
CLOUDFLARE_EMBEDDING_MODEL = "@cf/baai/bge-small-en-v1.5"
CLOUDFLARE_RERANKER_MODEL = "@cf/baai/bge-reranker-base"
USE_CLOUDFLARE_RETRIEVAL = "true"
```

See `.streamlit/secrets.toml.example`.

## 5. Failure behavior

The hosted retrieval path uses:

```text
pgvector + PostgreSQL FTS
→ RRF
→ Cloudflare BGE reranker
```

If the optional reranker times out or is temporarily unavailable, the request falls back
to the already-fused RRF ranking instead of failing the whole RAG request.
