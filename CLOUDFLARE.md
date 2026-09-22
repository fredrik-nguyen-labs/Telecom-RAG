# Cloudflare Workers AI

Telecom-RAG can use Cloudflare Workers AI for hosted generation, query embeddings and reranking.

The default hosted models are:

```text
router:      @cf/zai-org/glm-4.7-flash
generation:  @cf/google/gemma-4-26b-a4b-it
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
local Ollama. The hosted Streamlit app additionally uses the dedicated GLM router model;
Notebook 02 keeps one generator model by default so its retrieval/generation evaluation
remains easy to reproduce.

You can verify it in the notebook cell that prints:

```text
Evaluation provider: cloudflare
Evaluation model: ...
```

The notebook evaluates the current local FAISS dense + cross-encoder path alongside
BM25/RRF ablations. Cloudflare is used for generation when its credentials are present;
the hosted Supabase retrieval benchmark is handled separately by
`scripts/evaluate_hosted.py`.

## 4. Streamlit Community Cloud

Add the same values to **App settings → Secrets**:

```toml
CLOUDFLARE_ACCOUNT_ID = "..."
CLOUDFLARE_API_TOKEN = "..."
CLOUDFLARE_GENERATOR_MODEL = "@cf/google/gemma-4-26b-a4b-it"
CLOUDFLARE_ROUTER_MODEL = "@cf/zai-org/glm-4.7-flash"
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


## Model choices

The public app separates routing from answer generation:

- `@cf/zai-org/glm-4.7-flash` handles the tiny semantic route decision.
- `@cf/google/gemma-4-26b-a4b-it` handles the final grounded answer.

This avoids spending a large model call on a one-token routing decision while giving the
answer stage substantially more capability than the original Llama 3.2 3B setup.

For a quality-first experiment, you can override only the generator:

```bash
export CLOUDFLARE_GENERATOR_MODEL="@cf/openai/gpt-oss-120b"
```

The rest of the retrieval/router architecture stays unchanged.


## Hosted request behavior

The app uses Cloudflare's standard OpenAI-compatible chat request shape for both the Gemma generator and GLM router. This matches the last known-good pre-cleanup hosted configuration.
