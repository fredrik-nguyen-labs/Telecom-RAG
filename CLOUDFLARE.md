# Cloudflare Workers AI setup

Telecom-RAG uses **Cloudflare Workers AI** as the recommended hosted LLM for the public demo.

The default hosted model is:

```text
@cf/meta/llama-3.2-3b-instruct
```

Local development still defaults to Ollama/Qwen3 4B.

## Architecture

```text
Streamlit
   ↓
Supabase pgvector + Postgres FTS
   ↓
cross-encoder reranker
   ↓
Cloudflare Workers AI
   ↓
Llama 3.2 3B Instruct
```

Cloudflare exposes an OpenAI-compatible Chat Completions endpoint, so the project can
reuse LangChain's `ChatOpenAI` adapter with a different base URL.

## 1. Create a Cloudflare account

Create/log into a Cloudflare account and open **Workers AI** in the dashboard.

No domain or website is required for the REST API path used by this project.

## 2. Get the Account ID and API token

In **Workers AI**, choose **Use REST API**.

Copy:

```text
Account ID
```

Then create a **Workers AI API Token**. If creating a custom token, it needs Workers AI
read/edit permissions.

Keep the token private. Do not commit it to GitHub.

## 3. Test the token locally

Set:

```bash
export CLOUDFLARE_ACCOUNT_ID="YOUR_ACCOUNT_ID"
export CLOUDFLARE_API_TOKEN="YOUR_API_TOKEN"
export CLOUDFLARE_MODEL="@cf/meta/llama-3.2-3b-instruct"
```

Then run the app:

```bash
uv run streamlit run app.py
```

If those variables are present, the sidebar should show:

```text
Free hosted LLM configured
Provider: Cloudflare Workers AI
```

The app uses this OpenAI-compatible base URL internally:

```text
https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1
```

## 4. Add the credentials to Streamlit Community Cloud

Open the deployed Streamlit app's **Settings -> Secrets** and add:

```toml
CLOUDFLARE_ACCOUNT_ID = "YOUR_ACCOUNT_ID"
CLOUDFLARE_API_TOKEN = "YOUR_API_TOKEN"
CLOUDFLARE_MODEL = "@cf/meta/llama-3.2-3b-instruct"

USE_SUPABASE = "true"
SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "YOUR_PUBLISHABLE_KEY"
```

You do not need an OpenAI API key for this deployment.

## 5. Free-tier behavior

Workers AI has a daily free inference allocation. When the free allocation is exhausted,
Cloudflare rejects further free-plan inference until the quota resets.

This makes it appropriate for a low-traffic portfolio demo, but it is not an unlimited
public inference service.

The application also retains its per-browser-session request-unit guardrail to reduce
accidental usage.

## 6. Changing the model

The model is configurable without code changes:

```bash
export CLOUDFLARE_MODEL="@cf/meta/llama-3.2-1b-instruct"
```

or set the corresponding Streamlit secret.

The default 3B model is intentionally modest: retrieval and citations provide the factual
context, while the LLM mainly synthesizes the retrieved telecom evidence.

## 7. Optional fallbacks

Provider priority in the Streamlit app is:

```text
Cloudflare configured → Cloudflare Workers AI
else OpenAI configured → OpenAI
else → selectable local Ollama / Cloudflare / OpenAI
```

For the public deployment, Cloudflare + Supabase is the intended zero-OpenAI-credit path.
