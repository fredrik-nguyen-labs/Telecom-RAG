# Deploy Telecom-RAG to Streamlit Community Cloud

The repository is prepared so a fresh Streamlit container can bootstrap the demo automatically:

1. download/process the Ericsson/AERPAW KPI data,
2. download the public RAG corpus,
3. build BGE embeddings and the FAISS index,
4. initialize BM25 lexical retrieval and a cross-encoder reranker,
5. use a hosted OpenAI model for generation.

If the KPI download is temporarily unavailable, the deployed app falls back to **documentation-only RAG** instead of crashing.

## What you need to do yourself

You only need to configure the external accounts/secrets. Do **not** commit an API key to GitHub.

---

## 1. Create a separate OpenAI API project

Use a dedicated API project for this public demo rather than your default/general project.

Suggested project name:

```text
Telecom-RAG-Demo
```

In the OpenAI API platform:

1. create/select the project,
2. enable only the model(s) you want the demo to use,
3. configure a small project spend limit,
4. make sure the spend control is configured as an **enforced/hard limit**, not only a notification threshold,
5. optionally set conservative project/model rate limits,
6. create a project-scoped API key.

A small public portfolio demo does not need a large budget. Pick a limit you are comfortable losing if the URL is abused.

The app also has local guardrails:
- max question length: 700 characters,
- max retrieved chunks: 5,
- max generated output: 650 tokens,
- 12 request units per browser session,
- LLM-only comparison costs an extra request unit.

These app limits are **not security boundaries** because a determined user can open a new session. The OpenAI project hard spend limit is the important billing protection.

---

## 2. Connect Streamlit Community Cloud to GitHub

Go to:

https://share.streamlit.io

Sign in with GitHub.

The repository is currently:

```text
fredrik-nguyen-labs/Telecom-RAG
```

Because the repository belongs to the `fredrik-nguyen-labs` organization and is private, authorize Streamlit Community Cloud to access that organization/repository when GitHub asks.

Make sure you are in the Streamlit workspace corresponding to the GitHub repository owner.

---

## 3. Create the Streamlit app

Create a new app with:

```text
Repository:     fredrik-nguyen-labs/Telecom-RAG
Branch:         main
Main file:      app.py
Python version: 3.12
```

Choose a custom subdomain if available, for example:

```text
telecom-rag
```

which would produce a URL like:

```text
https://telecom-rag.streamlit.app
```

If that subdomain is already taken, choose another short descriptive name.

---

## 4. Add Streamlit secrets

Before deploying (or from App settings -> Secrets afterward), add:

```toml
OPENAI_API_KEY = "YOUR_PROJECT_SCOPED_API_KEY"
OPENAI_MODEL = "gpt-5.6-luna"
```

Do not add quotes around the key anywhere else in the repository and do not commit a real `secrets.toml`.

The checked-in template is:

```text
.streamlit/secrets.toml.example
```

The application automatically detects an OpenAI secret and switches the public deployment to the hosted provider. Local development without the secret continues to default to Ollama.

---

## 5. Deploy

Click **Deploy**.

On a completely fresh container the app will prepare its reproducible assets automatically. The first startup is heavier because it may need to:

- download the public KPI archive,
- process the separate timestamped KPI streams,
- download up to 10 configured RAG sources,
- download the BGE embedding model,
- build the versioned FAISS index,
- download the cross-encoder reranker on the first reranked query.

Generated data, downloaded documents and FAISS files live only in the running Streamlit environment and are rebuilt after a clean container restart when necessary.

---

## 6. Make the app public

Because the GitHub repository is private, the Streamlit app may initially be private.

After deployment:

1. open the app,
2. click **Share** or open **App settings**,
3. go to the sharing/privacy setting,
4. select the option that makes the app public.

This lets you keep the source repository private while sharing the live portfolio demo.

---

## 7. Verify the deployed app

The sidebar should show:

```text
Hosted model configured
KPI table:   ✅   (or ⚠️ docs-only if Dryad was unavailable)
RAG sources: ✅
FAISS/BGE index: ✅
```

Test these two cases:

### Documentation question

```text
What do RSRP and SINR measure in a 5G NR network?
```

Expected behavior:
- route: `docs-only`,
- retrieval mode: `reranked`,
- RAG answer,
- source citations,
- retrieved chunks and retrieval metadata shown below the answer.

### KPI-aware question

Select an observation and ask:

```text
Why might this observation have this throughput, and which radio measurements are most relevant to investigate?
```

Expected behavior:
- route: `kpi+docs`,
- dataset-relative KPI context,
- retrieved technical documentation,
- cited explanation.

Then enable **Also run the LLM-only baseline** to demonstrate the controlled RAG comparison. This makes two LLM calls.

---

## 8. If deployment fails

Open **Manage app -> logs** in Streamlit.

Common causes:

### Dependency installation failure

Check that Streamlit is deploying with Python **3.12** and that it is using the root:

```text
requirements.txt
```

### `OPENAI_API_KEY is not set`

Add the key in Streamlit App settings -> Secrets and reboot the app.

### Billing / quota error

Check:
- the API project has available billing/credits,
- the API key belongs to the intended project,
- the project has not reached its configured hard spend limit,
- the selected model is enabled for that project.

### KPI data shows docs-only mode

This is not fatal. It means the automatic Dryad bootstrap failed. The RAG chatbot remains usable. Check the **Bootstrap details** expander in the sidebar for the download/preprocessing error.

### Corpus or FAISS bootstrap fails

The app stops because RAG cannot work without its knowledge base. Inspect **Deployment bootstrap details** and Streamlit logs. All exact source URLs are listed in:

```text
docs/SOURCES.md
```

---

## 9. Updating the deployed app

Streamlit Community Cloud watches the connected GitHub branch.

After future code changes are pushed to `main`, the deployed app updates automatically. Dependency changes in `requirements.txt` trigger dependency reinstallation.

---

## Recommended CV links

Once the app is public, your CV/project section can contain both:

```text
Live demo: https://<your-subdomain>.streamlit.app
GitHub:    https://github.com/fredrik-nguyen-labs/Telecom-RAG
```

If you keep the GitHub repository private, use only the live demo URL on the CV until you decide to make the repository public.
