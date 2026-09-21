# How to Run Telecom-RAG

This guide shows the exact order for running the project locally.

## 1. Clone the repository

```bash
git clone https://github.com/fredrik-nguyen-labs/Telecom-RAG.git
cd Telecom-RAG
```

## 2. Create a Python environment

Python **3.12** is recommended.

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade pip and install the exact notebook/development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes `requirements.txt` plus Jupyter.

---

## 3. Download the Ericsson/AERPAW KPI dataset

Run:

```bash
python scripts/download_data.py
```

The project is pinned to:

- Dataset: **Ericsson 5G NSA network RF and throughput measurements on AERPAW network**
- Dryad DOI: https://doi.org/10.5061/dryad.wh70rxx06
- Version: **May 21, 2025**
- Archive: **Ericsson_Amir.zip**

The script downloads and extracts the dataset into:

```text
data/raw/ericsson_5g_nsa/
```

### If Dryad blocks the automatic download

Manually download `Ericsson_Amir.zip` from the Dryad DOI page and place it at:

```text
data/downloads/Ericsson_Amir.zip
```

Then rerun:

```bash
python scripts/download_data.py
```

---

## 4. Process the KPI data

### Recommended: use the notebook

Start Jupyter:

```bash
jupyter lab
```

Open and run:

```text
notebooks/01_data_processing_and_eda.ipynb
```

Run all cells from top to bottom.

This notebook:

1. discovers the released KPI files,
2. parses timestamps,
3. aligns the separate KPI streams by nearest timestamp,
4. checks missing values,
5. performs exploratory data analysis,
6. computes an Isolation Forest anomaly score,
7. saves the final observation table.

The resulting file is:

```text
data/processed/kpi_observations.csv
```

### Script-only alternative

You can create the same processed table without Jupyter:

```bash
python scripts/prepare_kpi_data.py
```

---

## 5. Download the RAG documents

Run:

```bash
python scripts/download_docs.py
```

The starter RAG corpus currently contains **4 configured sources**:

1. ETSI / 3GPP TS 38.215 — NR physical-layer measurements
2. ETSI / 3GPP TS 38.214 — NR physical-layer procedures for data
3. AERPAW Ericsson 5G NSA dataset description
4. AERPAW Ericsson experiment post-processing documentation

They are downloaded into:

```text
docs/corpus/
```

The downloaded third-party files are intentionally ignored by Git. The repository stores the source URLs and download script instead.

See:

```text
docs/SOURCES.md
```

for the exact URLs.

---

## 6. Install and start the local LLM

The default free/local setup uses **Ollama + Qwen3 4B**.

Install Ollama from:

https://ollama.com/

Then pull the model:

```bash
ollama pull qwen3:4b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

Leave Ollama running while using the notebook or Streamlit app.

No API key is needed for the local setup.

---

## 7. Run the RAG demo and evaluation notebook

Start Jupyter if it is not already running:

```bash
jupyter lab
```

Open:

```text
notebooks/02_rag_demo_and_evaluation.ipynb
```

Run the cells from top to bottom.

The notebook demonstrates:

1. document loading,
2. text extraction,
3. chunking,
4. SentenceTransformer embeddings,
5. FAISS indexing,
6. semantic retrieval,
7. LangGraph routing,
8. documentation-only RAG,
9. KPI-aware RAG,
10. retrieval Hit@k evaluation,
11. **same LLM without RAG vs with RAG** evaluation.

The default evaluation runs only the first 5 questions to keep local inference reasonably fast.

Change:

```python
limit=5
```

to:

```python
limit=None
```

to run the complete evaluation set.

---

## 8. Build the FAISS index from the command line

This is optional if Notebook 02 has already built and saved the index.

Run:

```bash
python scripts/build_index.py
```

The generated index is stored in:

```text
vector_store/
```

and is ignored by Git because it can be reproduced from the document corpus.

---

## 9. Run the Streamlit application

Make sure:

- the Python environment is active,
- KPI processing has been run,
- RAG documents have been downloaded,
- the FAISS index exists or can be built,
- Ollama is running.

Then run:

```bash
streamlit run app.py
```

Streamlit will print a local URL, normally similar to:

```text
http://localhost:8501
```

The app lets you:

- select a real measured KPI observation,
- inspect its KPI values,
- ask a telecom question,
- turn RAG on or off,
- compare against the same LLM without RAG,
- inspect the LangGraph route,
- inspect the data-derived KPI context,
- inspect retrieved chunks and source citations.

---

## 10. Optional: use OpenAI instead of Ollama

The code supports a hosted OpenAI model through the same LangChain interface.

Set:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-5.6-luna"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_MODEL="gpt-5.6-luna"
```

Then select `openai` as the provider in Streamlit or change the provider in Notebook 02.

For Streamlit Community Cloud, put the values in **Streamlit Secrets** rather than committing them.

A template is provided at:

```text
.streamlit/secrets.toml.example
```

---

## Fastest complete run

If you just want the shortest path from a fresh clone to the working demo:

```bash
python -m pip install -r requirements-dev.txt

python scripts/download_data.py
python scripts/prepare_kpi_data.py

python scripts/download_docs.py
python scripts/build_index.py

ollama pull qwen3:4b
ollama serve
```

Then, in another terminal with the same virtual environment:

```bash
streamlit run app.py
```

---

## Recommended first-time workflow

For learning and for understanding the project well enough to explain it in an interview, use this order:

```text
01_data_processing_and_eda.ipynb
        ↓
02_rag_demo_and_evaluation.ipynb
        ↓
Streamlit app
```

The notebooks explain the reasoning behind the implementation; the scripts and Streamlit app are the reusable application version.


---

## Public deployment

For the Streamlit Community Cloud deployment, including API billing protection, secrets, self-bootstrap behavior, and how to make the app public while keeping the GitHub repository private, see:

```text
DEPLOYMENT.md
```
