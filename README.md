# Telecom-RAG

A portfolio project for **5G network diagnostics with Retrieval-Augmented Generation (RAG)**.

The project combines:

- real Ericsson/AERPAW 5G NSA KPI measurements,
- reproducible pandas/scikit-learn analysis,
- public telecom standards/documentation,
- BGE retrieval embeddings,
- FAISS dense search + BM25 lexical search,
- reciprocal-rank fusion + cross-encoder reranking,
- LangChain components,
- LangGraph orchestration,
- the same LLM **with vs without RAG** evaluation,
- a Streamlit demo.

For local setup and the exact run order, see **[RUNNING.md](RUNNING.md)**. For the public Streamlit deployment, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.The central idea is deliberately simple:

```text
KPI measurements                    Technical documents
(the case to diagnose)              (knowledge used to explain it)
       |                                      |
       v                                      v
pandas / anomaly analysis       section-aware chunking
                                      |
                              BGE dense + BM25
                                      |
                              RRF + cross-encoder
       |                                      |
       +------------------+-------------------+
                          v
                       LangGraph
                          |
                          v
                         LLM
                          |
                          v
              explanation + source citations
```

## 1. Data

The KPI side is pinned to the public Dryad release:

**Ericsson 5G NSA network RF and throughput measurements on AERPAW network**  
DOI: https://doi.org/10.5061/dryad.wh70rxx06  
Version: **May 21, 2025**  
Archive: **Ericsson_Amir.zip**

The experiment contains LTE/NR measurements such as RSRP, SINR, CQI, MCS, RI, cell IDs, throughput and UAV geolocation for two yaw orientations.

Download/extract it with:

```bash
python scripts/download_data.py
```

If Dryad rejects the automated request, manually download `Ericsson_Amir.zip` from the DOI page, place it at:

```text
data/downloads/Ericsson_Amir.zip
```

and rerun the script.

The processing notebook aligns the separate KPI streams by timestamp and writes:

```text
data/processed/kpi_observations.csv
```

See [`data/README.md`](data/README.md) for the exact raw files and their roles.

## 2. RAG corpus

Run:

```bash
python scripts/download_docs.py
```

The corpus currently configures **10 focused sources**:

- ETSI / 3GPP TS 38.215 — NR physical-layer measurements
- ETSI / 3GPP TS 38.214 — NR physical-layer procedures for data
- ETSI / 3GPP TS 38.300 — NR / NG-RAN overall description
- ETSI / 3GPP TS 36.214 — LTE physical-layer measurements
- AERPAW Ericsson dataset description
- AERPAW Ericsson post-processing documentation
- Ericsson material on Massive MIMO / beamforming
- Ericsson traffic-pattern and capacity/coverage analysis
- Ericsson network-performance optimization material
- Ericsson Mobility Report June 2025

Downloaded third-party files are excluded from Git; the URLs and download script are committed for reproducibility.

See [`docs/SOURCES.md`](docs/SOURCES.md).

## 3. Exact environment

Recommended: **Python 3.12**.

Create a virtual environment and install the pinned notebook environment:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements.txt` contains the exact runtime/deployment pins.  
`requirements-dev.txt` adds the exact Jupyter packages.

### Local LLM (free)

The default development model is **Qwen3 4B through Ollama**.

Install Ollama separately, then:

```bash
ollama pull qwen3:4b
ollama serve
```

No LLM API key is needed for local use.

### Optional hosted LLM

For Streamlit Community Cloud or another hosted deployment, set:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.6-luna
```

The code keeps the LLM provider behind the same LangChain interface, so the RAG pipeline does not change.

## 4. Run the notebooks

Start Jupyter:

```bash
jupyter lab
```

Run in order:

### [`notebooks/01_data_processing_and_eda.ipynb`](notebooks/01_data_processing_and_eda.ipynb)

Explains and performs:

- raw file discovery,
- robust timestamp parsing,
- nearest-time KPI alignment,
- missing-data inspection,
- KPI distributions/correlations,
- Isolation Forest anomaly scoring,
- creation of the final observation table.

### [`notebooks/02_rag_demo_and_evaluation.ipynb`](notebooks/02_rag_demo_and_evaluation.ipynb)

Explains and performs:

- document loading and PDF extraction,
- section-aware contextual chunking,
- BGE retrieval embeddings,
- FAISS dense search,
- BM25 lexical search,
- reciprocal-rank fusion,
- cross-encoder reranking,
- dense vs hybrid vs reranked retrieval ablation,
- LangGraph workflow construction,
- documentation-only questions,
- KPI-aware questions,
- source hit/recall/precision and MRR evaluation,
- a 20-question benchmark split into general, corpus-specific, applied-diagnostic and cross-source categories,
- **same LLM: LLM-only vs final reranked RAG** comparison,
- semantic similarity, required-fact recall, citation validity, context-support proxy and latency analysis.

## 5. Run without notebooks

The same pipeline is exposed through scripts:

```bash
python scripts/prepare_kpi_data.py
python scripts/download_docs.py
python scripts/build_index.py
```

## 6. Run the Streamlit app

Local/Ollama:

```bash
streamlit run app.py
```

The app lets you:

- select a real KPI observation,
- ask a KPI-aware or documentation-only question,
- enable/disable RAG,
- optionally run the same LLM without RAG for comparison,
- inspect the route selected by LangGraph,
- inspect the dataset-relative KPI context,
- inspect retrieved source chunks and citations.

### Streamlit Community Cloud

Ollama runs on your own machine, so a normal Streamlit Community Cloud deployment should use the hosted provider instead. Add the following in Streamlit **Secrets**:

```toml
OPENAI_API_KEY = "..."
OPENAI_MODEL = "gpt-5.6-luna"
```

A template is included at `.streamlit/secrets.toml.example`.

## 7. Why LangGraph here?

LangGraph is not being added just to make the stack sound more complicated.

The graph performs a real conditional workflow:

```text
START
  |
route question
  |----------------------|
  |                      |
KPI-related             docs-only
  |                      |
analyze KPI              |
  |                      |
  +------> retrieve <-----+
             |
          generate
             |
            END
```

A question such as **“What is RSRP?”** skips KPI analysis. A question about the selected network observation first creates dataset-relative KPI context and then retrieves documentation.

## 8. Evaluation philosophy

The main comparison is controlled:

```text
same model + same question

LLM only
vs
LLM + retrieved context
```

Retrieval is evaluated separately from generation. This matters because RAG can fail in two different places:

1. **retrieval failure** — the useful document/chunk was never found;
2. **generation failure** — the right context was retrieved but the LLM still produced a poor answer.

The included benchmark is intentionally small and hand-auditable. It is meant to demonstrate experimental thinking, not claim a production-grade telecom benchmark.

## 9. Repository structure

```text
Telecom-RAG/
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── data/
│   ├── README.md
│   ├── raw/                 # downloaded, ignored by Git
│   └── processed/           # generated, ignored by Git
├── docs/
│   ├── SOURCES.md
│   └── corpus/              # downloaded, ignored by Git
├── eval/
│   └── questions.json
├── notebooks/
│   ├── 01_data_processing_and_eda.ipynb
│   └── 02_rag_demo_and_evaluation.ipynb
├── scripts/
│   ├── download_data.py
│   ├── download_docs.py
│   ├── prepare_kpi_data.py
│   └── build_index.py
└── src/telecom_rag/
    ├── config.py
    ├── data.py
    ├── kpi.py
    ├── documents.py
    ├── rag.py
    ├── retrieval.py
    ├── graph.py
    └── evaluation.py
```

## 10. Retrieval architecture

The default retrieval path is now:

```text
question
  ↓
clean retrieval query
  ↓
BGE/FAISS top-15      BM25 top-15
       \                /
        reciprocal-rank fusion
                 ↓
          ~20 candidates
                 ↓
      cross-encoder reranking
                 ↓
             top-4
                 ↓
               LLM
```

The Streamlit UI also exposes `dense`, `hybrid`, and `reranked` modes so the retrieval ablation can be demonstrated without changing code.

The detailed numeric KPI summary is **not** appended to the embedding query anymore. Only compact KPI concepts are used for retrieval; full values/percentiles are supplied later to the generator.

## 11. Evaluation philosophy

The benchmark is deliberately harder than the original definition-heavy version. Generic questions remain, but the majority now test corpus-specific facts, exact standards/files, applied Ericsson performance explanations, and cross-source retrieval.

Useful next experiments are still controlled ablations: chunk size, candidate counts, final top-k, BGE-small vs BGE-base, corpus subsets, and reranker on/off.

The project intentionally avoids hard-coding unsupported “good/bad” KPI thresholds and avoids unnecessary multi-agent complexity.
