# Results

Evaluation results for the current Telecom-RAG system. Local and hosted measurements are reported separately because they use different retrieval and inference backends.

## Local retrieval

The local retrieval benchmark evaluates all 20 questions at `k=4`.

| Retrieval mode | Source hit | Source recall | Source precision | MRR | Evidence recall | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense BGE + FAISS | **0.900** | **0.875** | 0.675 | 0.792 | **0.800** | **0.016 s** |
| Hybrid BGE + BM25 + RRF | 0.850 | 0.850 | 0.650 | 0.817 | **0.800** | 0.021 s |
| Hybrid + cross-encoder reranking | **0.900** | **0.875** | **0.788** | **0.829** | 0.775 | 1.141 s |

The historical ablation shows that BM25/RRF alone did not improve source recall over dense retrieval, while the hybrid+reranker configuration improved precision and MRR at a substantial latency cost. The default local application path has since been simplified to dense BGE/FAISS candidates followed directly by the cross-encoder reranker. That exact dense+reranker path was introduced after this recorded run, so its metrics are not retroactively inferred from the hybrid+reranker row; rerun the notebook to measure it directly.

## Local generation

The current local generation comparison contains the first 5 benchmark questions and compares the same local LLM with and without retrieved context.

| Mode | Semantic similarity | Required-term recall | Answer-context similarity | Generation latency | Total latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLM only | 0.789 | 0.733 | — | 4.001 s | 4.001 s |
| RAG | **0.800** | **1.000** | 0.840 | 4.169 s | 5.340 s |

On this bounded sample, RAG increased required-term recall from **0.733 to 1.000**. The strongest improvement was on corpus-specific questions, where required-term recall increased from **0.000 to 1.000**.

Citation metrics are not treated as representative in this local generation run because the 96-token output cap frequently truncated answers before citation markers.

### Local run configuration

- 20 retrieval questions
- 5 generation questions
- local BGE embeddings + FAISS
- BM25 + RRF evaluated as an ablation
- MiniLM cross-encoder reranker
- Ollama generation with `qwen2.5:3b`
- `temperature=0`
- `max_output_tokens=96`

The local corpus was incomplete during this run: 6 of 10 configured sources were downloaded successfully, while four Ericsson pages returned HTTP 403. The results should therefore be interpreted as a partial-corpus benchmark.

## Hosted evaluation

A separate hosted benchmark is implemented for the production stack:

```text
Supabase pgvector + PostgreSQL FTS
                 ↓
                RRF
                 ↓
      Cloudflare BGE reranker
                 ↓
       Cloudflare generation
```

It also includes the dedicated semantic-routing benchmark.

The hosted evaluation has **not produced benchmark scores yet** because the GitHub Actions environment does not currently contain the required Cloudflare and Supabase credentials. No hosted results are reported here until that run completes successfully.

## Reproducing the local results

Run the evaluation notebook to regenerate the detailed CSVs and executed notebook under `eval/results/`. Generated evaluation artifacts are intentionally ignored by Git; this file keeps the stable summary in version control.

```bash
uv run jupyter lab
```

Open `notebooks/02_rag_evaluation.ipynb` and run the evaluation cells.
