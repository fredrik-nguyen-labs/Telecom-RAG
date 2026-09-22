# Evaluation results

Run date: 2026-09-22
Source revision: `0eb771d`
Execution environment: `uv` + local Ollama

## What was run

- Retrieval ablation: all 20 evaluation questions, `k=4`, across dense, hybrid, and reranked retrieval (`60` rows total).
- Generation comparison: the first 5 of the 20 questions, comparing `llm_only` with `rag` (`10` rows total). This bounded local run keeps the notebook reproducible on the local model; the source notebook still defines the full 20-question generation benchmark.
- Model: `qwen2.5:3b`, `temperature=0`, `max_output_tokens=96`.
- Provider: Ollama. Cloudflare credentials were not present, so the notebook selected local inference.
- Retrieval model stack: BGE embeddings, BM25/RRF hybrid retrieval, and the configured cross-encoder reranker.

The executed artifact contains the run metadata and all 46 executed cells:
`eval/results/02_rag_demo_and_evaluation.executed.ipynb`.

## Retrieval metrics

| Mode | Source hit | Source recall | Source precision | MRR | Evidence-term recall | Mean latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reranked | 0.900 | 0.875 | 0.788 | 0.829 | 0.775 | 1.141 |
| Dense | 0.900 | 0.875 | 0.675 | 0.792 | 0.800 | 0.016 |
| Hybrid | 0.850 | 0.850 | 0.650 | 0.817 | 0.800 | 0.021 |

Reranking tied dense retrieval on source hit and recall, but improved precision by 0.113 and MRR by 0.037. It was substantially slower: 1.141 seconds versus 0.016 seconds for dense and 0.021 seconds for hybrid. Evidence-term recall was slightly lower for reranking in this run (0.775 versus 0.800).

## Generation metrics: LLM-only versus RAG

| Mode | Semantic similarity | Required-term recall | Citation present | Answer-context similarity | Evidence-term recall | Generation latency (s) | Total latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM-only | 0.789 | 0.733 | 0.000 | — | — | 4.001 | 4.001 |
| RAG | 0.800 | 1.000 | 0.000 | 0.840 | 0.833 | 4.169 | 5.340 |

On the 5-question bounded comparison, RAG improved semantic similarity by `+0.011` and required-term recall by `+0.267` (26.7 percentage points). It added 1.339 seconds end-to-end on average, mostly from retrieval. The RAG answers were semantically well aligned with the retrieved context (`answer_context_similarity=0.840`) and recovered all required terms in this sample.

By category:

- Corpus-specific questions: semantic similarity improved from `0.600` to `0.734`; required-term recall improved from `0.000` to `1.000`.
- General telecom questions: semantic similarity moved from `0.836` to `0.816`; required-term recall improved from `0.917` to `1.000`.

## Data and limitations

The run configured 10 sources and downloaded 6 successfully. These Ericsson URLs returned HTTP 403 during this pass:

- `ericsson_massive_mimo_beamforming`
- `ericsson_traffic_patterns`
- `ericsson_ai_network_performance`
- `ericsson_mobility_report_june_2025`

The unavailable beamforming source is the expected source for one evaluation question, so the retrieval results should be read as a partial-corpus result. The source manifest is regenerated locally at `docs/corpus/_sources.json`.

Citation presence and citation-support metrics were zero in the bounded generation run. The 96-token cap truncates the notebook's structured answer format before citation markers; this is an evaluation configuration limitation, not evidence that the retrieved context was unsupported. A full-quality generation benchmark should use a larger output budget and a stable 10/10 source corpus.

The final local run used `qwen2.5:3b`. The installed `qwen3:4b` checkpoint emits a visible thinking trace with its Ollama template and did not produce answer text within the bounded token budget, so it was not used for the reported generation metrics.

## Raw outputs

- `eval/results/retrieval_ablation.csv`
- `eval/results/retrieval_summary.csv`
- `eval/results/generation_comparison.csv`
- `eval/results/generation_summary.csv`
- `eval/results/generation_by_category.csv`
