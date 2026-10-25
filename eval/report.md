# RAG pipeline evaluation report

*Generated 2026-10-25 13:15:12 — 60 questions across 2 repos (generator: `openai/gpt-oss-120b`, answers: `openai/gpt-oss-120b`, judge: `allam-2-7b`)*

## Dataset

- Questions: **60** (17 from Aadarsh6__Glypha_Web, 43 from Sushant1-oops__knowledge-chat)
- Ground truth: reference answer + the exact file the question was generated from (retrieval metrics are file-level, binary relevance)
- Failure rate: **0.0%**

## Retrieval quality (hybrid + rerank, global scope)

| Metric | @4 | @8 | @16 |
| --- | --- | --- | --- |
| RECALL | 0.983 | 0.983 | 0.983 |
| PRECISION | 0.246 | 0.123 | 0.061 |
| MRR | 0.939 | 0.939 | 0.939 |
| NDCG | 0.950 | 0.950 | 0.950 |
| Exact source-chunk in top-16 | 0.967 |

## Answer quality (Ragas, LLM judge + embeddings)

| Metric | What it measures | Mean | Rows scored |
| --- | --- | --- | --- |
| `answer_relevancy` | answer addresses the question | 0.6809 | 10 / 10 |
| `answer_correctness` | agreement with the reference answer | 0.8993 | 4 / 10 |

*Not LLM-scored: `faithfulness` - the free judge tier caps output tokens per minute, and a verdict for this metric needs a longer output than that window. Groundedness is covered by the deterministic citation checks below.*

## Agreement with the reference answer (deterministic, no judge)

- Content-word F1 between answer and reference over 60 answered questions: mean **0.4105**, p50 0.3938, range 0.0625-0.7246.
- This only shows that the answer talks about the same entities as the reference; semantic agreement is the Ragas `answer_correctness` row above. It is reported because it covers every answered question while the judge covers a quota-limited sample.

## Citation accuracy (deterministic)

- Checks cover 60 of 60 questions (questions without an answer are excluded).
- Answers with at least one `file:line` citation: **90.0%**
- Mean citations per answer: **1.52**
- Citations grounded in retrieved sources: **100.0%**
- Answers citing the ground-truth file: **90.0%**
- Cited files that hit the ground-truth file: **0.9907**

## Latency (single-user, sequential)

| Stage | p50 | p95 | mean |
| --- | --- | --- | --- |
| Retrieval (Qdrant + BM25 + rerank) | 6,352 ms | 33,343 ms | 16,438 ms |
| Retrieval, warm (first 5 skipped) | 1,702 ms | 1,860 ms | 1,689 ms |
| First token (Groq TTFB) | 831 ms | 1,376 ms | 884 ms |
| Total | 29,442 ms | 81,039 ms | 39,368 ms |
| Total, warm (cold starts excluded) | 29,031 ms | 62,683 ms | 32,705 ms |

_Raw numbers include the process cold start (the first queries of a run
load the embedding + reranker models); the warm rows are steady state._

## Token usage

- Prompt tokens: **220,841** total (~3680.7 / answer)
- Completion tokens: **35,871** total (~597.9 / answer)

## Notes & caveats

- Questions were LLM-generated from real indexed chunks, so ground truth is exact at the file level but generation is single-file: cross-file questions are out of scope.
- Retrieval ran with the production default (hybrid + reranker, global scope over both repos), so precision@k includes cross-repo noise by design.
- Judge model `allam-2-7b` via https://api.groq.com/openai/v1 differs from the answering model, which avoids same-family self-preference bias; Ragas scores remain probabilistic LLM judgments, not ground truth. The judge is a free-tier route selected by remaining quota rather than by capability, so treat these scores as indicative, not as a quality gate.
- Judge contexts are the same chunks the answering model saw (capped at 8 x 2400 chars), so well-grounded claims are not penalised by trimming.
- Ragas covers a sample: 10 of 60 answers scored, because the free judge route is upstream-rate-limited; retrieval, citation, latency and token metrics cover every question.
- Answer failures are provider quota errors (HTTP 429 on Groq's free-tier tokens-per-day cap), not pipeline defects; re-running the eval resumes only the questions that failed.
