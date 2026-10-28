# Repo RAG

Ask any question about a GitHub repository. It clones the repo, filters out
noise (lock files, `node_modules`, minified bundles, generated code), chunks
the important text files, embeds them locally with sentence-transformers,
stores them in Qdrant, and streams grounded answers from a Groq LLM using
context found by **hybrid search** (BM25 keyword + vector semantic), re-ranked
by a local **cross-encoder**. Every claim is cited as `file:line`.

## Stack

- **Frontend**: React 19 + Vite 7 + TypeScript, Tailwind CSS v4
  (markdown rendering with `react-markdown` + `rehype-highlight`)
- **Backend**: FastAPI, streaming answers over **Server-Sent Events**
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (local, free)
- **Vector store**: Qdrant (local mode — embedded engine, persistent, in `data/qdrant`)
- **Keyword search**: BM25 (`rank-bm25`) with a code-aware tokenizer that
  splits `snake_case` / `camelCase` identifiers
- **Fusion**: weighted Reciprocal Rank Fusion (RRF) — vector 0.6, BM25 0.4
- **Reranker**: local cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`
  (capped at 20 candidate pairs to keep CPU latency low)
- **LLM**: Groq — default `openai/gpt-oss-120b`, configurable via `GROQ_MODEL`
- **Chat history**: SQLite (`data/app.db`) — every conversation is persisted and can be resumed

## Architecture

```
GitHub repo ──git clone──▶ filter ──▶ important text files only
   (skip node_modules, .git, lock files, *.min.js, source maps,
    generated code, .env secrets, binary/large files)

files ──chunker──▶ chunks (~1200 chars, 200 overlap, line-aligned)
   chunks + metadata (repo, file, lines) ──all-MiniLM-L6-v2──▶ Qdrant

question ─┬─ embed ──▶ semantic search (cosine, top 24) ─┐   (both run
          └─ tokenize ─▶ BM25 keyword search (top 24) ───┤    in parallel)
                                                         ▼
                            Reciprocal Rank Fusion (0.6 vec / 0.4 bm25)
                                                         ▼
                 cross-encoder rerank (top-k) + adjacent-chunk expansion
                                                         ▼
                       14k-char budgeted context ──stream──▶ Groq LLM
                                                         ▼
                                 SSE: sources → tokens → done (citations, timings)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your Groq API key in it
(free key: https://console.groq.com/keys):

```
GROQ_API_KEY=gsk_...
# optional, defaults to openai/gpt-oss-120b
GROQ_MODEL=openai/gpt-oss-120b
```

### Frontend

```bash
cd frontend
npm install
npm run build       # type-checks (tsc) and outputs frontend/dist
```

## Run

Production (single process — FastAPI serves the built frontend):

```bash
.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**.

Development (hot reload — run both):

```bash
# terminal 1: backend
.venv\Scripts\python.exe -m uvicorn server:app --port 8000
# terminal 2: frontend (proxies /api to port 8000)
cd frontend && npm run dev
```

Then open http://localhost:5173.

### Using the app

1. Paste a GitHub URL in the sidebar and click **+** — progress streams live
   (clone → filter → embed → done, with file/chunk counts).
2. Ask questions, e.g. *"What does this project do?"*, *"Where is config
   parsed?"*. Answers stream token by token with a metrics footer
   (sources, retrieval / first-token / total latency, model).
3. **Chats persist** in SQLite: the sidebar lists every conversation with a
   timestamp — click one to resume exactly where you left off, use
   **+ New** to start fresh, or delete old ones. Follow-up questions
   include the recent turns, so "go deeper on that function" works.
4. The **Sources** panel shows every retrieved chunk with its `rerank`,
   `vec` (semantic) and `bm25` (keyword) scores; click a card to read the
   raw chunk text.
5. Scope questions to one repo via the sidebar or the selector in the
   composer; toggle **hybrid retrieval**, the **reranker**, and **top-k**
   in the sidebar to compare retrieval quality.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | status, chunk count, model name |
| GET | `/api/repos` | indexed repositories |
| POST | `/api/repos` | index a repo — SSE: `progress` → `done` / `error` |
| DELETE | `/api/repos?repo=owner__name` | remove from index |
| GET | `/api/chats` | saved conversations, most recently active first |
| GET | `/api/chats/{id}` | one conversation with its messages |
| DELETE | `/api/chats/{id}` | delete a conversation |
| POST | `/api/chat` | ask a question — SSE: `chat` → `sources` → `token`* → `done` / `error` |

Chat SSE events: `chat` (`chat_id` — a new row is created on the fly when
the request has none, so the client can keep following the conversation),
then `sources` (reranked chunks with scores), then `token` (one streamed
delta each), then `done` (`citations`, `retrieval_ms`, `first_token_ms`,
`total_ms`, `model`). Each exchange is written to SQLite as it streams —
even a generation stopped mid-way keeps its partial answer.

## How answers stay grounded

- The system prompt forbids inventing files, APIs or behavior and requires a
  `path/to/file.py:LINE` citation after every claim; it must say so plainly
  when the context is insufficient.
- Context assembly ([repo_rag/rag.py](repo_rag/rag.py)) appends the adjacent
  chunk above and below each hit so the model sees surrounding code, within a
  14k-character budget that keeps the best-scoring chunks.
- After generation, `file:line` references are validated against the retrieved
  sources; model-internal `【Chunk N†…】` markers are translated back to real
  paths. Citations are displayed as chips under each answer.
- Follow-up questions include the last few turns (bounded to ~3k chars) for
  continuity, but the model is instructed to ground every claim in the
  retrieved repository context, not the chat history.

## Why hybrid + rerank?

- **Semantic search** finds conceptually related code even without shared
  words ("login flow" → `auth/session.py`).
- **BM25** excels at exact identifiers (`getUserID`, `MAX_CONTEXT_CHARS`).
- **RRF** merges both rankings without score-scale hacks.
- The **cross-encoder** reads question + chunk together and re-orders the
  fused candidates, which consistently improves precision over pure
  bi-encoder ranking.

## Evaluation

[eval/](eval) holds an end-to-end harness that measures the pipeline rather
than eyeballing it:

| Phase | What it produces |
| --- | --- |
| `testset.json` | 60 questions LLM-generated from real indexed chunks, each with a reference answer and the source file + chunk ids as ground truth |
| `retrieval_results.json` | top-16 hits per question with file-level Recall@k, Precision@k, MRR@k, NDCG@k (k = 4/8/16) and exact-source-chunk recall |
| `answers.json` | full-pipeline answers (hybrid retrieve → rerank → Groq), sequential so latency is realistic, plus prompt/completion tokens |
| `citation_checks.json` | deterministic citation checks: valid against the corpus, grounded in what was retrieved, and hitting the ground-truth file |
| `ragas_scores.json` | Ragas `faithfulness`, `answer_relevancy`, `answer_correctness` with an LLM judge and local embeddings, plus per-metric row coverage |
| `report.md`, `summary.json` | aggregated tables: quality, citation accuracy, p50/p95 latency per stage, token usage, failure rate |

```powershell
pip install -r requirements-eval.txt
.venv\Scripts\python.exe eval\generate_testset.py   # once: builds testset.json
.venv\Scripts\python.exe eval\run_eval.py           # resumable, phase checkpoints
```

- Stop the API server first: Qdrant's embedded engine takes an exclusive lock
  on `data/qdrant`, so only one process may read it at a time.
- Every phase is checkpointed, and answers are checkpointed per question, so a
  run interrupted by provider rate limits resumes where it stopped
  (`--force` recomputes everything, `--skip-ragas` skips the judge).
- The judge is provider-configurable (`RAGAS_JUDGE_MODEL`,
  `RAGAS_JUDGE_BASE_URL`, `RAGAS_JUDGE_KEY_ENV`) and deliberately a different
  model family from the answering model to avoid self-preference bias.
- Ragas fires ~15-25 short strict-JSON calls per answer, which free tiers
  struggle with: Groq enforces an output-tokens-per-minute cap, and OpenRouter
  `:free` routes return upstream 400/429/502 errors or reasoning-token bloat.
  On a capped tier the scoring pass has to be fully sequential
  (`RAGAS_BATCH=1 RAGAS_WORKERS=1`): two concurrent judge calls each reserve
  their whole `max_tokens`, which the per-minute budget cannot cover, and Ragas
  then reports the killed job as a `None` score. `RAGAS_JUDGE_MAX_TOKENS` sizes
  that reservation and `RAGAS_JUDGE_JOB_TIMEOUT` raises Ragas' per-job wall
  clock above the quota backoff. The harness waits out rate limits with the
  retry hint the API returns, checkpoints per batch and honours
  `RAGAS_TIME_BUDGET`, so an interrupted scoring run resumes without redoing
  finished rows.
- `RAGAS_METRICS` selects which metrics to score. `faithfulness` asks the judge
  to echo every atomic claim, so its verdict runs longer than a free Groq
  minute allows; when it cannot be scored the report says so and groundedness
  falls back to the deterministic citation checks.
- Ragas may cover a sample (`RAGAS_SAMPLE`); every other phase covers all
  questions, and `report.md` prints per-metric row coverage so an average is
  never read without its denominator.
- Tunables: `ANSWER_ATTEMPTS`, `ANSWER_DELAY_SECONDS`, `RAGAS_MAX_CONTEXTS`,
  `RAGAS_MAX_CONTEXT_CHARS`, `RAGAS_WORKERS`, `RAGAS_BATCH`,
  `RAGAS_TIME_BUDGET`, `RAGAS_JUDGE_TIMEOUT`, `RAGAS_JUDGE_RETRIES`,
  `RAGAS_JUDGE_MAX_TOKENS`, `RAGAS_JUDGE_JOB_TIMEOUT`, `RAGAS_METRICS`,
  `RAGAS_SAMPLE`.

## Deploy the backend on AWS (start/stop, not 24/7)

The backend runs in Docker on a single EC2 instance. Everything persistent
(Qdrant index, SQLite chats, repo registry) lives in `data/`, bind-mounted onto
the instance's EBS volume — so the instance can be **stopped when idle** (you
pay ~$2.4/month for the disk, nothing for compute) and started again with all
data intact.

Prerequisites: `aws configure` once with credentials that can create EC2
resources, and this branch pushed to GitHub (the instance clones your remote).

```powershell
# one-time setup — key pair, security group, Ubuntu 24.04 instance + boot script
.\deploy\aws-create.ps1                       # t3.medium in us-east-1 by default
.\deploy\aws-start.ps1                        # first boot builds the image (~5-10 min)
.\deploy\aws-env.ps1                          # push .env (GROQ_API_KEY) to the instance
.\deploy\aws-upload-data.ps1                  # optional: reuse the local index/chats

# day-to-day
.\deploy\aws-start.ps1                        # boots + waits for /api/health
.\deploy\aws-stop.ps1                         # compute billing stops, data persists
.\deploy\aws-ssh.ps1                          # shell on the box (docker compose logs)
```

The app is served at `http://<ip>:8000` — the container serves the built
frontend too, so it is one same-origin deployment with no CORS or TLS juggling.
The public IP rotates on every stop/start (the scripts refresh
`deploy/aws.json`; create with `-AllocateElasticIp` to keep one fixed IP).
HTTPS: the browser app works over plain HTTP because it is same-origin; if you
want the **Vercel** frontend (HTTPS) to call this backend, put Caddy with your
domain in front — the Vercel frontend cannot call an `http://` origin directly
(mixed content).

Updating the deployed code: commit + push, then
`ssh` in and run `git pull && docker compose -f deploy/docker-compose.yml up -d --build`.

## Troubleshooting

- **Windows fresh install pulls a multi-GB CUDA torch**: install the CPU
  build first:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  then `pip install -r requirements.txt`.
- **First query after starting the server is slow (~30–60 s)**: the embedding
  and reranker models load lazily on first use; subsequent queries are fast
  (warm retrieval ≈ 1–4 s on CPU).
- **`model_not_found` from Groq**: model availability on free accounts
  changes; check https://console.groq.com/docs/models and set `GROQ_MODEL`
  in `.env` (current alternatives: `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`).
- **Frontend changes don't appear**: rebuild with `npm run build`, or use the
  `npm run dev` flow above.

## Notes

- First run downloads the embedding model (~90 MB) and, on the first
  question, the reranker (~80 MB) — one-time, cached locally.
- Lock files (`package-lock.json`, `yarn.lock`, `Cargo.lock`, `go.sum`, ...),
  minified bundles, source maps, generated code (`*_pb2.py`, `*.pb.go`, ...),
  `.env` secrets, binaries and files over 200 KB are skipped during ingest.
- Tune retrieval in [repo_rag/retrieval.py](repo_rag/retrieval.py):
  `W_SEMANTIC` / `W_KEYWORD` / `RRF_K` / `CANDIDATES_PER_METHOD` /
  `RERANK_MAX_CANDIDATES`; context budget and neighbors in
  [repo_rag/rag.py](repo_rag/rag.py) (`MAX_CONTEXT_CHARS`, `NEIGHBORS_PER_HIT`).
