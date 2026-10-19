# Repo RAG

Ask any question about a GitHub repository. It clones the repo, filters out
noise (lock files, `node_modules`, minified bundles, generated code), chunks
the important text files, embeds them locally with sentence-transformers,
stores them in ChromaDB, and streams grounded answers from a Groq LLM using
context found by **hybrid search** (BM25 keyword + vector semantic), re-ranked
by a local **cross-encoder**. Every claim is cited as `file:line`.

## Stack

- **Frontend**: React 19 + Vite 7 + TypeScript, Tailwind CSS v4
  (markdown rendering with `react-markdown` + `rehype-highlight`)
- **Backend**: FastAPI, streaming answers over **Server-Sent Events**
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (local, free)
- **Vector store**: ChromaDB (persistent, in `data/chroma`)
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
   chunks + metadata (repo, file, lines) ──all-MiniLM-L6-v2──▶ ChromaDB

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
