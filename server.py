
import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from repo_rag import db, ingest, rag

app = FastAPI(title="Repo RAG", version="2.0.0")

db.init_db()

_env_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

print("CORS ALLOWED ORIGINS:", repr(_env_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *_env_origins,
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"

_SSE_DONE = object()  


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/health")
def health():
   
    # Loading the embedder here keeps the first question's latency down.
    ingest.get_embedder()
    return {
        "ok": True,
        "chunks": ingest.chunk_count(),
        "model": rag.get_model(),
    }


@app.get("/api/repos")
def list_repos():
    return [{"id": repo_id, **info} for repo_id, info in ingest.load_repos_index().items()]


class IndexRequest(BaseModel):
    url: str


@app.post("/api/repos")
def index_repo_stream(req: IndexRequest):
    """Clone, filter, chunk and embed a GitHub repo; stream progress via SSE."""
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    events: queue.Queue = queue.Queue()

    def run():
        try:
            stats = ingest.index_repo(
                url, progress_cb=lambda msg: events.put(("progress", msg))
            )
            events.put(("done", stats))
        except Exception as e:
            events.put(("error", str(e)))
        finally:
            events.put(_SSE_DONE)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            item = events.get()
            if item is _SSE_DONE:
                break
            kind, payload = item
            if kind == "progress":
                yield sse("progress", {"message": payload})
            elif kind == "done":
                yield sse("done", payload)
            else:
                yield sse("error", {"message": payload})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/repos")
def delete_repo(repo: str):
    if repo not in ingest.load_repos_index():
        raise HTTPException(status_code=404, detail=f"Repository {repo!r} is not indexed")
    ingest.delete_repo(repo)
    return {"ok": True}



@app.get("/api/chats")
def list_chats():
    """All saved conversations, most recently active first."""
    return db.list_chats()


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: int):
    chat = db.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    return chat


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int):
    if not db.delete_chat(chat_id):
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    return {"ok": True}


class ChatRequest(BaseModel):
    question: str
    chat_id: int | None = None  
    repo: str | None = None
    top_k: int = Field(default=8, ge=2, le=16)
    use_hybrid: bool = True
    use_reranker: bool = True


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Grounded answer stream, persisted to SQLite so it can be resumed.

    SSE events: chat -> sources -> token* -> done / error. The chat event
    carries the chat_id (created on the fly when not supplied) so the client
    can keep following the same conversation.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if req.chat_id is not None and db.get_chat(req.chat_id) is None:
        raise HTTPException(status_code=404, detail=f"Chat {req.chat_id} not found")

    def stream():
        chat_id = req.chat_id
        if chat_id is None:
            chat_id = db.create_chat(question[:80])["id"]
        yield sse("chat", {"chat_id": chat_id})
        history = db.chat_history(chat_id)

        parts: list[str] = []
        sources: list | None = None
        metrics: dict | None = None
        error: str | None = None
        saved = False

        def persist() -> None:
            nonlocal saved
            if saved:
                return
            saved = True
            db.append_message(chat_id, "user", question)
            db.append_message(
                chat_id, "assistant", "".join(parts),
                sources=sources, metrics=metrics, error=error,
            )

        try:
            for event, payload in rag.stream_answer(
                question,
                n_results=req.top_k,
                repo=req.repo,
                use_hybrid=req.use_hybrid,
                use_reranker=req.use_reranker,
                history=history,
            ):
                if event == "sources":
                    sources = payload
                elif event == "token":
                    parts.append(payload)
                elif event == "done":
                    metrics = payload
                elif event == "error":
                    error = str(payload)
                    yield sse("error", {"message": error})
                    continue
                yield sse(event, payload)
            persist()
        except GeneratorExit:
            persist()  
            raise
        except Exception as e:  
            error = str(e)
            persist()
            yield sse("error", {"message": error})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
