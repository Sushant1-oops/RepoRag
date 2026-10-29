
import json
import os
import queue
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from repo_rag import auth, db, ingest, rag

app = FastAPI(title="Repo RAG", version="2.1.0")

db.init_db()
ingest.startup_migrations()

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
    allow_credentials=True,
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


# --- authentication -----------------------------------------------------------------


class AuthRequest(BaseModel):
    username: str
    password: str


def _auth_error(e: auth.AuthError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/register")
def register(req: AuthRequest, request: Request, response: Response):
    """Create an account and sign in (session cookie)."""
    try:
        user = auth.register(req.username, req.password)
    except auth.AuthError as e:
        raise _auth_error(e) from e
    auth.issue_session(response, request, user["id"])
    _maybe_claim_legacy(user)
    return user


@app.post("/api/auth/login")
def login(req: AuthRequest, request: Request, response: Response):
    try:
        user = auth.login(req.username, req.password)
    except auth.AuthError as e:
        raise _auth_error(e) from e
    auth.issue_session(response, request, user["id"])
    return user


@app.post("/api/auth/logout")
def logout(request: Request, response: Response, user: dict = Depends(auth.get_current_user)):
    auth.logout(request)
    auth.clear_session_cookie(response)
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: dict = Depends(auth.get_current_user)):
    return user


def _maybe_claim_legacy(user: dict) -> None:
    """On registration, the owner named in LEGACY_OWNER_NAME inherits the
    pre-auth repos and chats so their existing data stays visible."""
    owner = os.getenv("LEGACY_OWNER_NAME", "").strip()
    if not owner or user["username"].lower() != owner.lower():
        return
    moved_repos = ingest.claim_legacy_data(user["id"])
    moved_chats = db.claim_legacy_chats(user["id"])
    if moved_repos or moved_chats:
        print(
            f"legacy data claimed by {user['username']}: "
            f"{moved_repos} repos, {moved_chats} chats"
        )


@app.get("/api/repos")
def list_repos(user: dict = Depends(auth.get_current_user)):
    return [
        {"id": repo["id"], "repo": repo.get("repo_id"), **repo}
        for repo in ingest.list_repos(user["id"])
    ]


class IndexRequest(BaseModel):
    url: str


@app.post("/api/repos")
def index_repo_stream(req: IndexRequest, user: dict = Depends(auth.get_current_user)):
    """Clone, filter, chunk and embed a GitHub repo; stream progress via SSE."""
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    events: queue.Queue = queue.Queue()

    def run():
        try:
            stats = ingest.index_repo(
                url,
                progress_cb=lambda msg: events.put(("progress", msg)),
                user_id=user["id"],
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
def delete_repo(repo: str, user: dict = Depends(auth.get_current_user)):
    info = ingest.load_repos_index().get(repo)
    # 404 both for missing and for other users' repos, so ids are not confirmed
    if info is None or info.get("user_id") != user["id"]:
        raise HTTPException(status_code=404, detail=f"Repository {repo!r} is not indexed")
    ingest.delete_repo(repo)
    return {"ok": True}



@app.get("/api/chats")
def list_chats(user: dict = Depends(auth.get_current_user)):
    """The signed-in user's saved conversations, most recently active first."""
    return db.list_chats(user["id"])


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: int, user: dict = Depends(auth.get_current_user)):
    chat = db.get_chat(chat_id)
    if chat is None or chat["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    return chat


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int, user: dict = Depends(auth.get_current_user)):
    if not db.delete_chat(chat_id, user["id"]):
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
def chat(req: ChatRequest, user: dict = Depends(auth.get_current_user)):
    """Grounded answer stream, persisted to SQLite so it can be resumed.

    SSE events: chat -> sources -> token* -> done / error. The chat event
    carries the chat_id (created on the fly when not supplied) so the client
    can keep following the same conversation. Retrieval is scoped to the
    signed-in user's repositories.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if req.chat_id is not None:
        chat = db.get_chat(req.chat_id)
        if chat is None or chat["user_id"] != user["id"]:
            raise HTTPException(status_code=404, detail=f"Chat {req.chat_id} not found")

    def stream():
        chat_id = req.chat_id
        if chat_id is None:
            chat_id = db.create_chat(question[:80], user["id"])["id"]
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
                user_id=user["id"],
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
