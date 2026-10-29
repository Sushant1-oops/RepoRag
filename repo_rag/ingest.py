"""Clone a GitHub repository, chunk its text files and store them in Qdrant."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient, models

# DATA_DIR lets deployments (e.g. Render) point Qdrant at a mounted disk;
# defaults to <project>/data locally.
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DB_DIR = DATA_DIR / "qdrant"
REPOS_INDEX_FILE = DATA_DIR / "repos.json"
COLLECTION_NAME = "repo_chunks"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim vectors
EMBED_DIM = 384
EMBED_BATCH = 64     # chunk texts embedded per encode() call
UPSERT_BATCH = 256   # chunks accumulated before one upsert

_STORE_LOCK = threading.Lock()  # the embedded engine is not thread-safe


@contextmanager
def store():
    """Serialize access to the embedded Qdrant engine and yield the client."""
    with _STORE_LOCK:
        yield get_client()

SKIP_DIRS = {
    ".git", ".github", ".idea", ".vscode", ".svn", "__pycache__",
    "node_modules", ".venv", "venv", "env", "dist", "build", "target",
    "vendor", ".next", ".nuxt", "coverage", ".gradle", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "htmlcov", ".cache", ".terraform",
    "site-packages", "Pods", "assets", "bower_components", ".yarn", ".pnp",
}

TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".kts", ".swift",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".go", ".rs", ".rb", ".php",
    ".m", ".mm", ".scala", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".sql", ".html", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
    ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".md", ".rst", ".txt", ".csv", ".tsv", ".tex", ".gitignore",
    ".dockerfile", ".tf", ".hcl", ".lua", ".pl", ".r", ".dart", ".ex",
    ".exs", ".erl", ".hs", ".clj", ".cljs", ".vb", ".asm", ".s", ".proto",
    ".gradle", ".properties", ".ipynb",
}

NO_EXTENSION_TEXT_FILES = {
    "dockerfile", "makefile", "gemfile", "rakefile", "license", "licence",
    "readme", "changelog", "procfile", "vagrantfile", ".gitignore",
    ".dockerignore", ".editorconfig", ".gitattributes", ".env.example",
}


LOCK_FILES = {
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "pipfile.lock", "uv.lock", "cargo.lock", "composer.lock",
    "gemfile.lock", "go.sum", "go.work.sum", "packages.lock.json", "mix.lock",
    "flake.lock", "podfile.lock", "cartfile.resolved", "bun.lockb", "deno.lock",
}


GENERATED_SUFFIXES = (
    ".min.js", ".min.css", ".min.html", ".bundle.js", ".chunk.js", ".map",
    ".pb.go", "_pb2.py", "_pb2_grpc.py", ".g.dart", ".pb.cc", ".pb.h",
    ".generated.cs", ".designer.cs", ".resx", ".snap",
)

MAX_FILE_BYTES = 200_000
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Qdrant in local mode: an embedded engine persisted under DB_DIR."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(DB_DIR))


@lru_cache(maxsize=1)
def get_embedder():
    """Shared sentence-transformers model (the store only keeps raw vectors)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def encode(texts: list[str]) -> list[list[float]]:
    """Embed texts with the shared model, ready to store as Qdrant vectors."""
    return get_embedder().encode(
        texts, batch_size=EMBED_BATCH, show_progress_bar=False
    ).tolist()


def _ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBED_DIM, distance=models.Distance.COSINE,
            ),
        )


def chunk_count() -> int:
    """Total number of stored chunks (0 before the first index run)."""
    with store() as client:
        if not client.collection_exists(COLLECTION_NAME):
            return 0
        return client.count(collection_name=COLLECTION_NAME, exact=True).count


def repo_filter(safe_repo: str) -> models.Filter:
    """Payload filter that narrows queries to a single repo."""
    return models.Filter(
        must=[models.FieldCondition(key="repo",
                                    match=models.MatchValue(value=safe_repo))]
    )


def repo_id_from_url(url: str) -> str:
    """owner/name from a GitHub URL, or the last path segment otherwise."""
    url = url.strip().rstrip("/").removesuffix(".git")
    match = re.match(r"(?:https?://)?(?:www\.)?github\.com/([^/]+/[^/]+)", url)
    if match:
        return match.group(1)
    return url.split("/")[-1] or "unknown"


def collection_safe_id(repo_id: str) -> str:
    """Payload-safe version of the repo id (used as the repo filter value)."""
    return repo_id.replace("/", "__")


# --- user scoping -----------------------------------------------------------------
# Registry keys and point payloads carry the owning user:  "u<id>__owner__name".
# Two users can index the same GitHub repo without colliding on point ids
# (chunk ids embed the scoped safe id) or on registry keys.

LEGACY_USER_ID = 0  # pre-auth data lives here until claimed
_USER_KEY_RE = re.compile(r"^u(\d+)__(.+)$")


def user_scoped_safe_id(repo_id: str, user_id: int) -> str:
    """Registry/payload key for one user's copy of a repo."""
    return f"u{user_id}__{collection_safe_id(repo_id)}"


def scope_filter(
    safe_repo: str | None = None, user_id: int | None = None
) -> models.Filter | None:
    """Qdrant filter narrowing to one repo and/or one user's chunks."""
    must = []
    if safe_repo:
        must.append(
            models.FieldCondition(key="repo", match=models.MatchValue(value=safe_repo))
        )
    if user_id is not None:
        must.append(
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
        )
    return models.Filter(must=must) if must else None


_migration_done = False


def _read_registry() -> dict:
    if REPOS_INDEX_FILE.exists():
        return json.loads(REPOS_INDEX_FILE.read_text(encoding="utf-8"))
    return {}


def _migrate_registry_and_payloads() -> None:
    """One-time move of the pre-auth registry into the scoped layout.

    Old entries are keyed by the raw repo id ("owner/name") with no user
    info; they move under legacy user 0, and the matching Qdrant payloads
    get the new "repo" value plus a user_id stamp so per-user filters work.
    Idempotent: fully-migrated entries are left untouched.
    """
    index = _read_registry()
    if not index:
        return
    new_index: dict = {}
    rewrites: list[tuple[str, str, int]] = []  # (old payload repo value, new key, user)
    changed = False
    for key, info in index.items():
        if not isinstance(info, dict):
            new_index[key] = info
            continue
        match = _USER_KEY_RE.match(key)
        user_id = info.get("user_id")
        if user_id is None:
            user_id = int(match.group(1)) if match else LEGACY_USER_ID
        raw_repo = info.get("repo_id")
        if raw_repo is None:
            raw_repo = (
                repo_id_from_url(info["url"]) if info.get("url")
                else (key.split("__", 1)[1].replace("__", "/") if match else key)
            )
        safe_key = user_scoped_safe_id(raw_repo, user_id)
        new_index[safe_key] = {**info, "repo_id": raw_repo, "user_id": user_id}
        if safe_key != key or "user_id" not in info or "repo_id" not in info:
            changed = True
            # old points may carry either the raw safe id or the scoped key
            rewrites.append((collection_safe_id(key), safe_key, user_id))
            if _USER_KEY_RE.match(key):
                rewrites.append((key, safe_key, user_id))
    if not changed:
        return
    save_repos_index(new_index)
    with store() as client:
        if not client.collection_exists(COLLECTION_NAME):
            return
        for old_value, new_key, user_id in rewrites:
            client.set_payload(
                COLLECTION_NAME,
                payload={"repo": new_key, "user_id": user_id},
                points=models.Filter(
                    must=[models.FieldCondition(
                        key="repo", match=models.MatchValue(value=old_value))]
                ),
            )
        # any point still missing a user_id belongs to the legacy namespace
        client.set_payload(
            COLLECTION_NAME,
            payload={"user_id": LEGACY_USER_ID},
            points=models.Filter(must_not=[models.FieldCondition(key="user_id")]),
        )


def _ensure_migrated() -> None:
    global _migration_done
    if _migration_done:
        return
    _migration_done = True
    _migrate_registry_and_payloads()


def startup_migrations() -> None:
    """Run the registry/payload migration once, e.g. at server start-up."""
    _ensure_migrated()


def clone_repo(url: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="repo_rag_"))
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(temp_dir / "repo")],
            check=True, capture_output=True, text=True, timeout=300,
        )
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"git clone failed:\n{e.stderr.strip()}") from e
    except subprocess.TimeoutExpired:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("git clone timed out after 300 seconds") from None
    return temp_dir / "repo"


def is_junk_file(path: Path) -> bool:
    """Lock files, minified bundles, source maps and generated code."""
    name = path.name.lower()
    if name in LOCK_FILES or name.endswith(".lock"):
        return True
    if name.endswith(GENERATED_SUFFIXES):
        return True
    return False


def is_text_file(path: Path) -> bool:
    if is_junk_file(path):
        return False
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if path.suffix == "" and path.name.lower() in NO_EXTENSION_TEXT_FILES:
        return True
    return False


def iter_repo_files(repo_dir: Path):
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_dir)
        if any(part.lower() in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if not is_text_file(path):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip() or "\x00" in text[:1024]:
            continue
        yield str(rel).replace("\\", "/"), text


def chunk_text(text: str):
    """Yield (start_line, end_line, chunk_text) tuples, line-aligned with overlap."""
    lines = text.splitlines(keepends=True)
    chunks = []
    cur, cur_len, start_line = [], 0, 1
    for i, line in enumerate(lines, start=1):
        if cur and cur_len + len(line) > CHUNK_MAX_CHARS:
            chunks.append((start_line, i - 1, "".join(cur)))
            tail, tail_len = [], 0
            for prev in reversed(cur):
                if tail_len + len(prev) > CHUNK_OVERLAP_CHARS:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            cur, cur_len = tail, tail_len
            start_line = i - len(tail)
        cur.append(line)
        cur_len += len(line)
    if cur:
        chunks.append((start_line, len(lines), "".join(cur)))
    return chunks


def load_repos_index() -> dict:
    _ensure_migrated()
    return _read_registry()


def save_repos_index(index: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPOS_INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def list_repos(user_id: int) -> list[dict]:
    """Registry entries owned by one user; `id` is the user-scoped safe key
    and `repo` the display name ("owner/name")."""
    return [
        {"id": key, **info}
        for key, info in load_repos_index().items()
        if isinstance(info, dict) and info.get("user_id") == user_id
    ]


def fetch_chunks(repo_id: str | None = None, user_id: int | None = None):
    """Stored chunks as (chunk_ids, texts, metas) for one repo and/or one user."""
    with store() as client:
        if not client.collection_exists(COLLECTION_NAME):
            return [], [], []
        scroll_filter = scope_filter(
            collection_safe_id(repo_id) if repo_id else None, user_id
        )
        ids, docs, metas = [], [], []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME, scroll_filter=scroll_filter,
                with_payload=True, limit=256, offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                ids.append(payload.get("chunk_id", str(point.id)))
                docs.append(payload.get("text", ""))
                metas.append({
                    "repo": payload.get("repo"), "file": payload.get("file"),
                    "start_line": payload.get("start_line"),
                    "end_line": payload.get("end_line"),
                })
            if offset is None:
                break
        return ids, docs, metas


def claim_legacy_data(new_user_id: int) -> int:
    """Reassign the legacy (pre-auth) repos to a real user; returns the count.

    Registry keys move from u0__... to u<new>__... and the affected Qdrant
    payloads follow, so retrieval scoped to the new user finds the chunks.
    """
    index = _read_registry()
    renames: list[tuple[str, str]] = []
    for key, info in list(index.items()):
        if not isinstance(info, dict) or info.get("user_id") != LEGACY_USER_ID:
            continue
        raw = info.get("repo_id") or repo_id_from_url(info.get("url", "")) or key
        new_key = user_scoped_safe_id(raw, new_user_id)
        index.pop(key)
        index[new_key] = {**info, "repo_id": raw, "user_id": new_user_id}
        renames.append((key, new_key))
    if not renames:
        return 0
    save_repos_index(index)
    with store() as client:
        if not client.collection_exists(COLLECTION_NAME):
            return len(renames)
        for old_key, new_key in renames:
            for old_value in {old_key, collection_safe_id(old_key)}:
                client.set_payload(
                    COLLECTION_NAME,
                    payload={"repo": new_key, "user_id": new_user_id},
                    points=models.Filter(
                        must=[models.FieldCondition(
                            key="repo", match=models.MatchValue(value=old_value))]
                    ),
                )
    return len(renames)


def delete_repo(repo_id: str) -> None:
    with store() as client:
        if client.collection_exists(COLLECTION_NAME):
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=repo_filter(collection_safe_id(repo_id))
                ),
            )
    index = load_repos_index()
    index.pop(repo_id, None)
    save_repos_index(index)


def index_repo(url: str, progress_cb=None, user_id: int = LEGACY_USER_ID) -> dict:
    """Clone, chunk, embed and index a repository for one user. Returns stats."""
    repo_id = repo_id_from_url(url)
    # every artifact (chunk ids, payload "repo", registry key) is namespaced
    # by the user so two users can index the same repo independently
    safe_id = user_scoped_safe_id(repo_id, user_id)
    progress = progress_cb or (lambda msg: None)

    progress(f"Cloning {repo_id} ...")
    repo_dir = clone_repo(url)
    try:
        delete_repo(repo_id)  # re-index: drop previous chunks for this repo

        with store() as client:
            _ensure_collection(client)
        batch_ids, batch_docs, batch_metas = [], [], []
        file_count, chunk_count = 0, 0
        seen_files = set()

        def upsert_batch() -> None:
            # Embed outside the store lock; only the upsert itself is serialized.
            vectors = encode(batch_docs)
            points = [
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, cid)),
                    vector=vector,
                    payload={
                        "chunk_id": cid, "repo": meta["repo"], "file": meta["file"],
                        "start_line": meta["start_line"], "end_line": meta["end_line"],
                        "text": doc, "user_id": user_id,
                    },
                )
                for cid, doc, meta, vector
                in zip(batch_ids, batch_docs, batch_metas, vectors)
            ]
            with store() as qdrant:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            batch_ids.clear()
            batch_docs.clear()
            batch_metas.clear()

        for rel_path, text in iter_repo_files(repo_dir):
            seen_files.add(rel_path)
            file_count += 1
            if file_count % 50 == 0:
                progress(f"Read {file_count} files, {chunk_count} chunks so far ...")
            for start, end, chunk in chunk_text(text):
                chunk_count += 1
                batch_ids.append(f"{safe_id}:{rel_path}:{start}-{end}:{chunk_count}")
                batch_docs.append(f"File: {rel_path}\n{chunk}")
                batch_metas.append({
                    "repo": safe_id, "file": rel_path,
                    "start_line": start, "end_line": end,
                })
            if len(batch_ids) >= UPSERT_BATCH:
                upsert_batch()

        if batch_ids:
            upsert_batch()

        index = load_repos_index()
        index[safe_id] = {
            "url": url, "files": file_count, "chunks": chunk_count,
            "repo_id": repo_id, "user_id": user_id,
        }
        save_repos_index(index)
        return {"repo": repo_id, "files": file_count, "chunks": chunk_count}
    finally:
        shutil.rmtree(repo_dir.parent, ignore_errors=True)
