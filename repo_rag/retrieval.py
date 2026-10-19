import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from rank_bm25 import BM25Okapi

from .ingest import collection_safe_id, get_collection, load_repos_index

CANDIDATES_PER_METHOD = 24  
RRF_K = 60                  
W_SEMANTIC = 0.6            
W_KEYWORD = 0.4             
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_MAX_CANDIDATES = 20  

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+")
_CASE_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens; split snake_case and camelCase identifiers."""
    tokens = []
    for word in _WORD_RE.findall(text):
        for piece in word.split("_"):
            for part in _CASE_RE.findall(piece):
                token = part.lower()
                if len(token) >= 2 or token.isdigit():
                    tokens.append(token)
    return tokens


def _index_version() -> tuple:
    """Cheap fingerprint of the store; a change invalidates the BM25 cache."""
    repos = tuple(sorted(load_repos_index()))
    return (get_collection().count(), repos)


@lru_cache(maxsize=8)
def _keyword_index(safe_repo: str | None, version: tuple):
    """Build a BM25 index over the stored chunks for one repo (or all)."""
    collection = get_collection()
    where = {"repo": safe_repo} if safe_repo else None
    data = collection.get(where=where, include=["documents", "metadatas"])
    ids = list(data["ids"])
    docs = list(data["documents"] or [])
    metas = list(data["metadatas"] or [])
    tokenized = [_tokenize(doc) for doc in docs]
    if not docs or max(map(len, tokenized), default=0) == 0:
        return None, [], [], []
    return BM25Okapi(tokenized), ids, docs, metas


def semantic_search(
    question: str, k: int = CANDIDATES_PER_METHOD, repo: str | None = None
) -> list[dict]:
    """Dense vector search over the indexed chunks (ChromaDB cosine)."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []
    where = {"repo": collection_safe_id(repo)} if repo else None
    result = collection.query(
        query_texts=[question], n_results=min(k, total), where=where
    )
    hits = []
    for cid, doc, meta, dist in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append({
            "id": cid,
            "text": doc,
            "repo": meta["repo"],
            "file": meta["file"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "semantic_score": 1 - dist,  # cosine similarity
            "keyword_score": None,
        })
    return hits


def keyword_search(
    question: str, k: int = CANDIDATES_PER_METHOD, repo: str | None = None
) -> list[dict]:
    """Sparse BM25 full-text search; exact identifiers rank high."""
    safe_repo = collection_safe_id(repo) if repo else None
    bm25, ids, docs, metas = _keyword_index(safe_repo, _index_version())
    if bm25 is None:
        return []
    query_tokens = _tokenize(question)
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    hits = []
    for i in order:
        if scores[i] <= 0: 
            break
        meta = metas[i]
        hits.append({
            "id": ids[i],
            "text": docs[i],
            "repo": meta["repo"],
            "file": meta["file"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "semantic_score": None,
            "keyword_score": float(scores[i]),
        })
    return hits


def neighbor_chunks(
    repo: str,
    file: str,
    start_line: int,
    end_line: int,
    before: int = 1,
    after: int = 1,
) -> list[str]:
    """Texts of the adjacent chunks (same file) for richer LLM context.

    Reuses the cached BM25 corpus, so this costs no extra ChromaDB calls.
    Line ranges overlap slightly by design, so adjacency is decided by the
    nearest span strictly above/below the hit rather than exact line equality.
    """
    safe_repo = collection_safe_id(repo)
    _, _, docs, metas = _keyword_index(safe_repo, _index_version())
    spans = [
        (meta["start_line"], meta["end_line"], doc)
        for doc, meta in zip(docs, metas)
        if meta["file"] == file
    ]
    if not spans:
        return []

    prev_doc, next_doc = None, None
    best_prev_end, best_next_start = -1, 10 ** 9
    for s, e, doc in spans:
        if e < start_line and e > best_prev_end:
            best_prev_end, prev_doc = e, doc
        if s > end_line and s < best_next_start:
            best_next_start, next_doc = s, doc

    picked = []
    if before and prev_doc is not None:
        picked.append(prev_doc)
    if after and next_doc is not None:
        picked.append(next_doc)
    return picked


def search(
    question: str,
    k: int = CANDIDATES_PER_METHOD,
    repo: str | None = None,
    use_hybrid: bool = True,
) -> list[dict]:
    """Run semantic (+ keyword) retrieval concurrently and fuse the rankings."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        semantic_future = pool.submit(semantic_search, question, k, repo)
        keyword_future = (
            pool.submit(keyword_search, question, k, repo) if use_hybrid else None
        )
        semantic_hits = semantic_future.result()
        keyword_hits = keyword_future.result() if keyword_future else []
    return _fuse(semantic_hits, keyword_hits)


def _fuse(semantic_hits: list[dict], keyword_hits: list[dict]) -> list[dict]:
    """Weighted Reciprocal Rank Fusion of both rankings into one candidate list."""
    fused: dict[str, dict] = {}
    for rank, hit in enumerate(semantic_hits):
        entry = fused.setdefault(hit["id"], dict(hit))
        entry["semantic_rank"] = rank
        if hit.get("semantic_score") is not None:
            entry["semantic_score"] = hit["semantic_score"]
    for rank, hit in enumerate(keyword_hits):
        entry = fused.setdefault(hit["id"], dict(hit))
        entry["keyword_rank"] = rank
        if hit.get("keyword_score") is not None:
            entry["keyword_score"] = hit["keyword_score"]
    for entry in fused.values():
        entry.setdefault("semantic_rank", None)
        entry.setdefault("keyword_rank", None)
        score = 0.0
        if entry["semantic_rank"] is not None:
            score += W_SEMANTIC / (RRF_K + entry["semantic_rank"] + 1)
        if entry["keyword_rank"] is not None:
            score += W_KEYWORD / (RRF_K + entry["keyword_rank"] + 1)
        entry["fused_score"] = score
    return sorted(fused.values(), key=lambda h: h["fused_score"], reverse=True)


@lru_cache(maxsize=1)
def _reranker():
   
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def rerank(question: str, hits: list[dict], top_n: int) -> list[dict]:
    """Re-score candidates with a local cross-encoder; falls back to fused order."""
    if not hits:
        return []
    try:
        model = _reranker()
        pairs = [[question, hit["text"]] for hit in hits]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
    except Exception:
        for hit in hits:
            hit["rerank_score"] = None
        return hits[:top_n]
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
    ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:top_n]


def hybrid_retrieve(
    question: str,
    n_results: int = 8,
    repo: str | None = None,
    use_hybrid: bool = True,
    use_reranker: bool = True,
) -> list[dict]:
    """Full retrieval pipeline: parallel candidates -> RRF fusion -> cross-encoder."""
    question = question.strip()
    if not question:
        return []
    candidates = search(question, repo=repo, use_hybrid=use_hybrid)
    if not candidates:
        return []
    if use_reranker:
        return rerank(question, candidates[:RERANK_MAX_CANDIDATES], n_results)
    for hit in candidates:
        hit["rerank_score"] = None
    return candidates[:n_results]
