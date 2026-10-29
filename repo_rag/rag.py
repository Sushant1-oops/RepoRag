import os
import re
import time
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

from . import retrieval

DEFAULT_MODEL = "openai/gpt-oss-120b" 
MAX_CONTEXT_CHARS = 14_000       
NEIGHBORS_PER_HIT = 1             

CITATION_RE = re.compile(r"([\w./\\-]+):(\d+)")
CHUNK_MARKER_RE = re.compile(r"【Chunk (\d+)(?:\.(\d+))?†L(\d+)(?:-L(\d+))?】")

SYSTEM_PROMPT = """You are a precise software-engineering assistant answering questions about a GitHub repository.

You receive numbered context chunks retrieved from the repository. Ground every statement in those chunks.

Rules:
- Answer ONLY from the context. Never invent files, functions, APIs, flags or behavior.
- Cite immediately after each claim as `path/to/file.py:LINE`, using the exact path and line numbers from the chunk headers.
- Never use 【 】 brackets or mention "Chunk N" in the answer — the reader only sees file paths and line numbers.
- If the context is insufficient, say so plainly and state what is missing. Do not guess.
- Quote identifiers, flags and paths exactly as they appear in the context.
- Be concise and structured: short paragraphs and bullet lists; fenced code blocks with a language tag for any code.
- When code chunks and docs disagree, prefer the source code and mention the discrepancy."""

USER_PROMPT = """Repository context ({n_chunks} chunks, best match first):

{context}

Question: {question}

Answer (grounded in the context above, with `file:line` citations):"""

HISTORY_MAX_CHARS = 3_000 


def _history_block(history: list[dict] | None) -> str:
    """Render recent turns so follow-ups keep context (answers stay grounded).

    Filled from the latest exchange backwards, then trimmed to the budget, so
    the newest messages always win.
    """
    if not history:
        return ""
    lines: list[str] = []
    used = 0
    for msg in reversed(history):
        line = f"{msg['role']}: {msg['content']}"
        if used + len(line) > HISTORY_MAX_CHARS:
            break
        lines.append(line)
        used += len(line)
    lines.reverse()
    return (
        "Recent conversation (for continuity; ground every claim in the "
        "repository context below, not in the chat history):\n\n"
        + "\n\n".join(lines)
        + "\n\n---\n\n"
    )


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a .env file with GROQ_API_KEY=... "
            "(get a free key at https://console.groq.com/keys)."
        )
    return Groq(api_key=api_key)


def get_model() -> str:
    load_dotenv()
    return os.environ.get("GROQ_MODEL", DEFAULT_MODEL)


def retrieve(
    question: str,
    n_results: int = 8,
    repo: str | None = None,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    user_id: int | None = None,
) -> list[dict]:
    """Hybrid retrieval: BM25 keyword + vector semantic, fused, then reranked."""
    return retrieval.hybrid_retrieve(
        question,
        n_results=n_results,
        repo=repo,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        user_id=user_id,
    )


def _neighbor_texts(hit: dict, user_id: int | None = None) -> list[str]:
    """Adjacent same-file chunk texts, with the 'File: ...' prefix stripped."""
    repo = hit.get("repo")
    if not repo:
        return []
    texts = retrieval.neighbor_chunks(
        repo, hit["file"], hit["start_line"], hit["end_line"],
        before=NEIGHBORS_PER_HIT, after=NEIGHBORS_PER_HIT, user_id=user_id,
    )
    return [text.split("\n", 1)[-1].strip() for text in texts]


def _context_parts(
    hits: list[dict], user_id: int | None = None
) -> list[str]:
    """Per-hit context blocks: header + chunk text + neighbor sub-blocks.

    Each hit is preceded by a header with the exact file path and line range
    the model must cite. Adjacent chunks from the same file are appended as
    numbered sub-blocks so the model sees the code around each match.
    """
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        blocks = [
            f"[Chunk {i} | {hit['file']}:{hit['start_line']}-{hit['end_line']}]",
            hit["text"].strip(),
        ]
        for j, neighbor in enumerate(_neighbor_texts(hit, user_id), start=1):
            blocks.append(
                f"[Chunk {i}.{j} | {hit['file']} — adjacent code]\n{neighbor}"
            )
        parts.append("\n".join(blocks))
    return parts


def _budgeted_context(
    hits: list[dict],
    max_chars: int = MAX_CONTEXT_CHARS,
    user_id: int | None = None,
) -> tuple[str, list[str]]:
    """Context string plus the exact per-hit blocks that fit the budget."""
    parts: list[str] = []
    used = 0
    for block in _context_parts(hits, user_id):
        if parts and used + len(block) > max_chars:
            break  # keep the best-scoring chunks within the budget
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts), parts


def build_context(
    hits: list[dict], max_chars: int = MAX_CONTEXT_CHARS, user_id: int | None = None
) -> str:
    """Assemble a size-budgeted, neighbor-expanded context block."""
    return _budgeted_context(hits, max_chars, user_id)[0]


def extract_citations(answer: str, hits: list[dict]) -> list[str]:
    """Collect `file:line` references that exist in the retrieved sources.

    Some models emit internal markers like 【Chunk 2†L22-L30】 instead of file
    paths; those are translated back to the real `file:line` of that chunk.
    """
    known_files = {hit["file"] for hit in hits}
    citations: list[str] = []

    def add(ref: str) -> None:
        if ref not in citations:
            citations.append(ref)

    for path, line in CITATION_RE.findall(answer):
        path = path.rstrip(".,;:`")
        if path in known_files:
            add(f"{path}:{line}")
    for match in CHUNK_MARKER_RE.finditer(answer):
        index = int(match.group(1)) - 1
        if 0 <= index < len(hits):
            add(f"{hits[index]['file']}:{match.group(3)}")
    return citations


def stream_answer(
    question: str,
    n_results: int = 8,
    repo: str | None = None,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    history: list[dict] | None = None,
    user_id: int | None = None,
):
    """Full grounded RAG step as a stream of (event, payload) tuples.

    `history` is an optional list of {role, content} dicts (oldest first)
    giving the model continuity for follow-up questions. `user_id` scopes
    retrieval to one account's indexed repos.
    """
    t0 = time.perf_counter()
    hits = retrieve(
        question, n_results=n_results, repo=repo,
        use_hybrid=use_hybrid, use_reranker=use_reranker, user_id=user_id,
    )
    retrieval_ms = int((time.perf_counter() - t0) * 1000)
    yield ("sources", hits)

    if not hits:
        yield ("done", {
            "empty": True,
            "message": "No repository content has been indexed yet. "
                       "Add a GitHub repository in the sidebar first.",
            "retrieval_ms": retrieval_ms,
        })
        return

    context, context_parts = _budgeted_context(hits, user_id=user_id)
    prompt = _history_block(history) + USER_PROMPT.format(
        n_chunks=len(hits), context=context, question=question
    )
    try:
        client = get_groq_client()
        stream = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
            stream=True,
            # the groq SDK's typed create() has no stream_options param; the
            # API supports it via extra_body and reports usage on the last chunk
            extra_body={"stream_options": {"include_usage": True}},
        )
    except Exception as e:
        yield ("error", f"Groq request failed: {e}")
        return

    t_generate = time.perf_counter()
    first_token_ms = None
    parts: list[str] = []
    usage = None
    try:
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage  # final chunk when include_usage is on
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue
            if first_token_ms is None:
                first_token_ms = int((time.perf_counter() - t_generate) * 1000)
            parts.append(delta)
            yield ("token", delta)
    except Exception as e:
        yield ("error", f"Streaming failed mid-answer: {e}")
        return

    answer = "".join(parts)
    yield ("done", {
        "citations": extract_citations(answer, hits),
        "retrieval_ms": retrieval_ms,
        "first_token_ms": first_token_ms,
        "total_ms": int((time.perf_counter() - t0) * 1000),
        "model": get_model(),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "contexts": context_parts,  # exact blocks the model saw (used by eval)
    })


def answer_question(
    question: str,
    n_results: int = 8,
    repo: str | None = None,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    history: list[dict] | None = None,
    user_id: int | None = None,
) -> dict:
    """Non-streaming convenience wrapper around stream_answer."""
    answer_parts, sources = [], []
    result: dict = {"answer": "", "sources": []}
    for event, payload in stream_answer(
        question, n_results=n_results, repo=repo,
        use_hybrid=use_hybrid, use_reranker=use_reranker, history=history,
        user_id=user_id,
    ):
        if event == "sources":
            sources = payload
        elif event == "token":
            answer_parts.append(payload)
        elif event == "done":
            result.update(payload)
        elif event == "error":
            result["error"] = payload
    result["answer"] = result.get("message") or "".join(answer_parts)
    result["sources"] = sources
    return result
