"""Generate the RAG evaluation test set from the real Qdrant chunk corpus.

Seed files are sampled from the indexed repos (largest/most substantial first,
proportionally across repos); one Groq call per file turns 2-3 real chunks
into self-contained developer questions with reference answers. The source
file path is recorded as ground truth for retrieval metrics (relevant_files)
and the source chunk ids for chunk-level checks.

Output: eval/testset.json
"""

import json
import os
import random
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from dotenv import load_dotenv
from groq import Groq

from repo_rag.ingest import fetch_chunks
OUT_FILE = EVAL_DIR / "testset.json"

TARGET_QUESTIONS = 60          # stays inside the requested 50-100 window
QUESTIONS_PER_FILE = (2, 3)    # random pick per seed file
MAX_FILES = 30
MAX_CHUNKS_PER_PROMPT = 3
MAX_ATTEMPTS = 3


PROMPT = """You are building an evaluation dataset for a code-search RAG system.

Below are real chunks from the file `{file}` in the GitHub repository `{repo}`.

{chunks}

Write {n} realistic questions a developer working with this repository might ask, each answerable ONLY from the content above. Mix styles: "what does X do", "where is Y implemented", "how does Z work", "which function/config handles W". Each question must be self-contained and mention concrete identifiers (function names, types, flags, routes) so it cannot be confused with a question about different code.

For each question, write a concise reference answer (2-4 sentences) grounded ONLY in the chunks above.

Return ONLY a JSON array, no markdown fences, shaped like:
[{{"question": "...", "answer": "..."}}]"""


def groq_client() -> Groq:
    load_dotenv()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set; add it to .env first")
    return Groq(api_key=key)


def pick_chunks(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """First, middle and last chunk of a file to represent its real content."""
    if len(items) <= MAX_CHUNKS_PER_PROMPT:
        return items
    keep = {0, len(items) // 2, len(items) - 1}
    return [items[i] for i in sorted(keep)]


def parse_json_array(raw: str) -> list[dict]:
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON array in response")
    return json.loads(raw[start:end + 1])


def main() -> None:
    random.seed(42)  # reproducible sampling
    client = groq_client()
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    ids, docs, metas = fetch_chunks()
    files: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for cid, doc, meta in zip(ids, docs, metas):
        files.setdefault((meta["repo"], meta["file"]), []).append((cid, doc))
    print(f"corpus: {len(ids)} chunks across {len(files)} files")

    # Rank files by content volume (big files are the core modules), then take
    # a per-repo share of the top files so both repos are covered.
    ranked = sorted(
        files.items(),
        key=lambda kv: sum(len(doc) for _, doc in kv[1]),
        reverse=True,
    )
    per_repo_files: dict[str, list] = {}
    for key, chunks in ranked:
        per_repo_files.setdefault(key[0], []).append((key, chunks))
    total_files = sum(len(v) for v in per_repo_files.values())

    seeds: list[tuple[tuple[str, str], list[tuple[str, str]]]] = []
    for repo, entries in per_repo_files.items():
        share = max(3, round(MAX_FILES * len(entries) / total_files))
        seeds.extend(entries[:share])
    print(f"seed files: {len(seeds)}")

    rows: list[dict] = []
    seen_questions: set[str] = set()
    for (repo, file), chunks in seeds:
        if len(rows) >= TARGET_QUESTIONS:
            break
        n = random.randint(*QUESTIONS_PER_FILE)
        shown = pick_chunks(chunks)
        chunk_block = "\n\n".join(
            f"--- chunk {cid} ---\n{doc.strip()}" for cid, doc in shown
        )
        prompt = PROMPT.format(repo=repo, file=file, chunks=chunk_block, n=n)

        items: list[dict] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=1600,
                )
                items = parse_json_array(resp.choices[0].message.content or "")
                break
            except Exception as e:
                print(f"  attempt {attempt} failed for {file}: {e}")
                time.sleep(4)

        added = 0
        for item in items:
            if len(rows) >= TARGET_QUESTIONS:
                break
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question or not answer or question.lower() in seen_questions:
                continue
            seen_questions.add(question.lower())
            rows.append({
                "qid": f"q{len(rows) + 1:03d}",
                "repo": repo,
                "question": question,
                "ground_truth_answer": answer,
                "relevant_files": [file],
                "source_chunk_ids": [cid for cid, _ in shown],
            })
            added += 1
        print(f"{repo} :: {file} -> +{added} (total {len(rows)})")
        time.sleep(1.2)  # stay friendly to the free-tier rate limit

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator_model": model,
        "n_questions": len(rows),
        "repos": sorted({row["repo"] for row in rows}),
        "questions": rows,
    }
    OUT_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(rows)} questions -> {OUT_FILE}")


if __name__ == "__main__":
    main()
