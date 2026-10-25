"""End-to-end RAG pipeline evaluation over eval/testset.json.

Phases (each checkpointed to eval/*.json; existing artifacts are reused
unless --force is passed):
  retrieval -> file-level Recall/Precision/MRR/NDCG @4/@8/@16 vs ground truth
  answers   -> full pipeline (hybrid retrieve + rerank + Groq), sequential so
               latency percentiles reflect real single-user usage
  citations -> deterministic citation checks (grounded in retrieval / gold)
  ragas     -> faithfulness / answer_relevancy / answer_correctness with an
               LLM judge (Groq) + local embeddings
  report    -> eval/report.md + eval/summary.json

Run: .venv/Scripts/python.exe eval/run_eval.py [--force] [--skip-ragas]

Provider quotas: Groq's free tier caps each model at ~200K tokens/day, so the
answers phase checkpoints every row and waits out 429 windows (the API states
how long to wait) instead of burning the remaining questions.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from repo_rag import ingest, rag, retrieval  # noqa: E402

# The judge is a different model from the answering model (openai/gpt-oss-120b) to
# avoid same-family self-preference bias and to use a separate quota. Ragas needs
# ~15-25 short strict-JSON calls per answer, which free routes cannot sustain:
# Groq enforces an output-tokens-per-minute cap, and OpenRouter's :free routes
# returned upstream 400/429/502 errors or reasoning-token bloat. The verified
# working judge is a paid OpenRouter route (~$1-2 for a full run), selected with:
#   RAGAS_JUDGE_MODEL=google/gemma-4-31b-it
#   RAGAS_JUDGE_BASE_URL=https://openrouter.ai/api/v1
#   RAGAS_JUDGE_KEY_ENV=OPENROUTER_API_KEY
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "qwen/qwen3.8-27b")
JUDGE_BASE_URL = os.environ.get("RAGAS_JUDGE_BASE_URL", "https://api.groq.com/openai/v1")
JUDGE_KEY_ENV = os.environ.get("RAGAS_JUDGE_KEY_ENV", "GROQ_API_KEY")
KS = (4, 8, 16)      # cut-offs for the retrieval metric curves
TOP_K = 8            # production default used for answer generation
ANSWER_DELAY_SECONDS = float(os.environ.get("ANSWER_DELAY_SECONDS", "2"))
ANSWER_ATTEMPTS = int(os.environ.get("ANSWER_ATTEMPTS", "8"))
# Faithfulness is judged against the context the model actually saw, so these
# caps only bound pathological blocks: dropping chunks or tails would make
# well-grounded claims look unsupported and deflate the score.
RAGAS_MAX_CONTEXTS = int(os.environ.get("RAGAS_MAX_CONTEXTS", "8"))
RAGAS_MAX_CONTEXT_CHARS = int(os.environ.get("RAGAS_MAX_CONTEXT_CHARS", "2400"))
RAGAS_WORKERS = int(os.environ.get("RAGAS_WORKERS", "1"))
# Groq caps qwen on output tokens per minute, and faithfulness emits long
# outputs, so the judge runs single-threaded with short timeouts and plenty of
# retries: waiting a minute for the per-minute window is faster than stalling.
JUDGE_TIMEOUT_SECONDS = float(os.environ.get("RAGAS_JUDGE_TIMEOUT", "60"))
JUDGE_RETRIES = int(os.environ.get("RAGAS_JUDGE_RETRIES", "8"))
# Groq rejects a request whose reserved output exceeds its per-minute budget, and
# the OpenAI client reserves the model's whole window when max_tokens is unset.
# Judge responses are short verdict lists, so an explicit cap fits the window.
JUDGE_MAX_TOKENS = int(os.environ.get("RAGAS_JUDGE_MAX_TOKENS", "1200"))
# Ragas wraps every metric job in RunConfig.timeout (180s by default) and kills
# it with a bare TimeoutError, which shows up as a None score. A judge retrying
# through per-minute quota windows legitimately needs longer than that, so the
# budget is raised here instead of losing the row.
JUDGE_JOB_TIMEOUT = float(os.environ.get("RAGAS_JUDGE_JOB_TIMEOUT", "900"))
RAGAS_SAMPLE = int(os.environ.get("RAGAS_SAMPLE", "0"))  # 0 = score every answer
RAGAS_BATCH = int(os.environ.get("RAGAS_BATCH", "6"))
# Judge spend is quota-bound, so one invocation stops at this budget and the
# next run resumes from ragas_partial.json instead of re-scoring.
RAGAS_TIME_BUDGET = float(os.environ.get("RAGAS_TIME_BUDGET", "900"))
METRICS = tuple(m.strip() for m in os.environ.get(
    # faithfulness asks the judge to echo every atomic claim, which needs well
    # over 1000 output tokens per call; on a free Groq tier with an
    # output-tokens-per-minute cap of 1000 that metric cannot be scored, and the
    # models small enough to fit do not follow the strict verdict format. Groundedness
    # is then covered by the deterministic citation checks instead.
    "RAGAS_METRICS", "faithfulness,answer_relevancy,answer_correctness"
).split(","))
# The first queries of a fresh process load the embedding + reranker models, so
# latency is also reported with that startup cost filtered out.
RETRIEVAL_WARMUP = 5


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def drop_if_stale(source: Path, derived: list) -> None:
    """Delete checkpoints that are not strictly newer than `source`.

    A derived artifact written in the same second as its source is treated as
    stale: citations and Ragas scores must never describe an older answer set.
    """
    for path in derived:
        if path.exists() and source.exists() and not (
            path.stat().st_mtime > source.stat().st_mtime
        ):
            path.unlink()
            print(f"stale checkpoint removed: {path.name}")


def percentile(values: list, pct: float):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


# ---------------------------------------------------------------- retrieval

def file_sequence(hits: list[dict]) -> list[str]:
    """Distinct files in retrieval order (rank = first chunk occurrence)."""
    seen, out = set(), []
    for hit in hits:
        if hit["file"] not in seen:
            seen.add(hit["file"])
            out.append(hit["file"])
    return out


def retrieval_metrics(ranked_files: list[str], relevant: list[str], k: int):
    """File-level Recall@k / Precision@k / MRR@k / NDCG@k (binary relevance)."""
    rel = set(relevant)
    top = ranked_files[:k]
    in_top = sum(1 for f in top if f in rel)
    recall = in_top / len(rel) if rel else 0.0
    precision = in_top / k if k else 0.0
    mrr = 0.0
    for rank, f in enumerate(ranked_files[:k], start=1):
        if f in rel:
            mrr = 1.0 / rank
            break
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, f in enumerate(ranked_files[:k], start=1) if f in rel
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    ndcg = dcg / ideal if ideal else 0.0
    return recall, precision, mrr, ndcg


def phase_retrieval(testset: dict, force: bool) -> list[dict]:
    out = EVAL_DIR / "retrieval_results.json"
    if out.exists() and not force:
        print("retrieval: reusing retrieval_results.json")
        return load_json(out, [])
    questions = testset["questions"]
    rows = []
    for i, q in enumerate(questions):
        t0 = time.perf_counter()
        hits = retrieval.hybrid_retrieve(q["question"], n_results=max(KS), repo=None)
        elapsed = int((time.perf_counter() - t0) * 1000)
        ranked = file_sequence(hits)
        row = {
            "qid": q["qid"],
            "ranked_files": ranked,
            "retrieval_ms": elapsed,
            "exact_chunk_hit": bool(
                {h["id"] for h in hits} & set(q["source_chunk_ids"])
            ),
        }
        for k in KS:
            rec, prec, mrr, ndcg = retrieval_metrics(ranked, q["relevant_files"], k)
            row[f"recall@{k}"] = rec
            row[f"precision@{k}"] = prec
            row[f"mrr@{k}"] = mrr
            row[f"ndcg@{k}"] = ndcg
        rows.append(row)
        print(f"[retrieval {i + 1}/{len(questions)}] {q['qid']} recall@8={row['recall@8']:.2f}")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


# ------------------------------------------------------------------ answers

def is_failed(res: dict) -> bool:
    return bool(res.get("error")) or bool(res.get("empty")) or not (res.get("answer") or "").strip()


_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?\s*([\d.]+)s")


def retry_wait_seconds(error, attempt: int):
    """Seconds to wait before retrying a failed answer.

    Groq 429s report when the rolling token window frees up ("try again in
    15m42.192s") and that hint is honoured; other transient errors (connection
    resets, empty streams) get exponential backoff. None means not retryable.
    """
    text = str(error or "")
    match = _RETRY_AFTER_RE.search(text)
    if match:
        wait = int(match.group(1) or 0) * 60 + float(match.group(2)) + 5
        return min(wait, 1200)
    if "No API key" in text:
        return None  # configuration problem: waiting cannot help
    return min(60 * (2 ** (attempt - 1)), 900)


def answer_with_backoff(question: str) -> dict:
    """Run one question through the full pipeline, waiting out rate limits."""
    result: dict = {}
    for attempt in range(1, ANSWER_ATTEMPTS + 1):
        t0 = time.perf_counter()
        try:
            result = rag.answer_question(question, n_results=TOP_K, repo=None)
        except Exception as e:  # keep going: failures are part of the score
            result = {"error": f"pipeline exception: {e}"}
        result["wall_ms"] = int((time.perf_counter() - t0) * 1000)
        result["attempts"] = attempt
        result["ok"] = not is_failed(result)
        if result["ok"]:
            return result
        wait = retry_wait_seconds(result.get("error"), attempt)
        if wait is None or attempt == ANSWER_ATTEMPTS:
            return result
        print(f"    attempt {attempt} failed: {str(result.get('error'))[:110]}")
        print(f"    waiting {wait:.0f}s for the provider quota window ...")
        time.sleep(wait)
    return result


def phase_answers(testset: dict, force: bool) -> list[dict]:
    """Sequential by design: concurrency would skew the latency percentiles.

    Completed rows from an earlier run are reused and the checkpoint is
    rewritten after every question, so a quota-limited run resumes cleanly.
    """
    out = EVAL_DIR / "answers.json"
    prior = {}
    if out.exists() and not force:
        prior = {r["qid"]: r for r in load_json(out, []) if not is_failed(r)}
        if prior:
            print(f"answers: reusing {len(prior)} completed rows")
    questions = testset["questions"]
    # Pre-seed every completed row so a checkpoint never drops answers that
    # were paid for in an earlier run; gaps are filled in place.
    results: list = []
    for q in questions:
        cached = prior.get(q["qid"])
        if cached:
            cached.setdefault("ok", True)
        results.append(cached)
    todo = [i for i, r in enumerate(results) if r is None]
    if todo:
        print(f"answers: {len(todo)} question(s) to (re)generate")
    for n, i in enumerate(todo, 1):
        q = questions[i]
        result = answer_with_backoff(q["question"])
        result["qid"] = q["qid"]
        result["question"] = q["question"]
        result["repo_gold"] = q["repo"]
        results[i] = result
        save_json(out, [r for r in results if r is not None])
        status = "ok" if result["ok"] else "FAILED"
        print(f"[answers {n}/{len(todo)}] {q['qid']} {status} "
              f"total={result.get('total_ms')}ms attempts={result['attempts']}")
        time.sleep(ANSWER_DELAY_SECONDS)
    save_json(out, [r for r in results if r is not None])
    return [r for r in results if r is not None]


# ---------------------------------------------------------------- citations

def phase_citations(testset: dict, retrieval_rows: list[dict],
                    answers: list[dict], force: bool) -> list[dict]:
    out = EVAL_DIR / "citation_checks.json"
    if out.exists() and not force:
        print("citations: reusing citation_checks.json")
        return load_json(out, [])
    ids, _docs, metas = ingest.fetch_chunks()
    corpus_files = {m["file"] for m in metas}
    gt = {q["qid"]: set(q["relevant_files"]) for q in testset["questions"]}
    retrieved = {r["qid"]: set(r["ranked_files"]) for r in retrieval_rows}

    rows = []
    for res in answers:
        if is_failed(res):
            continue  # citation quality is scored over questions that got an answer
        citations = res.get("citations") or []
        files = {c.rsplit(":", 1)[0] for c in citations}
        gold = gt.get(res["qid"], set())
        rows.append({
            "qid": res["qid"],
            "n_citations": len(citations),
            "cited_files": sorted(files),
            "valid_corpus": files <= corpus_files,
            "grounded_in_retrieval": files <= retrieved.get(res["qid"], set()),
            "has_gold_citation": bool(files & gold),
            "gold_citation_rate": (len(files & gold) / len(files)) if files else None,
        })
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


# -------------------------------------------------------------------- ragas

def patch_ragas_imports() -> None:
    """ragas 0.4.3 imports Vertex chat model classes that langchain-community
    0.4 no longer ships; ragas only lists them among supported classes, so a
    stub module satisfies the import without downgrading the langchain stack."""
    import types

    name = "langchain_community.chat_models.vertexai"
    try:
        __import__(name)
    except ModuleNotFoundError:
        stub = types.ModuleType(name)

        class ChatVertexAI:  # placeholder, never instantiated by this eval
            pass

        stub.ChatVertexAI = ChatVertexAI
        sys.modules[name] = stub


def phase_ragas(testset: dict, answers: list[dict], skip: bool, force: bool):
    """Score answers with Ragas in resumable batches.

    Judge calls are the slowest and most quota-hungry part of the run, so scores
    accumulate in ragas_partial.json: an interrupted run continues where it
    stopped instead of paying the judge twice for the same answer.
    """
    out = EVAL_DIR / "ragas_scores.json"
    partial_path = EVAL_DIR / "ragas_partial.json"
    if skip:
        return None
    load_dotenv(ROOT / ".env")
    if not os.environ.get(JUDGE_KEY_ENV):
        print(f"ragas: no judge key in ${JUDGE_KEY_ENV} - skipping")
        return None
    print(f"ragas: judge {JUDGE_MODEL} via {JUDGE_BASE_URL} (key from ${JUDGE_KEY_ENV})")

    gt = {q["qid"]: q["ground_truth_answer"] for q in testset["questions"]}
    q_by_qid = {q["qid"]: q["question"] for q in testset["questions"]}
    rows = []
    for res in answers:
        if is_failed(res) or not res.get("contexts"):
            continue
        contexts = [c[:RAGAS_MAX_CONTEXT_CHARS]
                    for c in res["contexts"][:RAGAS_MAX_CONTEXTS]]
        rows.append({
            "qid": res["qid"],
            "user_input": q_by_qid[res["qid"]],
            "response": res["answer"],
            "retrieved_contexts": contexts,
            "reference": gt[res["qid"]],
        })
    if RAGAS_SAMPLE and len(rows) > RAGAS_SAMPLE:
        step = len(rows) / RAGAS_SAMPLE
        rows = [rows[int(i * step)] for i in range(RAGAS_SAMPLE)]

    partial = load_json(partial_path, {})
    if partial.get("judge_model") != JUDGE_MODEL or force:
        partial = {"judge_model": JUDGE_MODEL, "rows": []}

    def scored_fully(row: dict) -> bool:
        return all(isinstance(row.get(name), (int, float)) for name in METRICS)

    scored = {r["qid"]: r for r in partial.get("rows", [])}
    # A flaky route turns failed judge calls into None. A row whose metrics are
    # all None has not actually been scored, and keeping it would freeze the
    # failure and stop later runs from retrying it, so it is dropped here.
    scored = {qid: row for qid, row in scored.items()
              if any(isinstance(row.get(name), (int, float)) for name in METRICS)}
    # Completeness is per metric, not per row: a pass that only asks for one
    # metric must not mark the row finished for the others, otherwise a second
    # pass would skip it and the missing column would stay None forever.
    todo = [row for row in rows if not scored_fully(scored.get(row["qid"], {}))]
    print(f"ragas: {len(scored)} row(s) already scored, {len(todo)} to go")

    if todo:
        patch_ragas_imports()
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (answer_correctness, answer_relevancy,
                                   faithfulness)
        from ragas.run_config import RunConfig

        available = {"faithfulness": faithfulness,
                     "answer_relevancy": answer_relevancy,
                     "answer_correctness": answer_correctness}
        metrics = [available[name] for name in METRICS if name in available]
        print(f"ragas: metrics {', '.join(METRICS)}")

        judge = LangchainLLMWrapper(
            ChatOpenAI(
                model=JUDGE_MODEL,
                base_url=JUDGE_BASE_URL,
                api_key=os.environ[JUDGE_KEY_ENV],
                temperature=0,
                timeout=JUDGE_TIMEOUT_SECONDS,
                max_retries=JUDGE_RETRIES,
                max_tokens=JUDGE_MAX_TOKENS,
            ),
            # Providers reject n>1 ("'n' : number must be at most 1"), which ragas
            # would otherwise send for answer_relevancy's 3 synthetic questions;
            # bypass_n fans that out into 3 single-completion requests.
            bypass_n=True,
        )
        embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ))
        # the metric frame carries the dataset columns but not our qid, so
        # results are joined back by the question text
        qid_by_question = {row["user_input"]: row["qid"] for row in rows}
        started = time.perf_counter()
        for start in range(0, len(todo), RAGAS_BATCH):
            batch = todo[start:start + RAGAS_BATCH]
            samples = [{k: v for k, v in row.items() if k != "qid"}
                       for row in batch]
            print(f"ragas: scoring batch {start // RAGAS_BATCH + 1} "
                  f"({len(batch)} answers) with {JUDGE_MODEL} ...")
            result = evaluate(
                EvaluationDataset.from_list(samples),
                metrics=metrics,
                llm=judge,
                embeddings=embeddings,
                run_config=RunConfig(max_workers=RAGAS_WORKERS, max_retries=15,
                                     max_wait=60,
                                     # quota backoff happens inside one job, so the
                                     # job may outlive ragas' 180s default
                                     timeout=JUDGE_JOB_TIMEOUT),
                show_progress=False,
            )
            unscored = []
            for record in json.loads(result.to_pandas().to_json(orient="records")):
                qid = qid_by_question.get(record.get("user_input"))
                if qid is None:
                    continue
                row_scores = {name: record.get(name) for name in METRICS}
                if not any(isinstance(v, (int, float))
                           for v in row_scores.values()):
                    unscored.append(qid)
                    continue
                # merge, so metrics scored by an earlier pass are kept
                scored[qid] = {**scored.get(qid, {}), "qid": qid, **row_scores}
            partial["rows"] = list(scored.values())
            save_json(partial_path, partial)
            done = sum(1 for row in scored.values() if scored_fully(row))
            print(f"ragas: {done}/{len(rows)} fully scored "
                  f"({time.perf_counter() - started:.0f}s elapsed)"
                  + (f", {len(unscored)} unusable ({', '.join(unscored)})"
                     if unscored else ""))
            if time.perf_counter() - started > RAGAS_TIME_BUDGET:
                print("ragas: time budget reached - re-run to keep scoring")
                break

    records = [scored[row["qid"]] for row in rows if row["qid"] in scored]
    means = {}
    n_by_metric = {}
    for name in METRICS:
        # a reasoning judge sometimes ignores the strict output format, leaving
        # None behind; means are taken over the rows that actually parsed, and
        # that coverage is reported so the average is never read blindly
        vals = [r[name] for r in records
                if isinstance(r.get(name), (int, float))]
        means[name] = round(sum(vals) / len(vals), 4) if vals else None
        n_by_metric[name] = len(vals)
    payload = {"judge_model": JUDGE_MODEL, "n_scored": len(records),
               "n_by_metric": n_by_metric,
               "n_candidates": len(rows), "n_answers": len(answers),
               "contexts_trimmed_to":
                   f"{RAGAS_MAX_CONTEXTS} x {RAGAS_MAX_CONTEXT_CHARS} chars",
               "means": means, "rows": records}
    save_json(out, payload)
    return payload


# ------------------------------------------------------------------- report

def phase_report(testset, retrieval_rows, answers, citations, ragas_payload):
    questions = testset["questions"]
    ok_answers = [r for r in answers if not is_failed(r)]
    failures = [r for r in answers if is_failed(r)]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    retrieval_table = {}
    for k in KS:
        retrieval_table[f"@{k}"] = {
            "recall": mean([r[f"recall@{k}"] for r in retrieval_rows]),
            "precision": mean([r[f"precision@{k}"] for r in retrieval_rows]),
            "mrr": mean([r[f"mrr@{k}"] for r in retrieval_rows]),
            "ndcg": mean([r[f"ndcg@{k}"] for r in retrieval_rows]),
        }
    exact_chunk_recall = mean([1.0 if r["exact_chunk_hit"] else 0.0
                               for r in retrieval_rows])

    latency = {}
    for field in ("retrieval_ms", "first_token_ms", "total_ms"):
        vals = [r[field] for r in ok_answers if isinstance(r.get(field), (int, float))]
        latency[field] = {"p50": percentile(vals, 50), "p95": percentile(vals, 95),
                          "mean": round(mean(vals), 1) if vals else None}

    # warm views: retrieval skips the model-loading questions, and answers keep
    # only those within 3x the median (anything longer is a process cold start)
    warm_retrieval = [r["retrieval_ms"] for r in retrieval_rows[RETRIEVAL_WARMUP:]
                      if isinstance(r.get("retrieval_ms"), (int, float))]
    latency["retrieval_ms_warm"] = {
        "p50": percentile(warm_retrieval, 50),
        "p95": percentile(warm_retrieval, 95),
        "mean": round(mean(warm_retrieval), 1) if warm_retrieval else None,
    }
    totals = sorted(r["total_ms"] for r in ok_answers
                    if isinstance(r.get("total_ms"), (int, float)))
    steady = [v for v in totals if v <= 3 * (percentile(totals, 50) or 0)]
    latency["total_ms_warm"] = {
        "p50": percentile(steady, 50),
        "p95": percentile(steady, 95),
        "mean": round(mean(steady), 1) if steady else None,
    }

    tokens = {
        "prompt_total": sum(r.get("prompt_tokens") or 0 for r in ok_answers),
        "completion_total": sum(r.get("completion_tokens") or 0 for r in ok_answers),
        "prompt_mean": round(mean([r.get("prompt_tokens") for r in ok_answers]), 1),
        "completion_mean": round(mean([r.get("completion_tokens") for r in ok_answers]), 1),
        "answers_with_usage": sum(1 for r in ok_answers if r.get("prompt_tokens")),
    }

    citation_summary = {
        "answers_with_citations": mean(
            [1.0 if c["n_citations"] else 0.0 for c in citations]),
        "citations_per_answer": round(mean([c["n_citations"] for c in citations]), 2),
        "grounded_in_retrieval_rate": mean(
            [1.0 if c["grounded_in_retrieval"] else 0.0 for c in citations]),
        "answers_with_gold_citation": mean(
            [1.0 if c["has_gold_citation"] else 0.0 for c in citations]),
        "gold_citation_rate": mean([c["gold_citation_rate"] for c in citations]),
    }

    # Deterministic agreement with the reference answer: content-word F1. It is
    # computed for every answered question and costs no judge quota, which
    # matters while the LLM judge is rate-limited, and it is reported apart from
    # Ragas because shared wording is not the same as semantic correctness.
    stop = {"the", "and", "for", "that", "this", "with", "from", "are", "was",
            "has", "have", "will", "can", "use", "used", "uses", "which", "when",
            "what", "how", "into", "its", "out", "about", "because", "returns"}
    gt_by_qid = {q["qid"]: q["ground_truth_answer"] for q in questions}

    def content_words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9_]{3,}", text.lower())
                if w not in stop}

    overlaps = []
    for res in ok_answers:
        got = content_words(res.get("answer") or "")
        want = content_words(gt_by_qid.get(res["qid"], ""))
        if not got or not want:
            continue
        hits = len(got & want)
        precision, recall = hits / len(got), hits / len(want)
        if precision and recall:
            overlaps.append(round(2 * precision * recall / (precision + recall), 4))
    reference_overlap = {
        "n": len(overlaps),
        "mean": round(mean(overlaps), 4) if overlaps else None,
        "p50": round(percentile(overlaps, 50), 4) if overlaps else None,
        "min": min(overlaps) if overlaps else None,
        "max": max(overlaps) if overlaps else None,
    }

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_questions": len(questions),
        "repos": testset["repos"],
        "generator_model": testset["generator_model"],
        "answer_model": (ok_answers[0].get("model") if ok_answers else None),
        "failure_rate": round(len(failures) / len(answers), 4) if answers else None,
        "retrieval": {"file_level": retrieval_table,
                      "exact_chunk_recall@16": exact_chunk_recall},
        "ragas": (ragas_payload or {}).get("means"),
        "ragas_judge": (ragas_payload or {}).get("judge_model"),
        "citations": citation_summary,
        "reference_overlap": reference_overlap,
        "latency": latency,
        "tokens": tokens,
    }
    (EVAL_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# RAG pipeline evaluation report", "",
        f"*Generated {summary['generated_at']} — {len(questions)} questions "
        f"across {len(testset['repos'])} repos "
        f"(generator: `{testset['generator_model']}`, "
        f"answers: `{summary['answer_model']}`, "
        f"judge: `{summary['ragas_judge']}`)*", "",
        "## Dataset", "",
        f"- Questions: **{len(questions)}** "
        f"({sum(1 for q in questions if q['repo'] == testset['repos'][0])} "
        f"from {testset['repos'][0]}, "
        f"{sum(1 for q in questions if q['repo'] == testset['repos'][-1])} "
        f"from {testset['repos'][-1]})",
        "- Ground truth: reference answer + the exact file the question was "
        "generated from (retrieval metrics are file-level, binary relevance)",
        f"- Failure rate: **{summary['failure_rate']:.1%}**", "",
        "## Retrieval quality (hybrid + rerank, global scope)", "",
        "| Metric | " + " | ".join(f"@{k}" for k in KS) + " |",
        "| --- | " + " | ".join("---" for _ in KS) + " |",
    ]
    for name in ("recall", "precision", "mrr", "ndcg"):
        lines.append(
            f"| {name.upper()} | "
            + " | ".join(f"{retrieval_table[f'@{k}'][name]:.3f}" for k in KS) + " |")
    lines += [
        f"| Exact source-chunk in top-16 | {exact_chunk_recall:.3f} |", "",
        "## Answer quality (Ragas, LLM judge + embeddings)", "",
    ]
    if ragas_payload:
        lines.append("| Metric | What it measures | Mean | Rows scored |")
        lines.append("| --- | --- | --- | --- |")
        desc = {
            "faithfulness": "claims supported by retrieved context",
            "answer_relevancy": "answer addresses the question",
            "answer_correctness": "agreement with the reference answer",
        }
        for metric, score in ragas_payload["means"].items():
            n = (ragas_payload.get("n_by_metric") or {}).get(metric, 0)
            lines.append(f"| `{metric}` | {desc[metric]} | {score} | {n} / "
                         f"{ragas_payload['n_scored']} |")
        missing = [name for name in desc if name not in ragas_payload["means"]]
        if missing:
            # an unscored metric is stated rather than left out, so the table is
            # never silently read as a complete quality picture
            lines += ["", "*Not LLM-scored: " + ", ".join(f"`{m}`" for m in missing)
                      + " - the free judge tier caps output tokens per minute, and "
                      "a verdict for this metric needs a longer output than that "
                      "window. Groundedness is covered by the deterministic "
                      "citation checks below.*"]
    else:
        lines.append("_Ragas scoring skipped._")
    lines += [
        "", "## Agreement with the reference answer (deterministic, no judge)", "",
        f"- Content-word F1 between answer and reference over "
        f"{reference_overlap['n']} answered questions: mean "
        f"**{reference_overlap['mean']}**, p50 {reference_overlap['p50']}, "
        f"range {reference_overlap['min']}-{reference_overlap['max']}.",
        "- This only shows that the answer talks about the same entities as the "
        "reference; semantic agreement is the Ragas `answer_correctness` row "
        "above. It is reported because it covers every answered question while "
        "the judge covers a quota-limited sample.",
        "", "## Citation accuracy (deterministic)", "",
        f"- Checks cover {len(citations)} of {len(answers)} questions "
        "(questions without an answer are excluded).",
        f"- Answers with at least one `file:line` citation: "
        f"**{citation_summary['answers_with_citations']:.1%}**",
        f"- Mean citations per answer: **{citation_summary['citations_per_answer']}**",
        f"- Citations grounded in retrieved sources: "
        f"**{citation_summary['grounded_in_retrieval_rate']:.1%}**",
        f"- Answers citing the ground-truth file: "
        f"**{citation_summary['answers_with_gold_citation']:.1%}**",
        f"- Cited files that hit the ground-truth file: "
        f"**{citation_summary['gold_citation_rate']}**", "",
        "## Latency (single-user, sequential)", "",
        "| Stage | p50 | p95 | mean |",
        "| --- | --- | --- | --- |",
    ]
    def fmt_ms(value):
        return "-" if value is None else f"{round(value):,} ms"

    for field in ("retrieval_ms", "retrieval_ms_warm", "first_token_ms",
                  "total_ms", "total_ms_warm"):
        s = latency[field]
        label = {"retrieval_ms": "Retrieval (Qdrant + BM25 + rerank)",
                 "retrieval_ms_warm": f"Retrieval, warm (first {RETRIEVAL_WARMUP} skipped)",
                 "first_token_ms": "First token (Groq TTFB)",
                 "total_ms": "Total",
                 "total_ms_warm": "Total, warm (cold starts excluded)"}[field]
        lines.append(f"| {label} | {fmt_ms(s['p50'])} | {fmt_ms(s['p95'])} "
                     f"| {fmt_ms(s['mean'])} |")
    lines += [
        "",
        "_Raw numbers include the process cold start (the first queries of a run",
        "load the embedding + reranker models); the warm rows are steady state._",
        "", "## Token usage", "",
        f"- Prompt tokens: **{tokens['prompt_total']:,}** total "
        f"(~{tokens['prompt_mean']} / answer)",
        f"- Completion tokens: **{tokens['completion_total']:,}** total "
        f"(~{tokens['completion_mean']} / answer)", "",
        "## Notes & caveats", "",
        "- Questions were LLM-generated from real indexed chunks, so ground "
        "truth is exact at the file level but generation is single-file: "
        "cross-file questions are out of scope.",
        "- Retrieval ran with the production default (hybrid + reranker, "
        "global scope over both repos), so precision@k includes cross-repo "
        "noise by design.",
        f"- Judge model `{JUDGE_MODEL}` via {JUDGE_BASE_URL} differs from the "
        "answering model, which avoids same-family self-preference bias; Ragas "
        "scores remain probabilistic LLM judgments, not ground truth. The judge is "
        "a free-tier route selected by remaining quota rather than by "
        "capability, so treat these scores as indicative, not as a quality gate.",
        f"- Judge contexts are the same chunks the answering model saw (capped at "
        f"{(ragas_payload or {}).get('contexts_trimmed_to', 'n/a')}), so "
        "well-grounded claims are not penalised by trimming.",
        f"- Ragas covers a sample: {(ragas_payload or {}).get('n_scored', 0)} of "
        f"{len(answers)} answers scored, because the free judge route is "
        "upstream-rate-limited; retrieval, citation, latency and token metrics "
        "cover every question.",
        "- Answer failures are provider quota errors (HTTP 429 on Groq's "
        "free-tier tokens-per-day cap), not pipeline defects; re-running the "
        "eval resumes only the questions that failed.",
    ]
    (EVAL_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: wrote {EVAL_DIR / 'report.md'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="recompute all phases, ignoring checkpoints")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="skip LLM-judge scoring")
    args = parser.parse_args()

    testset = load_json(EVAL_DIR / "testset.json", None)
    if not testset:
        raise SystemExit("eval/testset.json missing - run generate_testset.py first")

    retrieval_rows = phase_retrieval(testset, args.force)
    if args.force:
        (EVAL_DIR / "ragas_partial.json").unlink(missing_ok=True)
    answers = phase_answers(testset, args.force)
    # downstream checkpoints are only valid for the answers they were built from
    drop_if_stale(EVAL_DIR / "answers.json",
                  [EVAL_DIR / "citation_checks.json",
                   EVAL_DIR / "ragas_scores.json"])
    citations = phase_citations(testset, retrieval_rows, answers, args.force)
    ragas_payload = phase_ragas(testset, answers, args.skip_ragas, args.force)
    summary = phase_report(testset, retrieval_rows, answers, citations, ragas_payload)

    print("\n=== SUMMARY ===")
    print(json.dumps({k: summary[k] for k in
                      ("n_questions", "failure_rate", "retrieval", "ragas",
                       "latency", "tokens")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
