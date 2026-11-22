"""Rank every chunk for every question with TF-IDF and with Gemini embeddings, then score both.

Run from the repository root:

    python src/retrieve.py

Reads data/ (chunks, eval set) and writes, under runs/:
    rankings_v2.jsonl                      full ranking of all chunks per (question, retriever)
    retrieval_metrics.json                 evidence recall@k, MRR, random baseline, paired counts
    embeddings_<model>_<dim>.npz           embedding cache, so reruns need no API key
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common import DATA, RUNS, read_jsonl, sha256_file, write_jsonl
from llm import embed, make_client

# Pre-registered before any result was seen; never tuned on the eval set (25 questions).
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072
DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "QUESTION_ANSWERING"
BATCH_SIZE = 20  # texts per request: keeps one request well under the free-tier tokens-per-minute cap
TFIDF_DESCRIPTION = "TfidfVectorizer(strip_accents='unicode', lowercase=True), fit on the chunks only"

KS = (1, 3, 5)
RETRIEVERS = ("tfidf", "dense")
SCORE_DECIMALS = 6


# ---------------------------------------------------------------- inputs


def load_inputs(data_dir: Path = DATA):
    """Chunks, eval set and the sha256 of the chunks file (checked against the manifest)."""
    chunks_path = data_dir / "chunks_v2.jsonl"
    chunks_sha = sha256_file(chunks_path)
    manifest = json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["files"]["chunks_v2.jsonl"] != chunks_sha:
        raise RuntimeError("chunks_v2.jsonl does not match data/MANIFEST.json")
    return read_jsonl(chunks_path), read_jsonl(data_dir / "eval_set_v2.jsonl"), chunks_sha


# ---------------------------------------------------------------- scoring


def tfidf_scores(chunk_texts: list[str], queries: list[str]) -> tuple[np.ndarray, list[int]]:
    """Cosine scores (queries x chunks) and the indexes of queries with no term in the vocabulary."""
    vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True)
    chunk_matrix = vectorizer.fit_transform(chunk_texts)
    query_matrix = vectorizer.transform(queries)
    scores = (query_matrix @ chunk_matrix.T).toarray().astype(np.float64)
    empty = [i for i in range(len(queries)) if query_matrix[i].nnz == 0]
    return scores, empty


def _unit(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms == 0).any():
        raise ValueError("embedding vectors must be finite and non-zero")
    return vectors / norms


def dense_scores(chunk_vectors: np.ndarray, query_vectors: np.ndarray) -> np.ndarray:
    """Cosine scores (queries x chunks), computed in float64 after normalizing."""
    return _unit(query_vectors) @ _unit(chunk_vectors).T


def rank_row(scores: np.ndarray, chunk_ids: list[str]) -> list[dict]:
    """All chunks, best first. Scores are rounded and ties keep chunk order (a stable key)."""
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    rounded = [round(float(s), SCORE_DECIMALS) + 0.0 for s in scores]  # + 0.0 turns -0.0 into 0.0
    order = sorted(range(len(rounded)), key=lambda i: (-rounded[i], i))
    return [{"rank": r + 1, "chunk_id": chunk_ids[i], "score": rounded[i]} for r, i in enumerate(order)]


# ---------------------------------------------------------------- embedding cache


def cache_path(runs_dir: Path = RUNS) -> Path:
    return runs_dir / f"embeddings_{EMBED_MODEL}_{EMBED_DIM}.npz"


def chunk_key(chunk_id: str) -> str:
    return f"{DOCUMENT_TASK}|chunk:{chunk_id}"  # the chunk id already carries a hash of the text


def query_key(question: str) -> str:
    return f"{QUERY_TASK}|query:{hashlib.sha256(question.encode('utf-8')).hexdigest()}"


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        cache = dict(zip(data["keys"].tolist(), data["vectors"]))
    if any(vector.shape != (EMBED_DIM,) for vector in cache.values()):
        raise RuntimeError(f"{path} does not hold {EMBED_DIM}-dimension vectors: delete it to rebuild")
    return cache


def save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    """Sorted by key, so the same cache is always the same bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(cache)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, keys=np.array(keys), vectors=np.stack([cache[k] for k in keys]))
    tmp.replace(path)


def ensure_embeddings(items, cache, path, client_factory, pause: float, batch_size: int = BATCH_SIZE) -> int:
    """Embed only the keys missing from the cache, in paced batches, saving after every batch.

    `items` is a list of (key, text, task_type). The client is created lazily: a complete
    cache never touches the API or the key. Returns the number of API calls made.
    """
    missing: dict[str, dict[str, str]] = {}
    for key, text, task in items:
        if key not in cache:
            missing.setdefault(task, {})[key] = text
    calls, client = 0, None
    for task, pairs in missing.items():
        keys = list(pairs)
        for start in range(0, len(keys), batch_size):
            batch = keys[start : start + batch_size]
            if client is None:
                client = client_factory()
            if calls:
                time.sleep(pause)
            vectors, _meta = embed(client, EMBED_MODEL, [pairs[k] for k in batch], task, EMBED_DIM)
            for key, vector in zip(batch, vectors):
                cache[key] = vector
            save_cache(path, cache)  # paid results are never lost
            calls += 1
    return calls


# ---------------------------------------------------------------- metrics


def first_gold_rank(ranking: list[dict], gold: set[str]) -> int:
    for item in ranking:
        if item["chunk_id"] in gold:
            return item["rank"]
    raise ValueError("no gold chunk in the ranking")


def random_baseline(gold_counts: list[int], n_chunks: int) -> dict:
    """Exact expectation for a random ranking (hypergeometric), for questions with g gold chunks."""
    recall = {}
    for k in KS:
        # With k >= n_chunks every chunk is returned, so a gold chunk is always found.
        recall[str(k)] = sum(
            1.0 if k >= n_chunks else 1 - math.comb(n_chunks - g, k) / math.comb(n_chunks, k)
            for g in gold_counts
        ) / len(gold_counts)
    mrr = sum(
        sum(math.comb(n_chunks - r, g - 1) / math.comb(n_chunks, g) / r for r in range(1, n_chunks - g + 2))
        for g in gold_counts
    ) / len(gold_counts)
    return {"evidence_recall_at_k": recall, "mrr": mrr}


def compute_metrics(rankings: list[dict], eval_set: list[dict], n_chunks: int) -> dict:
    gold = {e["qa_id"]: set(e["gold_chunk_ids"]) for e in eval_set}
    qa_ids = [e["qa_id"] for e in eval_set]
    ranks: dict[str, dict[str, int]] = {name: {} for name in RETRIEVERS}
    for line in rankings:
        ranks[line["retriever"]][line["qa_id"]] = first_gold_rank(line["ranking"], gold[line["qa_id"]])

    result: dict = {"n_questions": len(qa_ids), "n_chunks": n_chunks, "retrievers": {}}
    for name in RETRIEVERS:
        values = [ranks[name][q] for q in qa_ids]
        result["retrievers"][name] = {
            "evidence_recall_at_k": {str(k): sum(v <= k for v in values) / len(values) for k in KS},
            "mrr": sum(1 / v for v in values) / len(values),
            "first_gold_rank": {q: ranks[name][q] for q in qa_ids},
        }
    result["random"] = random_baseline([len(gold[q]) for q in qa_ids], n_chunks)
    result["paired_counts"] = {}
    for k in KS:
        tfidf_hit = {q for q in qa_ids if ranks["tfidf"][q] <= k}
        dense_hit = {q for q in qa_ids if ranks["dense"][q] <= k}
        result["paired_counts"][str(k)] = {
            "both": len(tfidf_hit & dense_hit),
            "dense_only": len(dense_hit - tfidf_hit),
            "tfidf_only": len(tfidf_hit - dense_hit),
            "neither": len(set(qa_ids) - tfidf_hit - dense_hit),
        }
    return result


def format_table(result: dict) -> str:
    rows = [(name, result["retrievers"][name]) for name in RETRIEVERS] + [("random (exact)", result["random"])]
    lines = ["| retriever | recall@1 | recall@3 | recall@5 | MRR |", "| --- | --- | --- | --- | --- |"]
    for name, r in rows:
        recall = r["evidence_recall_at_k"]
        lines.append(f"| {name} | {recall['1']:.3f} | {recall['3']:.3f} | {recall['5']:.3f} | {r['mrr']:.3f} |")
    lines.append("")
    lines.append("Paired (dense only / tfidf only / both / neither): " + "; ".join(
        f"@{k}: {c['dense_only']}/{c['tfidf_only']}/{c['both']}/{c['neither']}"
        for k, c in result["paired_counts"].items()
    ))
    return "\n".join(lines)


# ---------------------------------------------------------------- pipeline


def run(data_dir: Path = DATA, runs_dir: Path = RUNS, client_factory=make_client, pause: float = 15.0):
    """Score both retrievers. Returns (metrics, number of API calls)."""
    chunks, eval_set, chunks_sha = load_inputs(data_dir)
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    questions = [e["question"] for e in eval_set]

    tfidf, empty = tfidf_scores(texts, questions)
    path = cache_path(runs_dir)
    cache = load_cache(path)
    items = [(chunk_key(c["chunk_id"]), c["text"], DOCUMENT_TASK) for c in chunks]
    items += [(query_key(q), q, QUERY_TASK) for q in questions]
    calls = ensure_embeddings(items, cache, path, client_factory, pause)
    dense = dense_scores(
        np.stack([cache[chunk_key(i)] for i in chunk_ids]),
        np.stack([cache[query_key(q)] for q in questions]),
    )

    versions = {name: metadata.version(name) for name in ("numpy", "scikit-learn", "google-genai")}
    configs = {
        "tfidf": {"vectorizer": TFIDF_DESCRIPTION, "numpy": versions["numpy"], "scikit-learn": versions["scikit-learn"]},
        "dense": {
            "model": EMBED_MODEL,
            "dim": EMBED_DIM,
            "document_task": DOCUMENT_TASK,
            "query_task": QUERY_TASK,
            "numpy": versions["numpy"],
            "google-genai": versions["google-genai"],
        },
    }
    rankings = []
    for index, row in enumerate(eval_set):
        for name, scores in (("tfidf", tfidf), ("dense", dense)):
            rankings.append(
                {
                    "qa_id": row["qa_id"],
                    "retriever": name,
                    "query_text": row["question"],
                    "ranking": rank_row(scores[index], chunk_ids),
                    "config": configs[name],
                    "chunks_sha256": chunks_sha,
                }
            )
    result = compute_metrics(rankings, eval_set, len(chunks))
    result["chunks_sha256"] = chunks_sha
    result["queries_without_known_terms"] = [eval_set[i]["qa_id"] for i in empty]
    if empty:
        print(f"warning: no TF-IDF term in the vocabulary for {result['queries_without_known_terms']}", file=sys.stderr)

    write_jsonl(runs_dir / "rankings_v2.jsonl", rankings)
    (runs_dir / "retrieval_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return result, calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--pause", type=float, default=15.0, help="seconds between embedding requests")
    args = parser.parse_args()
    result, calls = run(pause=args.pause)
    print(format_table(result))
    print(f"\nembedding API calls this run: {calls}")


if __name__ == "__main__":
    main()
