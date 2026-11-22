"""Build chunks, gold chunks and the evaluation set from the curated documents.

Run from the repository root:

    python src/build_dataset.py build      # write chunks_v2, eval_set_v2, paired_questions_v2
    python src/build_dataset.py check      # validate the files on disk (and that a rebuild is identical)
    python src/build_dataset.py manifest   # write data/MANIFEST.json (sha256 of every v2 file)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from pathlib import Path

from common import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    DATA,
    chunk_spans,
    count_occurrences,
    read_jsonl,
    sha256_file,
    stable_chunk_id,
    write_jsonl,
)

PAIRED_SEED = 42
PAIRED_COUNT = 5
HIGHLIGHT_COUNT = 3
CHUNK_KEYS = {"chunk_id", "doc_id", "chunk_index", "char_start", "char_end", "word_count", "text"}
V2_FILES = (
    "raw/accepted_pool.jsonl",
    "raw/selected_documents.jsonl",
    "corpus_v2.jsonl",
    "chunks_v2.jsonl",
    "eval_set_v2.jsonl",
    "eval_set_v2.csv",
    "paired_questions_v2.json",
    "curation_log.jsonl",
    "review.md",
)


def build_chunks(selected: list[dict]) -> list[dict]:
    chunks = []
    for doc in selected:
        for span in chunk_spans(doc["context"], CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS):
            text = doc["context"][span["char_start"] : span["char_end"]]
            chunks.append(
                {
                    "chunk_id": stable_chunk_id(doc["doc_id"], span["chunk_index"], text),
                    "doc_id": doc["doc_id"],
                    **span,
                    "text": text,
                }
            )
    return chunks


def gold_chunk_ids(doc: dict, chunks: list[dict]) -> list[str]:
    """Chunks of the document that contain the whole evidence span."""
    start, end = doc["evidence_char_start"], doc["evidence_char_end"]
    ids = [
        c["chunk_id"]
        for c in chunks
        if c["doc_id"] == doc["doc_id"] and c["char_start"] <= start and end <= c["char_end"]
    ]
    if not ids:
        raise ValueError(f"{doc['doc_id']}: no chunk contains the evidence span")
    return ids


def build_eval_set(selected: list[dict], chunks: list[dict]) -> list[dict]:
    return [
        {
            "qa_id": f"Q{i:03d}",
            "doc_id": doc["doc_id"],
            "question": doc["question"],
            "gold_answer": doc["answer"],
            "evidence_span": doc["evidence_span"],
            "evidence_char_start": doc["evidence_char_start"],
            "evidence_char_end": doc["evidence_char_end"],
            "gold_chunk_ids": gold_chunk_ids(doc, chunks),
        }
        for i, doc in enumerate(selected, start=1)
    ]


def build_paired(eval_set: list[dict]) -> dict:
    qa_ids = sorted(row["qa_id"] for row in eval_set)
    paired = random.Random(PAIRED_SEED).sample(qa_ids, PAIRED_COUNT)
    return {
        "seed": PAIRED_SEED,
        "method": f"random.Random(seed).sample(sorted(qa_ids), {PAIRED_COUNT})",
        "paired": paired,
        "highlighted": paired[:HIGHLIGHT_COUNT],
    }


def _csv_text(eval_set: list[dict]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(eval_set[0]))
    writer.writeheader()
    for row in eval_set:
        writer.writerow({**row, "gold_chunk_ids": json.dumps(row["gold_chunk_ids"])})
    return out.getvalue()


def _paired_text(eval_set: list[dict]) -> str:
    return json.dumps(build_paired(eval_set), indent=2) + "\n"


def build(data_dir: Path = DATA) -> None:
    selected = read_jsonl(data_dir / "raw" / "selected_documents.jsonl")
    chunks = build_chunks(selected)
    eval_set = build_eval_set(selected, chunks)
    write_jsonl(data_dir / "chunks_v2.jsonl", chunks)
    write_jsonl(data_dir / "eval_set_v2.jsonl", eval_set)
    (data_dir / "eval_set_v2.csv").write_text(_csv_text(eval_set), encoding="utf-8", newline="")
    (data_dir / "paired_questions_v2.json").write_text(_paired_text(eval_set), encoding="utf-8")


def check(data_dir: Path = DATA) -> list[str]:
    """Return the list of problems found (empty when everything holds)."""
    problems: list[str] = []
    selected = read_jsonl(data_dir / "raw" / "selected_documents.jsonl")
    chunks = read_jsonl(data_dir / "chunks_v2.jsonl")
    eval_set = read_jsonl(data_dir / "eval_set_v2.jsonl")
    docs = {d["doc_id"]: d for d in selected}

    if not 20 <= len(selected) <= 30:
        problems.append(f"{len(selected)} documents, expected 20 to 30")
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        problems.append("duplicate chunk ids")
    for c in chunks:
        if set(c) != CHUNK_KEYS:
            problems.append(f"{c.get('chunk_id')}: unexpected keys {sorted(set(c) ^ CHUNK_KEYS)}")
        doc = docs.get(c["doc_id"])
        if doc is None or c["text"] != doc["context"][c["char_start"] : c["char_end"]]:
            problems.append(f"{c['chunk_id']}: text is not the slice of its document")
    chunk_ids = set(ids)
    for row in eval_set:
        doc = docs.get(row["doc_id"])
        if doc is None:
            problems.append(f"{row['qa_id']}: unknown document")
            continue
        span = doc["context"][row["evidence_char_start"] : row["evidence_char_end"]]
        if span != row["evidence_span"] or count_occurrences(doc["context"], span) != 1:
            problems.append(f"{row['qa_id']}: evidence span is missing or not unique")
        if not row["gold_chunk_ids"]:
            problems.append(f"{row['qa_id']}: no gold chunk")
        if not set(row["gold_chunk_ids"]) <= chunk_ids:
            problems.append(f"{row['qa_id']}: gold chunk id does not exist")
    if len(eval_set) != len(selected):
        problems.append("eval set and document counts differ")

    rebuilt = build_chunks(selected)
    if rebuilt != chunks:
        problems.append("rebuilding the chunks gives a different result")
    try:
        rebuilt_eval = build_eval_set(selected, rebuilt)
    except ValueError as exc:
        problems.append(str(exc))
    else:
        if rebuilt_eval != eval_set:
            problems.append("eval set differs from a rebuild (gold chunks, question or answer changed)")
        if _paired_text(rebuilt_eval) != _read_text(data_dir / "paired_questions_v2.json"):
            problems.append("paired_questions_v2.json differs from a rebuild")
        if _csv_text(rebuilt_eval) != _read_text(data_dir / "eval_set_v2.csv"):
            problems.append("eval_set_v2.csv differs from a rebuild")
    return problems


def _read_text(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def manifest(data_dir: Path = DATA) -> dict:
    problems = check(data_dir)
    if problems:
        raise RuntimeError("refusing to write a manifest, check() found: " + "; ".join(problems))
    log = read_jsonl(data_dir / "curation_log.jsonl")
    run = next(line for line in log if line["type"] == "run")
    versions = sorted(
        {
            line["judge_meta"]["model_version"]
            for line in log
            if line.get("judge_meta") and line["judge_meta"].get("model_version")
        }
    )
    files = list(V2_FILES)
    if (data_dir / "human_review.json").exists():
        files.append("human_review.json")
    result = {
        "dataset": run["dataset"],
        "revision": run["revision"],
        "seed": run["seed"],
        "shuffle_buffer": run["shuffle_buffer"],
        "judge_model": run["judge_model"],
        "judge_model_versions": versions,
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "files": {name: sha256_file(data_dir / name) for name in files},
    }
    (data_dir / "MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["build", "check", "manifest"])
    command = parser.parse_args().command
    if command == "build":
        build()
        print("wrote chunks_v2, eval_set_v2 and paired_questions_v2")
    elif command == "check":
        problems = check()
        for p in problems:
            print("PROBLEM:", p)
        raise SystemExit(1 if problems else 0)
    else:
        manifest()
        print("wrote data/MANIFEST.json")


if __name__ == "__main__":
    main()
