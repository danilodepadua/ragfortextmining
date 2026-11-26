"""Evaluate the frozen retrieval rankings and blind RAG grades offline.

Run from the repository root:

    python src/evaluate.py

Reads data/, rag_models.toml and the saved M2/M3 artifacts. Writes only
runs/m4_evaluation_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from tempfile import NamedTemporaryFile

import rag
from common import DATA, RUNS, read_jsonl, verify_manifest
from retrieve import KS, RETRIEVERS, compute_metrics, first_gold_rank

SUMMARY_FILENAME = "m4_evaluation_summary.json"
GRADE_FIELDS = ("g_id", "grade", "cited_supports", "notes")
GRADES = ("Correct", "Partial", "Wrong")
SUPPORT_VALUES = ("Yes", "No", "N/A")
RRF_K = 60
ANSWER_FIELDS = (
    "g_id", "qa_id", "retriever", "grade", "cited_supports", "answerable",
    "cited_numbers", "cited_chunk_ids", "invalid_cited_numbers", "gold_in_context",
    "cites_gold", "consistent", "latency_ms",
)


def load_grades(path: Path, expected_ids: set[str]) -> dict[str, dict]:
    """Validate the fixed CSV schema and preserve the grader's notes verbatim."""
    grades = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            if next(reader, None) != list(GRADE_FIELDS):
                raise ValueError(f"{path}: expected header {','.join(GRADE_FIELDS)}")
            for fields in reader:
                location = f"{path}:{reader.line_num}"
                if len(fields) != len(GRADE_FIELDS):
                    raise ValueError(f"{location}: expected {len(GRADE_FIELDS)} columns")
                g_id, grade, supports, notes = fields
                if g_id in grades:
                    raise ValueError(f"{location}: duplicate g_id {g_id!r}")
                if g_id not in expected_ids:
                    raise ValueError(f"{location}: unexpected g_id {g_id!r}")
                if grade not in GRADES:
                    raise ValueError(f"{location}: invalid grade {grade!r} for {g_id}")
                if supports not in SUPPORT_VALUES:
                    raise ValueError(f"{location}: invalid cited_supports {supports!r} for {g_id}")
                grades[g_id] = {"grade": grade, "cited_supports": supports, "notes": notes}
        except csv.Error as exc:
            raise ValueError(f"{path}:{reader.line_num}: malformed CSV ({exc})") from exc
    if missing := expected_ids - grades.keys():
        raise ValueError(f"{path}: missing g_id values {sorted(missing)}")
    return grades


def join_grades(run_path: Path, grades_path: Path, jobs: list[dict]) -> list[dict]:
    """Unblind by the logged g_id, validating each slot against the frozen jobs."""
    records = {}
    for record in read_jsonl(run_path):
        g_id = record.get("g_id")
        if g_id in records:
            raise ValueError(f"{run_path}: duplicate g_id {g_id!r}")
        records[g_id] = record
    expected_ids = {job["g_id"] for job in jobs}
    if records.keys() != expected_ids:
        raise ValueError(f"{run_path}: expected one successful run for each of {sorted(expected_ids)}")

    grades = load_grades(grades_path, expected_ids)
    answers = []
    for job in jobs:
        g_id = job["g_id"]
        record, grade = records[g_id], grades[g_id]
        if record["error"] is not None or any(record.get(field) != value for field, value in job.items()):
            raise ValueError(f"{run_path}: {g_id} does not match its frozen paired job")
        if (grade["cited_supports"] == "N/A") != (not record["cited_chunk_ids"]):
            raise ValueError(f"{grades_path}: {g_id} requires cited_supports = N/A exactly when no chunk was cited")
        answers.append({**record, **grade})
    return answers


def reciprocal_rank_fusion(rankings: list[list[dict]]) -> list[dict]:
    """Fuse saved ranks using k = 60 and chunk ID as the stable tie-break."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for item in ranking:
            scores[item["chunk_id"]] += 1 / (RRF_K + item["rank"])
    order = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [{"rank": rank, "chunk_id": chunk_id, "score": scores[chunk_id]} for rank, chunk_id in enumerate(order, 1)]


def document_diagnostics(
    rankings: dict[str, list[dict]], questions: dict[str, dict], chunks: dict[str, dict], gold_ranks: dict[str, int],
) -> dict:
    """Measure document hits and locate the target document on evidence misses."""
    wrong_doc = []
    misses = {str(k): {"target_document_present": [], "target_document_absent": []} for k in KS}
    for qa_id, question in questions.items():
        ranking = rankings[qa_id]
        doc_ids = [chunks[item["chunk_id"]]["doc_id"] for item in ranking[: max(KS)]]
        if doc_ids[0] != question["doc_id"]:
            wrong_doc.append({
                "qa_id": qa_id, "question": question["question"], "gold_doc_id": question["doc_id"],
                "ranked_doc_id": doc_ids[0], "ranked_chunk_id": ranking[0]["chunk_id"], "first_gold_rank": gold_ranks[qa_id],
            })
        for k in KS:
            if gold_ranks[qa_id] > k:
                outcome = "target_document_present" if question["doc_id"] in doc_ids[:k] else "target_document_absent"
                misses[str(k)][outcome].append(qa_id)
    hits = len(questions) - len(wrong_doc)
    return {
        "doc_recall_at_1": hits / len(questions),
        "doc_hits_at_1": hits,
        "wrong_doc_at_1": wrong_doc,
        "evidence_misses": misses,
    }


def evaluate_retrieval(inputs: dict, metrics_path: Path) -> dict:
    questions, chunks, rankings = inputs["eval_set"], inputs["chunks"], inputs["rankings"]
    if rankings.keys() != {(qa_id, name) for qa_id in questions for name in RETRIEVERS}:
        raise ValueError("rankings_v2.jsonl: expected both retrievers for each frozen question")
    for (qa_id, name), line in rankings.items():
        ranking = line["ranking"]
        if (
            line["chunks_sha256"] != inputs["chunks_sha256"]
            or {item["chunk_id"] for item in ranking} != chunks.keys()
            or [item["rank"] for item in ranking] != list(range(1, len(chunks) + 1))
        ):
            raise ValueError(f"rankings_v2.jsonl: invalid ranking for {qa_id}/{name}")

    metrics = compute_metrics(list(rankings.values()), list(questions.values()), len(chunks))
    metrics["chunks_sha256"] = inputs["chunks_sha256"]
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    if any(saved.get(key) != value for key, value in metrics.items()):
        raise ValueError(f"{metrics_path}: metrics disagree with the frozen rankings")

    for name in RETRIEVERS:
        method = metrics["retrievers"][name]
        by_question = {qa_id: rankings[qa_id, name]["ranking"] for qa_id in questions}
        method.update(document_diagnostics(by_question, questions, chunks, method["first_gold_rank"]))

    doc_sizes = Counter(chunk["doc_id"] for chunk in chunks.values())
    metrics["random"]["doc_recall_at_1"] = mean(doc_sizes[q["doc_id"]] / len(chunks) for q in questions.values())
    fused = {
        qa_id: reciprocal_rank_fusion([rankings[qa_id, name]["ranking"] for name in RETRIEVERS])
        for qa_id in questions
    }
    gold_ranks = {
        qa_id: first_gold_rank(fused[qa_id], set(row["gold_chunk_ids"]))
        for qa_id, row in questions.items()
    }
    metrics["rrf"] = {
        "k": RRF_K,
        "evidence_recall_at_k": {str(k): mean(rank <= k for rank in gold_ranks.values()) for k in KS},
        "mrr": mean(1 / rank for rank in gold_ranks.values()),
        "first_gold_rank": gold_ranks,
        **document_diagnostics(fused, questions, chunks, gold_ranks),
    }
    return metrics


def describe(values: list[int]) -> dict:
    return {
        "n": len(values),
        "total": sum(values) if values else None,
        "median": median(values) if values else None,
        "min": min(values, default=None),
        "max": max(values, default=None),
    }


def summarize_answers(answers: list[dict]) -> dict:
    grades = Counter(row["grade"] for row in answers)
    supports = Counter(row["cited_supports"] for row in answers)
    usage = [row["generation"].get("usage") or {} for row in answers]
    token_fields = {
        "prompt": ("prompt_token_count", "prompt_eval_count"),
        "output": ("candidates_token_count", "eval_count"),
        "reported_total": ("total_token_count",),
    }
    return {
        "n": len(answers),
        "grades": {grade: grades[grade] for grade in GRADES},
        "cited_supports": {support: supports[support] for support in SUPPORT_VALUES},
        "correct_without_support": sum(row["grade"] == "Correct" and row["cited_supports"] != "Yes" for row in answers),
        "gold_in_context": sum(row["gold_in_context"] for row in answers),
        "cites_gold": sum(row["cites_gold"] for row in answers),
        "consistent": sum(row["consistent"] for row in answers),
        "invalid_citations": sum(len(row["invalid_cited_numbers"]) for row in answers),
        "latency_ms": describe([row["latency_ms"] for row in answers]),
        "tokens": {
            name: describe([item[key] for item in usage for key in keys if item.get(key) is not None])
            for name, keys in token_fields.items()
        },
    }


def by_retriever(answers: list[dict]) -> dict:
    return {name: summarize_answers([row for row in answers if row["retriever"] == name]) for name in RETRIEVERS}


def evaluate_rag(inputs: dict, jobs: list[dict], runs_dir: Path, config: dict) -> dict:
    models, answers, probes, non_perfect, answer_grades = {}, [], [], [], []
    for model, params in config["models"].items():
        paths = rag.paths_for(runs_dir, model)
        rows = join_grades(paths["runs"], paths["grades"], jobs)
        settings = rag.settings_for(model, rag.K, params)
        for row in rows:
            if (
                row["settings"] != settings
                or row["chunks_sha256"] != inputs["chunks_sha256"]
                or row["rankings_sha256"] != inputs["rankings_sha256"]
            ):
                raise ValueError(f"{paths['runs']}: {row['g_id']} has different model settings or inputs")
            answer = rag.RagAnswer(answerable=row["answerable"], cited=row["cited_numbers"], answer=row["answer"])
            checks = rag.check_answer(answer, [item["chunk_id"] for item in row["retrieved"]], row["gold_chunk_ids"])
            if any(row[field] != value for field, value in checks.items()):
                raise ValueError(f"{paths['runs']}: {row['g_id']} has inconsistent logged citation checks")
            row["model"] = model
            result = {"model": model, **{field: row[field] for field in ANSWER_FIELDS}}
            answer_grades.append(result)
            if row["grade"] != "Correct" or row["cited_supports"] != "Yes" or not row["gold_in_context"] or not row["consistent"]:
                non_perfect.append({**result, "answer": row["answer"], "notes": row["notes"]})

        final_slots = {(row["qa_id"], row["retriever"]) for row in rows}
        attempts = read_jsonl(paths["failures"]) if paths["failures"].exists() else []
        failures = [
            {
                **{field: attempt[field] for field in ("g_id", "qa_id", "retriever", "error", "latency_ms")},
                "recovered": (attempt["qa_id"], attempt["retriever"]) in final_slots,
            }
            for attempt in attempts
        ]
        dev = read_jsonl(paths["dev"]) if paths["dev"].exists() else []
        for row in dev:
            if row["qa_id"] == "Q019" and row["k"] == 1:
                probes.append({
                    "model": model,
                    **{field: row[field] for field in (
                        "qa_id", "retriever", "k", "answerable", "answer",
                        "cited_numbers", "gold_in_context", "cites_gold", "consistent",
                    )},
                    "retrieved_chunk_ids": [item["chunk_id"] for item in row["retrieved"]],
                    "rubric_grade": None if row["answerable"] else "Wrong",
                })
        models[model] = {
            "provider": params["provider"],
            "settings": settings,
            "overall": summarize_answers(rows),
            "by_retriever": by_retriever(rows),
            "highlighted": summarize_answers([row for row in rows if row["highlighted"]]),
            "development_answers": len(dev),
            "failed_attempts": failures,
        }
        answers.extend(rows)

    return {
        "default_models": config["run"],
        "models": models,
        "overall": summarize_answers(answers),
        "by_provider": {
            provider: summarize_answers([row for row in answers if models[row["model"]]["provider"] == provider])
            for provider in rag.PROVIDERS
        },
        "by_retriever": by_retriever(answers),
        "highlighted": summarize_answers([row for row in answers if row["highlighted"]]),
        "answer_grades": answer_grades,
        "non_perfect": non_perfect,
        "development_probe": probes,
    }


def evaluate(data_dir: Path = DATA, runs_dir: Path = RUNS, config_path: Path = rag.CONFIG_PATH) -> dict:
    """Recompute M4 from saved inputs. No provider client is constructed."""
    # rag.load_inputs checks the other three frozen files used by this evaluation.
    verify_manifest(data_dir, "corpus_v2.jsonl")
    inputs = rag.load_inputs(data_dir, runs_dir)
    manifest = json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    documents = read_jsonl(data_dir / "corpus_v2.jsonl")
    chunks = inputs["chunks"]
    paired = inputs["paired"]
    retrieval = evaluate_retrieval(inputs, runs_dir / "retrieval_metrics.json")
    jobs = rag.make_jobs(inputs, paired["paired"], rag.K, assign_g_ids=True)
    return {
        "corpus": {
            "dataset": manifest["dataset"],
            "revision": manifest["revision"],
            "documents": len(documents),
            "chunks": len(chunks),
            "document_types": dict(sorted(Counter(row["doc_type"] for row in documents).items())),
            "document_words": describe([row["word_count"] for row in documents]),
            "chunk_words": describe([row["word_count"] for row in chunks.values()]),
            "chunk_size_words": manifest["chunk_size_words"],
            "chunk_overlap_words": manifest["chunk_overlap_words"],
        },
        "retrieval": retrieval,
        "paired": {
            **paired,
            "slots": [
                {
                    **{field: job[field] for field in ("g_id", "qa_id", "retriever", "highlighted")},
                    "retrieved_chunk_ids": [item["chunk_id"] for item in job["retrieved"]],
                    "first_gold_rank": retrieval["retrievers"][job["retriever"]]["first_gold_rank"][job["qa_id"]],
                }
                for job in jobs
            ],
        },
        "rag": evaluate_rag(inputs, jobs, runs_dir, rag.load_config(config_path)),
    }


def write_summary(path: Path, summary: dict) -> None:
    """Write a deterministic UTF-8 JSON document, atomically replacing the prior one."""
    content = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    stream = NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}.", delete=False,
    )
    tmp = Path(stream.name)
    try:
        with stream:
            stream.write(content)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def format_table(summary: dict) -> str:
    result = summary["rag"]
    lines = [
        f"{result['overall']['n']} blind grades across {len(result['models'])} models",
        "",
        f"{'Model':24} {'Provider':8} {'Correct':>7} {'Partial':>7} {'Wrong':>5} {'Support':>7} "
        f"{'Gold cite':>8} {'Invalid':>7} {'Median s':>8}",
    ]
    for model, detail in result["models"].items():
        stats = detail["overall"]
        lines.append(
            f"{model:24} {detail['provider']:8} {stats['grades']['Correct']:>7} {stats['grades']['Partial']:>7} "
            f"{stats['grades']['Wrong']:>5} {stats['cited_supports']['Yes']:>7} {stats['cites_gold']:>8} "
            f"{stats['invalid_citations']:>7} {stats['latency_ms']['median'] / 1000:>8.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.parse_args()
    try:
        summary = evaluate()
        write_summary(RUNS / SUMMARY_FILENAME, summary)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(format_table(summary))
    print(f"\nWrote runs/{SUMMARY_FILENAME} from saved artifacts (0 API calls).")


if __name__ == "__main__":
    main()
