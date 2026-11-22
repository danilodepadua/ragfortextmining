"""Answer the paired questions from retrieved chunks only, once per retriever, and log everything.

Run from the repository root:

    python src/rag.py                    # final run: 5 paired questions x 2 retrievers (10 answers)
    python src/rag.py --dev Q010,Q011    # development run on NON-paired questions only

The final run is idempotent: records that already succeeded with the same prompt and settings are
kept, only missing or failed ones are called, and it refuses to continue if an existing record was
made with a different prompt or settings (delete runs/m3_runs.jsonl to start over). It writes:
    runs/m3_runs.jsonl            one record per (question, retriever): prompt, retrieval, answer, checks
    runs/m3_grading_sheet.md      blind sheet for the human grader (retriever hidden; g_id is the key)
Development runs write runs/m3_dev_runs.jsonl and no sheet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel

from common import DATA, ROOT, RUNS, read_jsonl, sha256_file, verify_manifest, write_jsonl
from llm import GenerationError, generate_json, make_client
from retrieve import RETRIEVERS

# Fixed before any paired question was answered (see the README for the protocol).
K = 3
DEFAULT_MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 0.0
SEED = 42
SHEET_SEED = 42
MAX_ERRORS_IN_A_ROW = 5

PROMPT_TEMPLATE = """Você responde perguntas sobre documentação técnica usando APENAS os trechos numerados abaixo.

Regras:
1. Use somente informações que estejam nos trechos. Não use conhecimento próprio.
2. Se os trechos contêm a resposta, use "answerable": true, liste em "cited" os números dos trechos que sustentam a resposta (apenas números que aparecem abaixo) e escreva a resposta em "answer".
3. Se os trechos não contêm a resposta, use "answerable": false, deixe "cited" vazio e diga em "answer" que o contexto não permite responder.

Trechos:
{chunks}

Pergunta: {question}
"""


class RagAnswer(BaseModel):
    # Field order is generation order: the model decides answerable and cites before it writes.
    answerable: bool
    cited: list[int]
    answer: str


def prompt_version() -> str:
    """Hash of the template and of the output schema: it changes whenever either changes."""
    material = PROMPT_TEMPLATE + json.dumps(RagAnswer.model_json_schema(), sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def build_prompt(question: str, chunk_texts: list[str]) -> str:
    chunks = "\n\n".join(f"[{i}]\n{text}" for i, text in enumerate(chunk_texts, start=1))
    return PROMPT_TEMPLATE.format(chunks=chunks, question=question)


def check_answer(answer: RagAnswer, retrieved_ids: list[str], gold_ids: list[str]) -> dict:
    """Citation checks against what the model was shown (not against the whole corpus)."""
    valid = [n for n in dict.fromkeys(answer.cited) if 1 <= n <= len(retrieved_ids)]
    invalid = [n for n in dict.fromkeys(answer.cited) if n not in valid]
    cited_ids = [retrieved_ids[n - 1] for n in valid]
    return {
        "cited_numbers": list(answer.cited),
        "cited_chunk_ids": cited_ids,
        "invalid_cited_numbers": invalid,
        "cites_gold": bool(set(cited_ids) & set(gold_ids)),
        "gold_in_context": bool(set(retrieved_ids) & set(gold_ids)),
        # an answer with no citation, or a refusal that cites, contradicts itself
        "consistent": (answer.answerable and bool(valid) and not invalid) or (not answer.answerable and not answer.cited),
    }


# ---------------------------------------------------------------- inputs and jobs


def load_inputs(data_dir: Path, runs_dir: Path) -> dict:
    verify_manifest(data_dir, "chunks_v2.jsonl", "eval_set_v2.jsonl", "paired_questions_v2.json")
    rankings_path = runs_dir / "rankings_v2.jsonl"
    eval_set = {e["qa_id"]: e for e in read_jsonl(data_dir / "eval_set_v2.jsonl")}
    rankings = {}
    for line in read_jsonl(rankings_path):
        if line["query_text"] != eval_set[line["qa_id"]]["question"]:
            raise RuntimeError(f"{line['qa_id']}: the ranking was made for a different question")
        rankings[(line["qa_id"], line["retriever"])] = line
    return {
        "chunks": {c["chunk_id"]: c for c in read_jsonl(data_dir / "chunks_v2.jsonl")},
        "eval_set": eval_set,
        "rankings": rankings,
        "paired": json.loads((data_dir / "paired_questions_v2.json").read_text(encoding="utf-8")),
        "chunks_sha256": sha256_file(data_dir / "chunks_v2.jsonl"),
        "rankings_sha256": sha256_file(rankings_path),
    }


def make_jobs(inputs: dict, qa_ids: list[str], k: int, assign_g_ids: bool) -> list[dict]:
    """One job per (question, retriever), in a fixed order. g_ids hide which retriever is which."""
    rng = random.Random(SHEET_SEED)
    highlighted = set(inputs["paired"]["highlighted"])
    jobs, counter = [], 0
    for qa_id in qa_ids:
        order = list(RETRIEVERS)
        if assign_g_ids:
            rng.shuffle(order)
        row = inputs["eval_set"][qa_id]
        for retriever in order:
            line = inputs["rankings"][(qa_id, retriever)]
            retrieved = line["ranking"][:k]
            texts = [inputs["chunks"][r["chunk_id"]]["text"] for r in retrieved]
            counter += 1
            jobs.append(
                {
                    "qa_id": qa_id,
                    "retriever": retriever,
                    "g_id": f"G{counter:02d}" if assign_g_ids else None,
                    "highlighted": qa_id in highlighted,
                    "k": k,
                    "query_text": row["question"],
                    "retrieved": retrieved,
                    "gold_chunk_ids": row["gold_chunk_ids"],
                    "retriever_config": line["config"],
                    "prompt_text": build_prompt(row["question"], texts),
                }
            )
    return jobs


def source_state() -> dict:
    """Which code produced the run: HEAD, whether src/ was dirty, and the hash of the two modules."""

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--", "src")
    return {
        "head": git("rev-parse", "HEAD"),
        "code_dirty": None if status is None else bool(status),
        "rag_sha256": sha256_file(Path(__file__)),
        "llm_sha256": sha256_file(Path(__file__).with_name("llm.py")),
        "google_genai": metadata.version("google-genai"),
    }


# ---------------------------------------------------------------- running


def settings_for(model: str, k: int) -> dict:
    return {"model_requested": model, "temperature": TEMPERATURE, "seed": SEED, "k": k, "prompt_version": prompt_version()}


def call_one(job: dict, client, settings: dict, provenance: dict) -> dict:
    record = {
        "run_id": uuid.uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{key: job[key] for key in ("qa_id", "retriever", "g_id", "highlighted", "k", "query_text", "retrieved", "gold_chunk_ids", "retriever_config", "prompt_text")},
        "settings": settings,
        "error": None,
        **provenance,
    }
    started = time.monotonic()
    try:
        answer, meta = generate_json(client, settings["model_requested"], job["prompt_text"], RagAnswer, TEMPERATURE, SEED)
    except GenerationError as exc:
        record.update(error=str(exc), generation=exc.meta)
    except Exception as exc:  # network, quota or other API error: log it, do not hide it
        record.update(error=f"{type(exc).__name__}: {exc}", generation=None)
    else:
        retrieved_ids = [r["chunk_id"] for r in job["retrieved"]]
        record.update(
            generation=meta,
            answerable=answer.answerable,
            answer=answer.answer,
            **check_answer(answer, retrieved_ids, job["gold_chunk_ids"]),
        )
    record["latency_ms"] = round((time.monotonic() - started) * 1000)
    return record


def run(
    data_dir: Path = DATA,
    runs_dir: Path = RUNS,
    client_factory=make_client,
    model: str = DEFAULT_MODEL,
    pause: float = 4.0,
    dev_ids: list[str] | None = None,
    dev_k: int = K,
) -> dict:
    """Run (or finish) the final run, or a development run. Returns a summary."""
    inputs = load_inputs(data_dir, runs_dir)
    paired = inputs["paired"]["paired"]
    if dev_ids is not None:
        if set(dev_ids) & set(paired):
            raise ValueError(f"--dev must not use paired questions: {sorted(set(dev_ids) & set(paired))}")
        qa_ids, k, out = dev_ids, dev_k, runs_dir / "m3_dev_runs.jsonl"
    else:
        qa_ids, k, out = paired, K, runs_dir / "m3_runs.jsonl"
    jobs = make_jobs(inputs, qa_ids, k, assign_g_ids=dev_ids is None)
    settings = settings_for(model, k)
    provenance = {
        "source": source_state(),
        "chunks_sha256": inputs["chunks_sha256"],
        "rankings_sha256": inputs["rankings_sha256"],
    }

    failures = out.with_name(out.stem + "_failures.jsonl")
    existing = read_jsonl(out) if out.exists() else []
    by_key = {(r["qa_id"], r["retriever"]): r for r in existing}
    todo = []
    for job in jobs:
        old = by_key.get((job["qa_id"], job["retriever"]))
        if old is not None and old["error"] is None:
            same = (
                old["prompt_text"] == job["prompt_text"]
                and old["settings"] == settings
                and old["g_id"] == job["g_id"]
                and old["chunks_sha256"] == provenance["chunks_sha256"]
                and old["rankings_sha256"] == provenance["rankings_sha256"]
            )
            if not same:
                raise RuntimeError(
                    f"{out.name} holds {job['qa_id']}/{job['retriever']} made with a different prompt, settings, "
                    "g_id or input files: delete the file to start over"
                )
        else:
            todo.append(job)
    job_keys = {(j["qa_id"], j["retriever"]) for j in jobs}
    records = {key: rec for key, rec in by_key.items() if key not in job_keys or rec["error"] is None}

    def save() -> None:
        ordered = [records[(j["qa_id"], j["retriever"])] for j in jobs if (j["qa_id"], j["retriever"]) in records]
        others = [r for key, r in records.items() if key not in job_keys]
        write_jsonl(out, others + ordered)

    calls = errors_in_a_row = 0
    client = None
    for job in todo:
        if client is None:
            client = client_factory()
        if calls:
            time.sleep(pause)
        record = call_one(job, client, settings, provenance)
        records[(job["qa_id"], job["retriever"])] = record
        calls += 1
        if record["error"]:  # a retry overwrites the log row; the failed attempt stays here
            with open(failures, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        save()  # paid results are never lost
        errors_in_a_row = errors_in_a_row + 1 if record["error"] else 0
        if errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
            raise RuntimeError(f"{MAX_ERRORS_IN_A_ROW} calls failed in a row; see {out.name}")

    failed = [key for key in job_keys if records.get(key, {}).get("error") is not None]
    if dev_ids is None and not failed:
        final = [records[(j["qa_id"], j["retriever"])] for j in jobs]
        (runs_dir / "m3_grading_sheet.md").write_text(grading_sheet(final, inputs), encoding="utf-8", newline="\n")
    return {"calls": calls, "failed": sorted(failed), "records": len(jobs), "out": out}


# ---------------------------------------------------------------- grading sheet


def grading_sheet(records: list[dict], inputs: dict) -> str:
    """Blind view for the grader: the retriever is not shown (the log's g_id is the key)."""
    lines = [
        "# M3 grading sheet",
        "",
        "Grade each answer G01 to G10 with the fixed rubric in the README and return the grades in",
        "`runs/m3_grades.csv` with the columns `g_id,grade,cited_supports,notes`.",
        "`grade` is Correct, Partial or Wrong. `cited_supports` is Yes, No or N/A (nothing cited).",
        "The two answers to the same question are next to each other, in a random order.",
        "",
    ]
    for r in records:
        star = " (highlighted)" if r["highlighted"] else ""
        gold = inputs["eval_set"][r["qa_id"]]
        cited = "\n\n".join(f"`{i}`\n\n> {inputs['chunks'][i]['text']}" for i in r["cited_chunk_ids"])
        lines += [
            f"## {r['g_id']} · {r['qa_id']}{star}",
            "",
            f"**Pergunta:** {r['query_text']}",
            "",
            f"**Resposta do modelo** (answerable: {str(r['answerable']).lower()}): {r['answer']}",
            "",
            f"**Trechos citados:** {cited if cited else 'nenhum'}",
            "",
            f"**Resposta de referência:** {gold['gold_answer']}",
            "",
            f"**Evidência de referência:** {gold['evidence_span']}",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between calls (15 requests per minute)")
    parser.add_argument("--dev", help="comma-separated NON-paired question ids for a development run")
    parser.add_argument("--dev-k", type=int, default=K, help="chunks in the context for a development run")
    args = parser.parse_args()
    if args.dev_k != K and args.dev is None:
        parser.error("--dev-k only applies to --dev")
    dev_ids = None
    if args.dev is not None:
        dev_ids = list(dict.fromkeys(q.strip() for q in args.dev.split(",") if q.strip()))
        if not dev_ids:
            parser.error("--dev needs at least one question id")
        if args.dev_k < 1:
            parser.error("--dev-k must be at least 1")
    summary = run(model=args.model, pause=args.pause, dev_ids=dev_ids, dev_k=args.dev_k)
    print(f"{summary['records']} answers requested, written to {summary['out'].name}; API calls this run: {summary['calls']}")
    if summary["failed"]:
        print(f"failed: {summary['failed']} (run again to retry them)", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
