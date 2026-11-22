"""Curate the 25-document PT-BR corpus from Madras1/rag-qa-fulltext-ptbr.

Two steps, run from the repository root:

    python src/curate.py collect    # stream the dataset, filter, judge with Gemini (network)
    python src/curate.py finalize   # offline: apply data/human_review.json, write the corpus files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from common import DATA, count_occurrences, normalize_text, read_jsonl, words_touched, write_jsonl

DATASET = "Madras1/rag-qa-fulltext-ptbr"
REVISION = "a9546f0d67416f3f2df57007f0ae60fde0cc0fab"
SEED = 42
SHUFFLE_BUFFER = 10_000

# Pinned after listing the models available to the key (override with --model). Aliases resolve silently
# (asking for gemini-3.5-flash was served by gemini-3.6-flash), so every judge call
# also logs the returned model_version.
DEFAULT_MODEL = "gemini-3.6-flash"

LANGUAGE = "pt-br"
DOMAIN = "tecnologia"
DOC_TYPES = ("tutorial", "manual", "guia", "faq")
MIN_WORDS = 400  # at least 2 chunks of 300 words
MAX_WORDS = 1500
MAX_SPAN_WORDS = 50  # equal to the chunk overlap, so a span always fits inside one chunk
MIN_ANSWER_CHARS = 8
CODE_TOKENS_PER_100_WORDS = 5

TARGET = 25
RESERVE = 5

# "segundo o texto", "de acordo com o artigo", "no texto", "neste documento", ...
_DEICTIC = re.compile(
    r"\b(segundo|de acordo com|conforme|com base em|baseado em)\s+(o|a)\s+"
    r"(texto|artigo|documento|trecho|passagem)\b"
    r"|\b(n[oa]|d[oa]|neste|nesta|nesse|nessa|presente)\s+"
    r"(texto|artigo|documento|trecho|passagem)\b",
    re.IGNORECASE,
)
_CODE = re.compile(
    r"\b(def|class|import|function|const|return|public|void|printf|SELECT|FROM)\b|[{};]|=>|</?\w+>"
)


class Judgement(BaseModel):
    is_technical: bool
    is_news_opinion_or_marketing: bool
    answer_supported_by_span: bool
    question_self_contained: bool
    reason: str

    @property
    def accepted(self) -> bool:
        return (
            self.is_technical
            and not self.is_news_opinion_or_marketing
            and self.answer_supported_by_span
            and self.question_self_contained
        )


def is_broken_or_empty(text: str) -> bool:
    if not text or not text.strip():
        return True
    if "�" in text:
        return True
    valid = sum(1 for c in text if c.isalnum() or c.isspace())
    return valid / len(text) < 0.60


def is_deictic(question: str) -> bool:
    return bool(_DEICTIC.search(question))


def code_density(text: str) -> float:
    """Code-like tokens per 100 words. Word boundaries keep "salvar" from matching "var"."""
    words = len(text.split())
    return 100 * len(_CODE.findall(text)) / words if words else 0.0


def cheap_reject(row: dict, context: str, doc_types=DOC_TYPES) -> str | None:
    """Reason a row is rejected before any QA or LLM work, or None if it passes."""
    if row.get("language") != LANGUAGE:
        return "language"
    if row.get("domain") != DOMAIN:
        return "domain"
    if row.get("doc_type") not in doc_types:
        return "doc_type"
    words = len(context.split())
    if words < MIN_WORDS:
        return "too_short"
    if words > MAX_WORDS:
        return "too_long"
    if is_broken_or_empty(context):
        return "broken"
    if code_density(context) > CODE_TOKENS_PER_100_WORDS:
        return "code_heavy"
    return None


def select_qa(context: str, qa_pairs: list[dict], used_questions: set[str]):
    """First QA with one untruncated, unique, short, literal evidence span.

    Returns (selection, reasons). `selection` is None when no QA qualifies;
    `reasons` lists why each QA was skipped.
    """
    reasons = []
    for index, qa in enumerate(qa_pairs):
        question = normalize_text(qa.get("question", "")).strip()
        answer = normalize_text(qa.get("answer", "")).strip()
        spans = [normalize_text(s).strip() for s in (qa.get("evidence_spans") or [])]
        if qa.get("evidence_truncated"):
            reasons.append("evidence_truncated")
        elif len(spans) != 1:
            reasons.append("not_single_span")
        elif not spans[0]:
            reasons.append("empty_span")
        elif len(answer) < MIN_ANSWER_CHARS:
            reasons.append("short_answer")
        elif is_deictic(question):
            reasons.append("deictic_question")
        elif question.lower() in used_questions:
            reasons.append("duplicate_question")
        else:
            span = spans[0]
            count = count_occurrences(context, span)
            if count == 0:
                reasons.append("span_not_found")
                continue
            if count > 1:
                reasons.append("span_repeated")
                continue
            start = context.find(span)
            if words_touched(context, start, start + len(span)) > MAX_SPAN_WORDS:
                reasons.append("span_too_long")
                continue
            return {
                "source_qa_index": index,
                "question": question,
                "answer": answer,
                "evidence_span": span,
                "evidence_char_start": start,
                "evidence_char_end": start + len(span),
            }, reasons
    return None, reasons


def judge_prompt(context: str, selection: dict) -> str:
    return (
        "Você avalia um par pergunta-resposta de um conjunto de teste de RAG sobre "
        "documentação técnica em português. Responda apenas com JSON.\n\n"
        "- is_technical: o documento é técnico (tutorial, manual, guia, FAQ, explicação técnica).\n"
        "- is_news_opinion_or_marketing: o documento é notícia, opinião, blog pessoal ou "
        "material promocional/comercial.\n"
        "- answer_supported_by_span: a resposta é sustentada pelo trecho de evidência.\n"
        "- question_self_contained: a pergunta faz sentido sozinha, sem ver o documento.\n"
        "- reason: uma frase curta.\n\n"
        f"Documento (início):\n{context[:4000]}\n\n"
        f"Pergunta: {selection['question']}\n"
        f"Resposta: {selection['answer']}\n"
        f"Trecho de evidência: {selection['evidence_span']}\n"
    )


def _context_hash(context: str) -> str:
    return hashlib.md5(context.strip().lower().encode("utf-8")).hexdigest()


def collect(
    stream,
    client,
    model: str,
    data_dir: Path = DATA,
    doc_types=DOC_TYPES,
    max_rows: int = 50_000,
    max_judge_calls: int = 100,
    sleep: float = 1.0,
    overwrite: bool = False,
) -> list[dict]:
    """Stream rows, filter, judge, and keep the first TARGET + RESERVE accepted documents."""
    from llm import generate_json

    pool_path = data_dir / "raw" / "accepted_pool.jsonl"
    log_path = data_dir / "curation_log.jsonl"
    if not overwrite and (pool_path.exists() or log_path.exists()):
        raise FileExistsError(f"{pool_path} or {log_path} exists: pass --overwrite to replace the selection")

    log: list[dict] = [
        {
            "type": "run",
            "dataset": DATASET,
            "revision": REVISION,
            "seed": SEED,
            "shuffle_buffer": SHUFFLE_BUFFER,
            "judge_model": model,
            "doc_types": list(doc_types),
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
            "max_span_words": MAX_SPAN_WORDS,
        }
    ]
    pool: list[dict] = []
    seen_hashes: set[str] = set()
    seen_questions: set[str] = set()
    rejects: Counter = Counter()
    qa_rejects: Counter = Counter()
    rows = judge_calls = errors_in_a_row = 0

    def flush() -> None:
        summary = {
            "type": "summary",
            "rows_streamed": rows,
            "judge_calls": judge_calls,
            "accepted": len(pool),
            "reject_counts": dict(rejects),
            "qa_reject_counts": dict(qa_rejects),
        }
        write_jsonl(pool_path, pool)
        write_jsonl(log_path, [*log, summary])

    try:
        for row in stream:
            if rows >= max_rows or len(pool) >= TARGET + RESERVE or judge_calls >= max_judge_calls:
                break
            rows += 1
            context = normalize_text(row.get("context", ""))
            reason = cheap_reject(row, context, doc_types)
            if reason is None:
                digest = _context_hash(context)
                if digest in seen_hashes:
                    reason = "duplicate_context"
                else:
                    seen_hashes.add(digest)
            if reason:
                rejects[reason] += 1
                continue
            selection, qa_reasons = select_qa(context, row.get("qa_pairs") or [], seen_questions)
            if selection is None:
                rejects["no_valid_qa"] += 1
                qa_rejects.update(qa_reasons)
                continue

            judge_calls += 1  # counts attempts, so errors also use the budget
            entry = {"type": "candidate", "document_id": row.get("document_id"), "doc_type": row["doc_type"]}
            try:
                judgement, meta = generate_json(client, model, judge_prompt(context, selection), Judgement)
            except Exception as exc:  # network, quota, blocked or schema error: log it, do not hide it
                errors_in_a_row += 1
                entry.update(decision="judge_error", reason=f"{type(exc).__name__}: {exc}")
                log.append(entry)
                if errors_in_a_row >= 5:
                    raise RuntimeError("5 judge calls failed in a row; see data/curation_log.jsonl") from exc
                time.sleep(sleep)
                continue
            errors_in_a_row = 0
            entry.update(
                decision="accepted" if judgement.accepted else "rejected_by_judge",
                judge=judgement.model_dump(),
                judge_meta=meta,
            )
            log.append(entry)
            if judgement.accepted:
                seen_questions.add(selection["question"].lower())
                pool.append(
                    {
                        "document_id": row.get("document_id"),
                        "hf_row_idx": row.get("hf_row_idx"),
                        "domain": row["domain"],
                        "doc_type": row["doc_type"],
                        "word_count": len(context.split()),
                        "context": context,
                        **selection,
                    }
                )
            time.sleep(sleep)
    finally:
        flush()  # also on Ctrl-C or a streaming error: paid judge results are never lost
    return pool


def finalize(data_dir: Path = DATA) -> list[dict]:
    """Offline: drop documents listed in data/human_review.json, keep the first TARGET."""
    pool = read_jsonl(data_dir / "raw" / "accepted_pool.jsonl")
    review_path = data_dir / "human_review.json"
    rejected: set = set()
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        unknown = set(review) - {"rejected"}
        if unknown:
            raise ValueError(f"unknown keys in human_review.json: {sorted(unknown)}")
        rejected = set(review.get("rejected", []))
        missing = rejected - {p["document_id"] for p in pool}
        if missing:
            raise ValueError(f"human_review.json lists ids that are not in the pool: {sorted(missing)}")
    kept = [p for p in pool if p["document_id"] not in rejected]
    if len(kept) < TARGET:
        raise RuntimeError(f"only {len(kept)} documents left after review, need {TARGET}")
    selected = [{"doc_id": f"D{i:03d}", **rec} for i, rec in enumerate(kept[:TARGET], start=1)]

    write_jsonl(data_dir / "raw" / "selected_documents.jsonl", selected)
    write_jsonl(
        data_dir / "corpus_v2.jsonl",
        [
            {
                k: rec[k]
                for k in ("doc_id", "document_id", "hf_row_idx", "word_count", "domain", "doc_type")
            }
            for rec in selected
        ],
    )
    (data_dir / "review.md").write_text(review_markdown(selected), encoding="utf-8")
    return selected


def review_markdown(selected: list[dict]) -> str:
    out = [
        "# Review of the selected documents\n",
        "To reject one, list its `document_id` in `data/human_review.json` "
        '(`{"rejected": ["doc-123"]}`) and run `python src/curate.py finalize` again.\n',
    ]
    for rec in selected:
        words = rec["context"].split()
        start, end = rec["evidence_char_start"], rec["evidence_char_end"]
        around = rec["context"][max(0, start - 150) : end + 150]
        out.append(
            f"\n## {rec['doc_id']} · {rec['document_id']} · {rec['doc_type']} · {rec['word_count']} palavras\n\n"
            f"**Início:** {' '.join(words[:60])} ...\n\n"
            f"**Pergunta:** {rec['question']}\n\n"
            f"**Resposta:** {rec['answer']}\n\n"
            f"**Evidência:** {rec['evidence_span']}\n\n"
            f"**Evidência no contexto:** ...{around}...\n"
        )
    return "".join(out)


def _open_stream():
    from datasets import load_dataset

    ds = load_dataset(DATASET, "documents", split="train", streaming=True, revision=REVISION)
    return iter(ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="stream, filter and judge (needs network and GEMINI_API_KEY)")
    c.add_argument("--include-doc-type", action="append", default=[], help="extra doc_type to accept")
    c.add_argument("--max-rows", type=int, default=50_000)
    c.add_argument("--max-judge-calls", type=int, default=100)
    c.add_argument("--sleep", type=float, default=1.0, help="seconds between judge calls")
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--overwrite", action="store_true", help="replace an existing accepted pool and log")
    sub.add_parser("finalize", help="offline: write the corpus files from the accepted pool")
    args = parser.parse_args()

    if args.command == "collect":
        from llm import make_client

        pool = collect(
            _open_stream(),
            make_client(),
            args.model,
            doc_types=DOC_TYPES + tuple(args.include_doc_type),
            max_rows=args.max_rows,
            max_judge_calls=args.max_judge_calls,
            sleep=args.sleep,
            overwrite=args.overwrite,
        )
        print(f"accepted {len(pool)} documents; see data/curation_log.jsonl")
    else:
        selected = finalize()
        print(f"wrote {len(selected)} documents; read data/review.md before freezing")


if __name__ == "__main__":
    main()
