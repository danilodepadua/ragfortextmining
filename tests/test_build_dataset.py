import json

import pytest

import build_dataset as bd
from common import write_jsonl


def make_doc(i, n_words=700, span_words=(260, 270)):
    words = [f"d{i}w{k}" for k in range(n_words)]
    context = " ".join(words)
    span = " ".join(words[span_words[0] : span_words[1]])
    start = context.find(span)
    return {
        "doc_id": f"D{i:03d}",
        "document_id": f"doc-{i}",
        "context": context,
        "question": f"Pergunta {i}?",
        "answer": "Resposta completa.",
        "evidence_span": span,
        "evidence_char_start": start,
        "evidence_char_end": start + len(span),
    }


def write_selected(data_dir, docs):
    write_jsonl(data_dir / "raw" / "selected_documents.jsonl", docs)
    log = [
        {"type": "run", "dataset": "d", "revision": "r", "seed": 42, "shuffle_buffer": 10, "judge_model": "m"},
        {"type": "candidate", "judge_meta": {"model_version": "fake-1"}},
    ]
    write_jsonl(data_dir / "curation_log.jsonl", log)
    write_jsonl(data_dir / "raw" / "accepted_pool.jsonl", docs)
    write_jsonl(data_dir / "corpus_v2.jsonl", [{"doc_id": d["doc_id"]} for d in docs])
    (data_dir / "review.md").write_text("review", encoding="utf-8")


def test_chunks_carry_only_chunk_fields_and_match_their_slice():
    doc = make_doc(1)
    chunks = bd.build_chunks([doc])
    assert len(chunks) == 3
    for c in chunks:
        assert set(c) == bd.CHUNK_KEYS
        assert c["text"] == doc["context"][c["char_start"] : c["char_end"]]
        assert doc["question"] not in c["text"] and doc["answer"] not in c["text"]


def test_gold_is_every_chunk_containing_the_span():
    # words 260..270 fall in chunk 0 (0-300) and chunk 1 (250-550): the overlap region
    doc = make_doc(1, span_words=(260, 270))
    chunks = bd.build_chunks([doc])
    assert bd.gold_chunk_ids(doc, chunks) == [chunks[0]["chunk_id"], chunks[1]["chunk_id"]]
    # words 100..110 are only in chunk 0
    doc = make_doc(1, span_words=(100, 110))
    chunks = bd.build_chunks([doc])
    assert bd.gold_chunk_ids(doc, chunks) == [chunks[0]["chunk_id"]]
    # a span that straddles the end of chunk 0 is contained by chunk 1 only
    doc = make_doc(1, span_words=(290, 310))
    chunks = bd.build_chunks([doc])
    assert bd.gold_chunk_ids(doc, chunks) == [chunks[1]["chunk_id"]]


def test_gold_fails_loudly_when_no_chunk_contains_the_span():
    doc = make_doc(1, span_words=(100, 110))
    chunks = bd.build_chunks([doc])
    doc["evidence_char_end"] = len(doc["context"]) + 5
    with pytest.raises(ValueError, match="no chunk contains"):
        bd.gold_chunk_ids(doc, chunks)


def test_gold_ignores_chunks_of_other_documents():
    docs = [make_doc(1, span_words=(100, 110)), make_doc(2, span_words=(100, 110))]
    chunks = bd.build_chunks(docs)
    eval_set = bd.build_eval_set(docs, chunks)
    assert all(g.startswith("chk_D001") for g in eval_set[0]["gold_chunk_ids"])
    assert all(g.startswith("chk_D002") for g in eval_set[1]["gold_chunk_ids"])
    assert [r["qa_id"] for r in eval_set] == ["Q001", "Q002"]


def test_paired_questions_are_reproducible_and_documented():
    eval_set = [{"qa_id": f"Q{i:03d}"} for i in range(1, 26)]
    first, second = bd.build_paired(eval_set), bd.build_paired(list(reversed(eval_set)))
    assert first == second  # sorted before sampling, so input order does not matter
    assert len(set(first["paired"])) == 5 and first["highlighted"] == first["paired"][:3]
    assert first["seed"] == 42 and "sample(sorted(qa_ids), 5)" in first["method"]


def test_build_check_and_manifest_end_to_end(tmp_path):
    docs = [make_doc(i, span_words=(100 + i, 110 + i)) for i in range(1, 22)]
    write_selected(tmp_path, docs)
    bd.build(tmp_path)
    assert bd.check(tmp_path) == []
    eval_rows = [json.loads(line) for line in (tmp_path / "eval_set_v2.jsonl").read_text().splitlines()]
    assert isinstance(eval_rows[0]["gold_chunk_ids"], list)
    csv_text = (tmp_path / "eval_set_v2.csv").read_text(encoding="utf-8")
    assert '"[""chk_D001_000_' in csv_text  # list is JSON-encoded in the CSV

    first = (tmp_path / "chunks_v2.jsonl").read_bytes()
    bd.build(tmp_path)
    assert (tmp_path / "chunks_v2.jsonl").read_bytes() == first  # deterministic

    result = bd.manifest(tmp_path)
    assert result["revision"] == "r" and result["judge_model_versions"] == ["fake-1"]
    assert result["judge_models"] == ["m"]
    assert set(result["files"]) == set(bd.V2_FILES)
    assert all(len(h) == 64 for h in result["files"].values())


def test_check_reports_problems(tmp_path):
    docs = [make_doc(i, span_words=(100 + i, 110 + i)) for i in range(1, 22)]
    write_selected(tmp_path, docs)
    bd.build(tmp_path)

    chunks = [json.loads(line) for line in (tmp_path / "chunks_v2.jsonl").read_text().splitlines()]
    chunks[0]["metadata"] = {"question": "vazou"}
    chunks[1]["text"] = chunks[1]["text"] + " extra"
    write_jsonl(tmp_path / "chunks_v2.jsonl", chunks)
    problems = " | ".join(bd.check(tmp_path))
    assert "unexpected keys" in problems
    assert "not the slice" in problems
    assert "different result" in problems


def test_check_flags_a_wrong_document_count(tmp_path):
    docs = [make_doc(i, span_words=(100 + i, 110 + i)) for i in range(1, 6)]
    write_selected(tmp_path, docs)
    bd.build(tmp_path)
    assert any("expected 20 to 30" in p for p in bd.check(tmp_path))


def _built(tmp_path, n=21):
    docs = [make_doc(i, span_words=(100 + i, 110 + i)) for i in range(1, n + 1)]
    write_selected(tmp_path, docs)
    bd.build(tmp_path)
    return docs


def test_check_detects_a_tampered_gold_set_or_question(tmp_path):
    _built(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "eval_set_v2.jsonl").read_text().splitlines()]
    other = next(r for r in rows if r["doc_id"] == "D002")["gold_chunk_ids"]
    rows[0]["gold_chunk_ids"] = other  # points at a chunk of another document
    rows[0]["question"] = "TAMPERED"
    write_jsonl(tmp_path / "eval_set_v2.jsonl", rows)
    assert any("eval set differs from a rebuild" in p for p in bd.check(tmp_path))


def test_check_detects_a_stale_csv_or_paired_file(tmp_path):
    _built(tmp_path)
    (tmp_path / "eval_set_v2.csv").write_text("stale", encoding="utf-8")
    (tmp_path / "paired_questions_v2.json").write_text("{}", encoding="utf-8")
    problems = " | ".join(bd.check(tmp_path))
    assert "eval_set_v2.csv differs" in problems and "paired_questions_v2.json differs" in problems


def test_check_flags_a_span_that_is_not_unique_in_its_document(tmp_path):
    docs = [make_doc(i, span_words=(100 + i, 110 + i)) for i in range(1, 22)]
    repeated = " ".join(["igual"] * 700)  # every window of this text occurs many times, overlapping
    span = " ".join(["igual"] * 10)
    docs[0].update(context=repeated, evidence_span=span, evidence_char_start=0, evidence_char_end=len(span))
    write_selected(tmp_path, docs)
    bd.build(tmp_path)
    assert any("missing or not unique" in p for p in bd.check(tmp_path))


def test_manifest_refuses_to_run_when_check_fails(tmp_path):
    _built(tmp_path)
    (tmp_path / "eval_set_v2.csv").write_text("stale", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to write a manifest"):
        bd.manifest(tmp_path)


def test_manifest_covers_the_human_review_decision_when_present(tmp_path):
    _built(tmp_path)
    (tmp_path / "human_review.json").write_text('{"rejected": []}', encoding="utf-8")
    assert "human_review.json" in bd.manifest(tmp_path)["files"]
