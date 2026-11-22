import json
from types import SimpleNamespace

import pytest

import rag
from common import read_jsonl, sha256_file, write_jsonl

CHUNK_IDS = [f"chk_D001_{i:03d}_{i:08x}" for i in range(6)]
CHUNK_TEXTS = [f"Texto do trecho {i} sobre o assunto {i}." for i in range(6)]
QUESTIONS = {f"Q{i:03d}": f"Pergunta número {i}?" for i in range(1, 5)}
GOLD = {"Q001": CHUNK_IDS[0], "Q002": CHUNK_IDS[1], "Q003": CHUNK_IDS[2], "Q004": CHUNK_IDS[3]}


def ranking(first, bury_gold_at=None):
    """All six chunks, `first` leading; optionally push the gold chunk down to a position."""
    order = [first] + [c for c in CHUNK_IDS if c != first]
    if bury_gold_at is not None:
        order.remove(first)
        order.insert(bury_gold_at, first)
    return [{"rank": r + 1, "chunk_id": c, "score": round(1 - r / 10, 6)} for r, c in enumerate(order)]


def make_world(tmp_path):
    data, runs = tmp_path / "data", tmp_path / "runs"
    chunks = [{"chunk_id": cid, "doc_id": "D001", "text": text} for cid, text in zip(CHUNK_IDS, CHUNK_TEXTS)]
    eval_set = [
        {"qa_id": q, "question": QUESTIONS[q], "gold_answer": f"Referência {q}", "evidence_span": f"Evidência {q}",
         "gold_chunk_ids": [GOLD[q]]}
        for q in QUESTIONS
    ]
    write_jsonl(data / "chunks_v2.jsonl", chunks)
    write_jsonl(data / "eval_set_v2.jsonl", eval_set)
    (data / "paired_questions_v2.json").write_text(
        json.dumps({"paired": ["Q001", "Q002"], "highlighted": ["Q001"]}), encoding="utf-8"
    )
    files = {n: sha256_file(data / n) for n in ("chunks_v2.jsonl", "eval_set_v2.jsonl", "paired_questions_v2.json")}
    (data / "MANIFEST.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    lines = []
    for q in QUESTIONS:
        lines.append({"qa_id": q, "retriever": "tfidf", "query_text": QUESTIONS[q], "config": {"v": "tfidf"},
                      "ranking": ranking(GOLD[q])})
        # dense buries the gold chunk of Q002 at position 4 (outside k = 3)
        lines.append({"qa_id": q, "retriever": "dense", "query_text": QUESTIONS[q], "config": {"v": "dense"},
                      "ranking": ranking(GOLD[q], bury_gold_at=3 if q == "Q002" else None)})
    write_jsonl(runs / "rankings_v2.jsonl", lines)
    return data, runs


class FakeClient:
    def __init__(self, reply=None, fail_with=None):
        self.models = SimpleNamespace(generate_content=self._call)
        self.prompts = []
        self.reply = reply or {"answerable": True, "cited": [1], "answer": "Resposta do modelo."}
        self.fail_with = fail_with

    def _call(self, model, contents, config):
        self.prompts.append(contents)
        if self.fail_with:
            raise self.fail_with
        return SimpleNamespace(text=json.dumps(self.reply), model_version="fake-1", usage_metadata=None)


def never_a_client():
    raise AssertionError("the client factory must not be called")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(rag.time, "sleep", lambda s: None)


# ---------------------------------------------------------------- prompt and checks


def test_prompt_numbers_the_chunks_in_order_and_ends_with_the_question():
    prompt = rag.build_prompt("Qual é a resposta?", ["primeiro texto", "segundo texto"])
    assert prompt.index("[1]\nprimeiro texto") < prompt.index("[2]\nsegundo texto")
    assert prompt.rstrip().endswith("Pergunta: Qual é a resposta?")
    assert "APENAS" in prompt and '"answerable": false' in prompt


def test_prompt_version_follows_the_template_and_the_schema(monkeypatch):
    before = rag.prompt_version()
    assert before == rag.prompt_version() and len(before) == 12
    monkeypatch.setattr(rag, "PROMPT_TEMPLATE", rag.PROMPT_TEMPLATE + "Seja breve.\n")
    assert rag.prompt_version() != before


def test_check_answer_maps_numbers_to_ids_and_flags_problems():
    ids, gold = ["a", "b", "c"], ["b"]
    ok = rag.check_answer(rag.RagAnswer(answerable=True, cited=[2, 2], answer="x"), ids, gold)
    assert ok["cited_chunk_ids"] == ["b"] and ok["cites_gold"] and ok["gold_in_context"] and ok["consistent"]
    assert ok["invalid_cited_numbers"] == [] and ok["cited_numbers"] == [2, 2]

    wrong = rag.check_answer(rag.RagAnswer(answerable=True, cited=[1, 7], answer="x"), ids, gold)
    assert wrong["cited_chunk_ids"] == ["a"] and wrong["invalid_cited_numbers"] == [7]
    assert not wrong["cites_gold"] and wrong["gold_in_context"] and not wrong["consistent"]

    assert not rag.check_answer(rag.RagAnswer(answerable=True, cited=[], answer="x"), ids, gold)["consistent"]
    assert not rag.check_answer(rag.RagAnswer(answerable=False, cited=[1], answer="x"), ids, gold)["consistent"]
    refusal = rag.check_answer(rag.RagAnswer(answerable=False, cited=[], answer="não sei"), ids, gold)
    assert refusal["consistent"] and not refusal["cites_gold"]

    missing = rag.check_answer(rag.RagAnswer(answerable=True, cited=[1], answer="x"), ["a", "c"], ["b"])
    assert missing["gold_in_context"] is False


# ---------------------------------------------------------------- jobs


def test_jobs_use_the_top_k_chunks_in_rank_order_with_one_template(tmp_path):
    data, runs = make_world(tmp_path)
    inputs = rag.load_inputs(data, runs)
    jobs = rag.make_jobs(inputs, ["Q001", "Q002"], 3, assign_g_ids=True)
    assert [(j["qa_id"], j["retriever"]) for j in jobs if j["qa_id"] == "Q001"] != []
    by = {(j["qa_id"], j["retriever"]): j for j in jobs}
    tfidf = by[("Q002", "tfidf")]
    assert [r["chunk_id"] for r in tfidf["retrieved"]][0] == GOLD["Q002"] and len(tfidf["retrieved"]) == 3
    top = [CHUNK_TEXTS[CHUNK_IDS.index(r["chunk_id"])] for r in tfidf["retrieved"]]
    assert tfidf["prompt_text"] == rag.build_prompt(QUESTIONS["Q002"], top)
    dense = by[("Q002", "dense")]  # the gold chunk is outside its top 3
    assert GOLD["Q002"] not in [r["chunk_id"] for r in dense["retrieved"]]
    template_only = lambda job: job["prompt_text"].split("Trechos:")[0]
    assert template_only(tfidf) == template_only(dense)
    assert tfidf["highlighted"] is False and by[("Q001", "tfidf")]["highlighted"] is True


def test_g_ids_are_stable_and_hide_the_retriever_order(tmp_path):
    data, runs = make_world(tmp_path)
    inputs = rag.load_inputs(data, runs)
    first = rag.make_jobs(inputs, ["Q001", "Q002"], 3, assign_g_ids=True)
    again = rag.make_jobs(inputs, ["Q001", "Q002"], 3, assign_g_ids=True)
    assert [(j["g_id"], j["retriever"]) for j in first] == [(j["g_id"], j["retriever"]) for j in again]
    assert [j["g_id"] for j in first] == ["G01", "G02", "G03", "G04"]
    assert all(j["g_id"] is None for j in rag.make_jobs(inputs, ["Q003"], 3, assign_g_ids=False))


def test_inputs_are_checked_against_the_manifest_and_the_questions(tmp_path):
    data, runs = make_world(tmp_path)
    (data / "eval_set_v2.jsonl").write_text((data / "eval_set_v2.jsonl").read_text() + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="eval_set_v2.jsonl does not match"):
        rag.load_inputs(data, runs)

    data, runs = make_world(tmp_path / "other")
    lines = read_jsonl(runs / "rankings_v2.jsonl")
    lines[0]["query_text"] = "outra pergunta?"
    write_jsonl(runs / "rankings_v2.jsonl", lines)
    with pytest.raises(RuntimeError, match="different question"):
        rag.load_inputs(data, runs)


# ---------------------------------------------------------------- final run


def test_final_run_logs_everything_and_writes_a_blind_sheet(tmp_path):
    data, runs = make_world(tmp_path)
    client = FakeClient()
    summary = rag.run(data, runs, lambda: client, pause=0)
    assert summary["calls"] == 4 and summary["failed"] == []

    records = read_jsonl(runs / "m3_runs.jsonl")
    assert len(records) == 4
    for r in records:
        assert r["error"] is None and r["k"] == 3 and len(r["retrieved"]) == 3
        assert r["settings"] == rag.settings_for(rag.DEFAULT_MODEL, 3) and r["prompt_text"] in client.prompts
        assert r["generation"]["model_version"] == "fake-1" and r["generation"]["raw_text"]
        assert r["chunks_sha256"] == sha256_file(data / "chunks_v2.jsonl")
        assert r["rankings_sha256"] == sha256_file(runs / "rankings_v2.jsonl")
        assert {"head", "code_dirty", "rag_sha256", "llm_sha256", "google_genai"} <= set(r["source"])
        assert {"cites_gold", "gold_in_context", "consistent", "cited_chunk_ids", "answer", "g_id"} <= set(r)
    dense_q2 = next(r for r in records if r["qa_id"] == "Q002" and r["retriever"] == "dense")
    assert dense_q2["gold_in_context"] is False and dense_q2["cites_gold"] is False

    sheet = (runs / "m3_grading_sheet.md").read_text(encoding="utf-8")
    assert "tfidf" not in sheet and "dense" not in sheet  # blind
    assert all(f"## G0{i} " in sheet for i in range(1, 5)) and "(highlighted)" in sheet
    assert "Referência Q001" in sheet and "Evidência Q001" in sheet
    assert CHUNK_TEXTS[0] in sheet  # the cited chunk text is shown
    assert b"\r" not in (runs / "m3_grading_sheet.md").read_bytes()


def test_a_complete_run_is_idempotent_and_makes_no_calls(tmp_path):
    data, runs = make_world(tmp_path)
    rag.run(data, runs, FakeClient, pause=0)
    before = {p.name: p.read_bytes() for p in runs.iterdir()}
    summary = rag.run(data, runs, never_a_client, pause=0)
    assert summary["calls"] == 0
    assert {p.name: p.read_bytes() for p in runs.iterdir()} == before


def test_a_failed_call_is_logged_and_only_that_one_is_retried(tmp_path):
    data, runs = make_world(tmp_path)

    class FailsOnce(FakeClient):
        def _call(self, model, contents, config):
            if len(self.prompts) == 1:
                self.prompts.append(contents)
                return SimpleNamespace(text="{truncated", model_version="fake-1", usage_metadata=None)
            return super()._call(model, contents, config)

    summary = rag.run(data, runs, FailsOnce, pause=0)
    assert summary["calls"] == 4 and len(summary["failed"]) == 1
    assert not (runs / "m3_grading_sheet.md").exists()  # no sheet until every answer exists
    failed = next(r for r in read_jsonl(runs / "m3_runs.jsonl") if r["error"])
    assert "invalid JSON" in failed["error"] and failed["generation"]["raw_text"] == "{truncated"

    retry = FakeClient()
    summary = rag.run(data, runs, lambda: retry, pause=0)
    assert summary["calls"] == 1 and summary["failed"] == []
    assert len(read_jsonl(runs / "m3_runs.jsonl")) == 4 and (runs / "m3_grading_sheet.md").exists()
    # the failed attempt is not lost when the retry overwrites its row in the log
    kept = read_jsonl(runs / "m3_runs_failures.jsonl")
    assert len(kept) == 1 and kept[0]["generation"]["raw_text"] == "{truncated"
    assert not list(runs.glob("*.tmp"))  # atomic writes leave no temp file behind


def test_it_refuses_to_continue_with_a_different_prompt_or_settings(tmp_path, monkeypatch):
    data, runs = make_world(tmp_path)
    rag.run(data, runs, FakeClient, pause=0)
    monkeypatch.setattr(rag, "PROMPT_TEMPLATE", rag.PROMPT_TEMPLATE + "Seja breve.\n")
    with pytest.raises(RuntimeError, match="delete the file to start over"):
        rag.run(data, runs, never_a_client, pause=0)
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="delete the file to start over"):
        rag.run(data, runs, never_a_client, pause=0, model="another-model")


def test_it_pauses_between_calls_only(tmp_path, monkeypatch):
    data, runs = make_world(tmp_path)
    sleeps = []
    monkeypatch.setattr(rag.time, "sleep", sleeps.append)
    rag.run(data, runs, FakeClient, pause=7)
    assert sleeps == [7, 7, 7]  # 4 calls, none before the first


# ---------------------------------------------------------------- development runs


def test_dev_runs_refuse_paired_questions(tmp_path):
    data, runs = make_world(tmp_path)
    with pytest.raises(ValueError, match="must not use paired questions"):
        rag.run(data, runs, never_a_client, dev_ids=["Q003", "Q001"])
    assert not (runs / "m3_dev_runs.jsonl").exists()


def test_dev_runs_use_their_own_log_allow_another_k_and_write_no_sheet(tmp_path):
    data, runs = make_world(tmp_path)
    summary = rag.run(data, runs, FakeClient, pause=0, dev_ids=["Q003", "Q004"], dev_k=1)
    assert summary["calls"] == 4 and summary["out"].name == "m3_dev_runs.jsonl"
    records = read_jsonl(runs / "m3_dev_runs.jsonl")
    assert {len(r["retrieved"]) for r in records} == {1} and all(r["g_id"] is None for r in records)
    assert not (runs / "m3_runs.jsonl").exists() and not (runs / "m3_grading_sheet.md").exists()


def test_five_failures_in_a_row_abort_but_keep_the_log(tmp_path):
    data, runs = make_world(tmp_path)
    extra = [{"qa_id": f"Q9{i}", "question": f"Extra {i}?", "gold_answer": "r", "evidence_span": "e",
              "gold_chunk_ids": [CHUNK_IDS[0]]} for i in range(3)]
    write_jsonl(data / "eval_set_v2.jsonl", read_jsonl(data / "eval_set_v2.jsonl") + extra)
    files = json.loads((data / "MANIFEST.json").read_text())["files"]
    files["eval_set_v2.jsonl"] = sha256_file(data / "eval_set_v2.jsonl")
    (data / "MANIFEST.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    lines = read_jsonl(runs / "rankings_v2.jsonl")
    for e in extra:
        for retriever in ("tfidf", "dense"):
            lines.append({"qa_id": e["qa_id"], "retriever": retriever, "query_text": e["question"],
                          "config": {}, "ranking": ranking(CHUNK_IDS[0])})
    write_jsonl(runs / "rankings_v2.jsonl", lines)

    boom = RuntimeError("429 quota")
    with pytest.raises(RuntimeError, match="5 calls failed in a row"):
        rag.run(data, runs, lambda: FakeClient(fail_with=boom), pause=0, dev_ids=["Q003", "Q004", "Q90", "Q91", "Q92"])
    records = read_jsonl(runs / "m3_dev_runs.jsonl")
    assert len(records) == 5 and all("429 quota" in r["error"] for r in records)


def test_cli_rejects_dev_k_without_dev(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rag.py", "--dev-k", "1"])
    with pytest.raises(SystemExit):
        rag.main()


@pytest.mark.parametrize("value", ["", ",", " "])
def test_cli_rejects_an_empty_dev_list_instead_of_running_the_paired_set(monkeypatch, value):
    monkeypatch.setattr("sys.argv", ["rag.py", "--dev", value])
    monkeypatch.setattr(rag, "run", lambda **kw: pytest.fail("must not start any run"))
    with pytest.raises(SystemExit):
        rag.main()


def test_it_refuses_a_log_made_from_other_rankings(tmp_path):
    data, runs = make_world(tmp_path)
    rag.run(data, runs, FakeClient, pause=0)
    path = runs / "m3_runs.jsonl"
    records = read_jsonl(path)
    records[0]["rankings_sha256"] = "0" * 64
    write_jsonl(path, records)
    with pytest.raises(RuntimeError, match="delete the file"):
        rag.run(data, runs, never_a_client, pause=0)
