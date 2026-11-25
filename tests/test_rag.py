import json
import re
from types import SimpleNamespace

import pytest

import rag
from common import read_jsonl, sha256_file, write_jsonl

MODEL = "fake-model"
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


def files(runs):
    return rag.paths_for(runs, MODEL)


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
    summary = rag.run(MODEL, data, runs, lambda: client, pause=0)
    assert summary["calls"] == 4 and summary["failed"] == []

    records = read_jsonl(files(runs)["runs"])
    assert len(records) == 4
    for r in records:
        assert r["error"] is None and r["k"] == 3 and len(r["retrieved"]) == 3
        assert r["settings"] == rag.settings_for(MODEL, 3, rag.DEFAULT_PARAMS) and r["prompt_text"] in client.prompts
        assert r["generation"]["model_version"] == "fake-1" and r["generation"]["raw_text"]
        assert r["chunks_sha256"] == sha256_file(data / "chunks_v2.jsonl")
        assert r["rankings_sha256"] == sha256_file(runs / "rankings_v2.jsonl")
        assert {"head", "code_dirty", "rag_sha256", "llm_sha256", "google_genai"} <= set(r["source"])
        assert {"cites_gold", "gold_in_context", "consistent", "cited_chunk_ids", "answer", "g_id"} <= set(r)
    dense_q2 = next(r for r in records if r["qa_id"] == "Q002" and r["retriever"] == "dense")
    assert dense_q2["gold_in_context"] is False and dense_q2["cites_gold"] is False

    sheet = files(runs)["sheet"].read_text(encoding="utf-8")
    assert "tfidf" not in sheet and "dense" not in sheet  # blind
    assert all(f"## G0{i} " in sheet for i in range(1, 5)) and "(highlighted)" in sheet
    assert "Referência Q001" in sheet and "Evidência Q001" in sheet
    assert CHUNK_TEXTS[0] in sheet  # the cited chunk text is shown
    assert b"\r" not in files(runs)["sheet"].read_bytes()


def test_a_complete_run_is_idempotent_and_makes_no_calls(tmp_path):
    data, runs = make_world(tmp_path)
    rag.run(MODEL, data, runs, FakeClient, pause=0)
    before = {p.name: p.read_bytes() for p in runs.iterdir()}
    summary = rag.run(MODEL, data, runs, never_a_client, pause=0)
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

    summary = rag.run(MODEL, data, runs, FailsOnce, pause=0)
    assert summary["calls"] == 4 and len(summary["failed"]) == 1
    assert not files(runs)["sheet"].exists()  # no sheet until every answer exists
    failed = next(r for r in read_jsonl(files(runs)["runs"]) if r["error"])
    assert "invalid JSON" in failed["error"] and failed["generation"]["raw_text"] == "{truncated"

    retry = FakeClient()
    summary = rag.run(MODEL, data, runs, lambda: retry, pause=0)
    assert summary["calls"] == 1 and summary["failed"] == []
    assert len(read_jsonl(files(runs)["runs"])) == 4 and files(runs)["sheet"].exists()
    # the failed attempt is not lost when the retry overwrites its row in the log
    kept = read_jsonl(files(runs)["failures"])
    assert len(kept) == 1 and kept[0]["generation"]["raw_text"] == "{truncated"
    assert not list(runs.glob("*.tmp"))  # atomic writes leave no temp file behind


def test_it_refuses_to_continue_with_a_different_prompt_or_settings(tmp_path, monkeypatch):
    data, runs = make_world(tmp_path)
    rag.run(MODEL, data, runs, FakeClient, pause=0)
    monkeypatch.setattr(rag, "PROMPT_TEMPLATE", rag.PROMPT_TEMPLATE + "Seja breve.\n")
    with pytest.raises(RuntimeError, match="delete the file to start over"):
        rag.run(MODEL, data, runs, never_a_client, pause=0)
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="delete the file to start over"):
        rag.run(MODEL, data, runs, never_a_client, params={"temperature": 0.5, "seed": 42}, pause=0)
    with pytest.raises(RuntimeError, match="delete the file to start over"):
        rag.run(MODEL, data, runs, never_a_client, params={"temperature": 0.0, "seed": 1}, pause=0)


def test_it_pauses_between_calls_only(tmp_path, monkeypatch):
    data, runs = make_world(tmp_path)
    sleeps = []
    monkeypatch.setattr(rag.time, "sleep", sleeps.append)
    rag.run(MODEL, data, runs, FakeClient, pause=7)
    assert sleeps == [7, 7, 7]  # 4 calls, none before the first


# ---------------------------------------------------------------- development runs


def test_dev_runs_refuse_paired_questions(tmp_path):
    data, runs = make_world(tmp_path)
    with pytest.raises(ValueError, match="must not use paired questions"):
        rag.run(MODEL, data, runs, never_a_client, dev_ids=["Q003", "Q001"])
    assert not files(runs)["dev"].exists()


def test_dev_runs_use_their_own_log_allow_another_k_and_write_no_sheet(tmp_path):
    data, runs = make_world(tmp_path)
    summary = rag.run(MODEL, data, runs, FakeClient, pause=0, dev_ids=["Q003", "Q004"], dev_k=1)
    assert summary["calls"] == 4 and summary["out"].name == "m3_dev_runs.fake-model.jsonl"
    records = read_jsonl(files(runs)["dev"])
    assert {len(r["retrieved"]) for r in records} == {1} and all(r["g_id"] is None for r in records)
    assert not files(runs)["runs"].exists() and not files(runs)["sheet"].exists()


def test_five_failures_in_a_row_abort_but_keep_the_log(tmp_path):
    data, runs = make_world(tmp_path)
    extra = [{"qa_id": f"Q9{i}", "question": f"Extra {i}?", "gold_answer": "r", "evidence_span": "e",
              "gold_chunk_ids": [CHUNK_IDS[0]]} for i in range(3)]
    write_jsonl(data / "eval_set_v2.jsonl", read_jsonl(data / "eval_set_v2.jsonl") + extra)
    manifest_files = json.loads((data / "MANIFEST.json").read_text())["files"]
    manifest_files["eval_set_v2.jsonl"] = sha256_file(data / "eval_set_v2.jsonl")
    (data / "MANIFEST.json").write_text(json.dumps({"files": manifest_files}), encoding="utf-8")
    lines = read_jsonl(runs / "rankings_v2.jsonl")
    for e in extra:
        for retriever in ("tfidf", "dense"):
            lines.append({"qa_id": e["qa_id"], "retriever": retriever, "query_text": e["question"],
                          "config": {}, "ranking": ranking(CHUNK_IDS[0])})
    write_jsonl(runs / "rankings_v2.jsonl", lines)

    boom = RuntimeError("429 quota")
    with pytest.raises(RuntimeError, match="5 calls failed in a row"):
        rag.run(MODEL, data, runs, lambda: FakeClient(fail_with=boom), pause=0, dev_ids=["Q003", "Q004", "Q90", "Q91", "Q92"])
    records = read_jsonl(files(runs)["dev"])
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
    rag.run(MODEL, data, runs, FakeClient, pause=0)
    path = files(runs)["runs"]
    records = read_jsonl(path)
    records[0]["rankings_sha256"] = "0" * 64
    write_jsonl(path, records)
    with pytest.raises(RuntimeError, match="delete the file"):
        rag.run(MODEL, data, runs, never_a_client, pause=0)


# ---------------------------------------------------------------- model config and per-model files


def write_config(tmp_path, text):
    path = tmp_path / "rag_models.toml"
    path.write_text(text, encoding="utf-8")
    return path


TWO_MODELS = """
run = ["model-a", "model-b"]
[models."model-a"]
temperature = 0.0
seed = 42
pause = 1.0
[models."model-b"]
temperature = 0.7
seed = 7
"""


def test_the_repo_config_loads_and_keeps_the_settings_used_for_the_committed_runs():
    config = rag.load_config(rag.CONFIG_PATH)
    assert config["run"] == ["gemini-3.5-flash-lite"]
    assert config["models"]["gemini-3.5-flash-lite"] == {"provider": "gemini", "temperature": 0.0, "seed": 42, "pause": 4.0}


def test_the_model_names_in_the_rag_docstring_examples_exist_in_the_repo_config():
    config = rag.load_config(rag.CONFIG_PATH)
    examples = re.findall(r"--model\s+(\S+)", rag.__doc__)
    assert examples
    for example in examples:
        assert set(example.split(",")) <= set(config["models"]), example


def test_config_defaults_and_values(tmp_path):
    config = rag.load_config(write_config(tmp_path, TWO_MODELS))
    assert config["run"] == ["model-a", "model-b"]
    assert config["models"]["model-a"] == {"provider": "gemini", "temperature": 0.0, "seed": 42, "pause": 1.0}
    assert config["models"]["model-b"] == {"provider": "gemini", "temperature": 0.7, "seed": 7, "pause": rag.DEFAULT_PAUSE}


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "at least one"),
        ('models = [1]\n', "at least one"),
        ('[models]\nx = 1\n', "must be a table"),
        ('[models.""]\n', "non-empty"),
        ('[models." m"]\n', "non-empty"),
        ('[models."a,b"]\n', "commas"),
        ('[models."m"]\n', "run must be"),
        ('run = ["m"]\n[models."m"]\npause = nan\n', "pause"),
        ('run = ["m"]\n[models."m"]\npause = inf\n', "pause"),
        ('run = ["A", "a"]\n[models."A"]\n[models."a"]\n', "same file name"),
        ('stray = 1\nrun = ["m"]\n[models."m"]\n', "unknown top-level"),
        ('run = ["m"]\n[models."m"]\ntemp = 1\n', "unknown keys"),
        ('run = ["x"]\n[models."m"]\n', "without a .models. table"),
        ('run = []\n[models."m"]\n', "non-empty list"),
        ('run = ["m"]\n[models."m"]\npause = -1\n', "pause"),
        ('run = ["m"]\n[models."m"]\nprovider = "openai"\n', "provider"),
        ('run = ["m"]\n[models."m"]\ntemperature = 3\n', "temperature"),
        ('run = ["m"]\n[models."m"]\nseed = 1.5\n', "seed"),
        ('run = ["m"]\n[models."m"]\nseed = true\n', "seed"),
        ('run = ["a/b"]\n[models."a/b"]\n[models."a_b"]\n', "same file name"),
    ],
)
def test_config_errors_are_rejected_at_load_time(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        rag.load_config(write_config(tmp_path, text))


def test_file_names_carry_the_model_and_stay_inside_runs(tmp_path):
    paths = rag.paths_for(tmp_path, "gemini-3.5-flash-lite")
    assert paths["runs"].name == "m3_runs.gemini-3.5-flash-lite.jsonl"
    assert paths["failures"].name == "m3_runs_failures.gemini-3.5-flash-lite.jsonl"
    assert paths["dev"].name == "m3_dev_runs.gemini-3.5-flash-lite.jsonl"
    assert paths["sheet"].name == "m3_grading_sheet.gemini-3.5-flash-lite.md"
    assert paths["grades"].name == "m3_grades.gemini-3.5-flash-lite.csv"
    odd = rag.paths_for(tmp_path, "models/x y")
    assert {p.parent for p in odd.values()} == {tmp_path} and odd["runs"].name == "m3_runs.models_x_y.jsonl"


def test_two_models_keep_separate_results_and_settings(tmp_path):
    data, runs = make_world(tmp_path)
    config = rag.load_config(write_config(tmp_path, TWO_MODELS))
    seen = []

    class Recording(FakeClient):
        def _call(self, model, contents, config):
            seen.append((model, config.temperature, config.seed))
            return super()._call(model, contents, config)

    summaries = list(rag.run_models(["model-a", "model-b"], config, pause=0, data_dir=data, runs_dir=runs, client_factory=Recording))
    assert [s["model"] for s in summaries] == ["model-a", "model-b"] and all(s["calls"] == 4 for s in summaries)
    assert set(seen) == {("model-a", 0.0, 42), ("model-b", 0.7, 7)}

    for model, temperature in (("model-a", 0.0), ("model-b", 0.7)):
        paths = rag.paths_for(runs, model)
        records = read_jsonl(paths["runs"])
        assert len(records) == 4 and {r["settings"]["model_requested"] for r in records} == {model}
        assert {r["settings"]["temperature"] for r in records} == {temperature}
        sheet = paths["sheet"].read_text(encoding="utf-8")
        assert f"runs/m3_grades.{model}.csv" in sheet
    # the same G01..G04 order for every model, so the grader sees the same layout
    ids = [[r["g_id"] for r in read_jsonl(rag.paths_for(runs, m)["runs"])] for m in ("model-a", "model-b")]
    assert ids[0] == ids[1]

    before = {p.name: p.read_bytes() for p in runs.iterdir()}
    again = list(rag.run_models(["model-a", "model-b"], config, pause=0, data_dir=data, runs_dir=runs, client_factory=never_a_client))
    assert [s["calls"] for s in again] == [0, 0]
    assert {p.name: p.read_bytes() for p in runs.iterdir()} == before


def test_a_new_model_does_not_collide_with_the_logs_of_another(tmp_path):
    data, runs = make_world(tmp_path)
    rag.run("model-a", data, runs, FakeClient, pause=0)
    summary = rag.run("model-b", data, runs, FakeClient, pause=0)  # no "different settings" refusal
    assert summary["calls"] == 4


def test_one_model_aborting_does_not_stop_the_next(tmp_path):
    data, runs = make_world(tmp_path)
    config = rag.load_config(write_config(tmp_path, TWO_MODELS))
    extra = [{"qa_id": f"Q9{i}", "question": f"Extra {i}?", "gold_answer": "r", "evidence_span": "e",
              "gold_chunk_ids": [CHUNK_IDS[0]]} for i in range(3)]
    write_jsonl(data / "eval_set_v2.jsonl", read_jsonl(data / "eval_set_v2.jsonl") + extra)
    manifest_files = json.loads((data / "MANIFEST.json").read_text())["files"]
    manifest_files["eval_set_v2.jsonl"] = sha256_file(data / "eval_set_v2.jsonl")
    (data / "MANIFEST.json").write_text(json.dumps({"files": manifest_files}), encoding="utf-8")
    lines = read_jsonl(runs / "rankings_v2.jsonl")
    for e in extra:
        for retriever in ("tfidf", "dense"):
            lines.append({"qa_id": e["qa_id"], "retriever": retriever, "query_text": e["question"],
                          "config": {}, "ranking": ranking(CHUNK_IDS[0])})
    write_jsonl(runs / "rankings_v2.jsonl", lines)

    class FailsForModelA(FakeClient):
        def _call(self, model, contents, config):
            if model == "model-a":
                raise RuntimeError("429 quota")
            return super()._call(model, contents, config)

    summaries = list(rag.run_models(
        ["model-a", "model-b"], config, pause=0, data_dir=data, runs_dir=runs, client_factory=FailsForModelA,
        dev_ids=["Q003", "Q004", "Q90", "Q91", "Q92"],
    ))
    assert "5 calls failed in a row" in summaries[0]["error"]
    assert summaries[1]["calls"] == 10 and summaries[1]["failed"] == []
    assert len(read_jsonl(rag.paths_for(runs, "model-a")["dev"])) == 5  # the aborted log is kept


def test_a_refusal_or_a_missing_key_in_one_model_does_not_hide_or_block_the_others(tmp_path):
    data, runs = make_world(tmp_path)
    config = rag.load_config(write_config(tmp_path, TWO_MODELS.replace('run = ["model-a", "model-b"]', 'run = ["model-a", "model-b", "model-c"]') + '[models."model-c"]\n'))
    rag.run("model-b", data, runs, FakeClient, params={"temperature": 0.5, "seed": 1}, pause=0)  # log with other settings

    def factory_fails_for_a():
        raise RuntimeError("GEMINI_API_KEY is not set")

    done = []
    for summary in rag.run_models(["model-b", "model-c"], config, pause=0, data_dir=data, runs_dir=runs, client_factory=FakeClient):
        done.append(summary)
    assert "delete the file to start over" in done[0]["error"] and done[1]["calls"] == 4  # B refused, C still ran

    summaries = list(rag.run_models(["model-a"], config, pause=0, data_dir=data, runs_dir=runs, client_factory=factory_fails_for_a))
    assert "RuntimeError: GEMINI_API_KEY is not set" in summaries[0]["error"]


def test_dev_logs_are_per_model_and_failures_files_follow_the_model(tmp_path):
    data, runs = make_world(tmp_path)
    rag.run("model-a", data, runs, FakeClient, pause=0, dev_ids=["Q003"])
    rag.run("model-b", data, runs, FakeClient, pause=0, dev_ids=["Q003"], dev_k=1)  # other settings, other file
    assert rag.paths_for(runs, "model-a")["dev"].exists() and rag.paths_for(runs, "model-b")["dev"].exists()

    rag.run("model-c", data, runs, lambda: FakeClient(fail_with=RuntimeError("boom")), pause=0, dev_ids=["Q003"])
    assert len(read_jsonl(rag.paths_for(runs, "model-c")["dev_failures"])) == 2


def run_cli(monkeypatch, config_path, *args):
    monkeypatch.setattr("sys.argv", ["rag.py", "--config", str(config_path), *args])
    monkeypatch.setattr(rag, "run_models", lambda models, config, pause=None, **kw: calls.append((models, pause, kw)) or [])
    rag.main()


calls: list = []


def test_cli_selects_models_from_the_config_and_overrides_the_pause(tmp_path, monkeypatch):
    path = write_config(tmp_path, TWO_MODELS)
    calls.clear()
    run_cli(monkeypatch, path)
    assert calls[-1][0] == ["model-a", "model-b"] and calls[-1][1] is None
    run_cli(monkeypatch, path, "--model", "model-b", "--pause", "0")
    assert calls[-1][0] == ["model-b"] and calls[-1][1] == 0


@pytest.mark.parametrize("args", [("--model", "typo"), ("--model", ","), ("--pause", "-1"), ("--pause", "nan"), ("--pause", "inf")])
def test_cli_rejects_unknown_models_and_bad_pause_before_any_call(tmp_path, monkeypatch, args):
    calls.clear()
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, write_config(tmp_path, TWO_MODELS), *args)
    assert calls == []


def test_cli_exits_non_zero_when_a_model_failed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["rag.py", "--config", str(write_config(tmp_path, TWO_MODELS))])
    out = tmp_path / "x.jsonl"
    monkeypatch.setattr(rag, "run_models", lambda models, config, pause=None, **kw: [
        {"model": "model-a", "records": 10, "out": out, "calls": 10, "failed": []},
        {"model": "model-b", "error": "RunAborted: 5 calls failed in a row", "failed": [], "calls": None},
    ])
    with pytest.raises(SystemExit) as exc:
        rag.main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "model-a: 10 answers requested" in captured.out and "model-b: ERROR" in captured.err


# ---------------------------------------------------------------- local (Ollama) models

OLLAMA_PARAMS = {"provider": "ollama", "temperature": 0.0, "seed": 42, "pause": 0.0}
LOCAL = "llama3.1:8b"


class FakeOllama:
    """The part of OllamaClient that generation uses: request(), digest, version, base_url."""

    def __init__(self, digest="sha256:aaa", reply=None, done_reason="stop", fail=False):
        self.digest, self.version, self.base_url = digest, "0.40.18", "http://fake:11434"
        self.bodies = []
        self.reply = reply or {"answerable": True, "cited": [1], "answer": "Resposta local."}
        self.done_reason, self.fail = done_reason, fail

    def request(self, method, path, body=None):
        assert (method, path) == ("POST", "/api/chat")
        self.bodies.append(body)
        if self.fail:
            raise RuntimeError("Ollama answered HTTP 400 for /api/chat: the prompt is longer than the context length")
        return {
            "message": {"content": json.dumps(self.reply), "thinking": None},
            "done_reason": self.done_reason,
            "prompt_eval_count": 2500,
            "eval_count": 30,
        }


def test_gemini_settings_keep_exactly_their_original_keys_and_ollama_adds_its_own():
    gemini = rag.settings_for("m", 3, {"provider": "gemini", "temperature": 0.0, "seed": 42})
    assert set(gemini) == {"model_requested", "temperature", "seed", "k", "prompt_version"}
    local = rag.settings_for(LOCAL, 3, OLLAMA_PARAMS)
    assert set(local) - set(gemini) == {"provider", "num_ctx", "num_predict", "think"}
    assert local["think"] is False and local["provider"] == "ollama"


def test_committed_logs_still_match_the_settings_the_config_would_produce():
    config = rag.load_config(rag.CONFIG_PATH)
    logs = sorted(rag.RUNS.glob("m3_runs.*.jsonl"))
    assert logs, "the repository commits M3 logs"
    for path in logs:
        for record in read_jsonl(path):
            model = record["settings"]["model_requested"]
            assert record["settings"] == rag.settings_for(model, record["k"], config["models"][model]), path.name


def test_the_repo_config_defines_the_four_local_models_outside_the_default_run():
    config = rag.load_config(rag.CONFIG_PATH)
    local = [m for m, p in config["models"].items() if p["provider"] == "ollama"]
    assert sorted(local) == ["gemma4:e4b", "llama3.1:8b", "llama3.2:3b", "qwen3:8b"]
    assert not set(local) & set(config["run"])


def test_ollama_run_sends_the_frozen_request_and_logs_digest_and_usage(tmp_path):
    data, runs = make_world(tmp_path)
    server = FakeOllama()
    summary = rag.run(LOCAL, data, runs, never_a_client, params=OLLAMA_PARAMS, pause=0, ollama_factory=lambda m: server)
    assert summary["calls"] == 4 and summary["failed"] == []
    body = server.bodies[0]
    assert body["format"] == rag.RagAnswer.model_json_schema()
    assert body["think"] is False and body["truncate"] is False and body["shift"] is False and body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 1024}
    assert body["messages"] == [{"role": "user", "content": body["messages"][0]["content"]}]
    paths = rag.paths_for(runs, LOCAL)
    assert paths["runs"].name == "m3_runs.llama3.1_8b.jsonl"
    for r in read_jsonl(paths["runs"]):
        assert r["error"] is None and r["settings"] == rag.settings_for(LOCAL, 3, OLLAMA_PARAMS)
        g = r["generation"]
        assert g["model_version"] == "sha256:aaa" and g["server_version"] == "0.40.18" and g["finish_reason"] == "stop"
        assert g["usage"]["prompt_eval_count"] == 2500 and g["provider"] == "ollama"


def test_an_ollama_rerun_needs_no_server_when_the_log_is_complete(tmp_path):
    data, runs = make_world(tmp_path)
    rag.run(LOCAL, data, runs, never_a_client, params=OLLAMA_PARAMS, pause=0, ollama_factory=lambda m: FakeOllama())

    def no_server(model):
        raise AssertionError("a complete log must not need the server")

    summary = rag.run(LOCAL, data, runs, never_a_client, params=OLLAMA_PARAMS, pause=0, ollama_factory=no_server)
    assert summary["calls"] == 0


def test_an_ollama_rerun_refuses_when_the_tag_now_points_to_other_weights(tmp_path):
    data, runs = make_world(tmp_path)
    rag.run(LOCAL, data, runs, never_a_client, params=OLLAMA_PARAMS, pause=0, ollama_factory=lambda m: FakeOllama("sha256:aaa"))
    path = rag.paths_for(runs, LOCAL)["runs"]
    write_jsonl(path, read_jsonl(path)[:3])  # one answer missing, so the server is needed again
    with pytest.raises(RuntimeError, match="digest .*aaa.* but the server has sha256:bbb"):
        rag.run(LOCAL, data, runs, never_a_client, params=OLLAMA_PARAMS, pause=0, ollama_factory=lambda m: FakeOllama("sha256:bbb"))
    summary = rag.run(LOCAL, data, runs, never_a_client, params=OLLAMA_PARAMS, pause=0, ollama_factory=lambda m: FakeOllama("sha256:aaa"))
    assert summary["calls"] == 1


def test_an_ollama_error_is_logged_with_the_servers_explanation(tmp_path):
    data, runs = make_world(tmp_path)
    summary = rag.run(LOCAL, data, runs, never_a_client, params=OLLAMA_PARAMS, pause=0,
                      ollama_factory=lambda m: FakeOllama(fail=True), dev_ids=["Q003"])
    assert summary["calls"] == 2 and len(summary["failed"]) == 2
    assert all("longer than the context length" in r["error"] for r in read_jsonl(rag.paths_for(runs, LOCAL)["dev"]))


def test_a_gemini_and_an_ollama_model_run_side_by_side(tmp_path):
    data, runs = make_world(tmp_path)
    config = rag.load_config(write_config(
        tmp_path, 'run = ["model-a", "llama3.1:8b"]\n[models."model-a"]\n[models."llama3.1:8b"]\nprovider = "ollama"\npause = 0\n'))
    summaries = list(rag.run_models(
        ["model-a", LOCAL], config, data_dir=data, runs_dir=runs, client_factory=FakeClient, ollama_factory=lambda m: FakeOllama()))
    assert [s["calls"] for s in summaries] == [4, 4]
    assert {p.name for p in runs.glob("m3_runs.*.jsonl")} == {"m3_runs.model-a.jsonl", "m3_runs.llama3.1_8b.jsonl"}
    assert "provider" not in read_jsonl(rag.paths_for(runs, "model-a")["runs"])[0]["settings"]


def test_an_unreachable_or_unpulled_local_model_is_reported_and_the_next_model_runs(tmp_path):
    data, runs = make_world(tmp_path)
    config = rag.load_config(write_config(
        tmp_path, 'run = ["llama3.1:8b", "model-a"]\n[models."model-a"]\n[models."llama3.1:8b"]\nprovider = "ollama"\npause = 0\n'))

    def not_pulled(model):
        raise RuntimeError(f"model {model} is not on http://fake: run `docker exec ragfortextmining-ollama ollama pull {model}`")

    summaries = list(rag.run_models(
        [LOCAL, "model-a"], config, pause=0, data_dir=data, runs_dir=runs, client_factory=FakeClient, ollama_factory=not_pulled))
    assert "ollama pull llama3.1:8b" in summaries[0]["error"] and summaries[1]["calls"] == 4
