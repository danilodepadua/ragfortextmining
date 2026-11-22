import json
from types import SimpleNamespace

import pytest

import curate
from common import read_jsonl


def words(n, prefix="palavra"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def qa(question="Como configurar o serviço?", span="palavra10 palavra11 palavra12", **kw):
    base = {
        "question": question,
        "answer": "Defina o parâmetro correto.",
        "evidence_spans": [span],
        "evidence_truncated": False,
    }
    base.update(kw)
    return base


def row(i, n_words=450, **kw):
    base = {
        "document_id": f"doc-{i}",
        "hf_row_idx": i,
        "language": "pt-br",
        "domain": "tecnologia",
        "doc_type": "tutorial",
        "context": words(n_words, f"d{i}x"),
        "qa_pairs": [],
    }
    base.update(kw)
    return base


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Segundo o texto, qual é a função do cache?", True),
        ("Quais passos são mencionados no texto?", True),
        ("De acordo com o artigo, o que é necessário?", True),
        ("Neste documento, quem é o autor?", True),
        ("Como configurar o cache do servidor?", False),
        ("O que é um contexto de execução?", False),
    ],
)
def test_is_deictic(question, expected):
    assert curate.is_deictic(question) is expected


def test_code_density_uses_word_boundaries():
    prose = "Para salvar e ativar o recurso, clique em avançar e depois confirmar. " * 20
    assert curate.code_density(prose) == 0
    code = "def f(): return 1; class A { } import x; " * 20
    assert curate.code_density(code) > curate.CODE_TOKENS_PER_100_WORDS


@pytest.mark.parametrize(
    "overrides,context,reason",
    [
        ({"language": "en"}, None, "language"),
        ({"domain": "saude"}, None, "domain"),
        ({"doc_type": "noticia"}, None, "doc_type"),
        ({}, words(399), "too_short"),
        ({}, words(1501), "too_long"),
        ({}, "� " + words(450), "broken"),
        ({}, None, None),
    ],
)
def test_cheap_reject(overrides, context, reason):
    r = row(1, **overrides)
    text = context if context is not None else r["context"]
    assert curate.cheap_reject(r, text) == reason


def test_select_qa_accepts_a_valid_span_and_reports_offsets():
    context = words(450)
    selection, reasons = curate.select_qa(context, [qa()], set())
    assert selection["evidence_span"] == "palavra10 palavra11 palavra12"
    start, end = selection["evidence_char_start"], selection["evidence_char_end"]
    assert context[start:end] == selection["evidence_span"]
    assert reasons == []


@pytest.mark.parametrize(
    "item,reason",
    [
        (qa(evidence_truncated=True), "evidence_truncated"),
        (qa(evidence_spans=["palavra10", "palavra11"]), "not_single_span"),
        (qa(answer="curta"), "short_answer"),
        (qa(question="Segundo o texto, o que fazer?"), "deictic_question"),
        (qa(span="trecho inexistente"), "span_not_found"),
    ],
)
def test_select_qa_rejections(item, reason):
    selection, reasons = curate.select_qa(words(450), [item], set())
    assert selection is None and reasons == [reason]


def test_select_qa_strips_the_span_so_offsets_match_words():
    context = words(450)
    selection, reasons = curate.select_qa(context, [qa(span="\npalavra10 palavra11\n")], set())
    assert reasons == [] and selection["evidence_span"] == "palavra10 palavra11"
    assert context[selection["evidence_char_start"] : selection["evidence_char_end"]] == "palavra10 palavra11"
    selection, reasons = curate.select_qa(context, [qa(span="  \n ")], set())
    assert selection is None and reasons == ["empty_span"]


def test_select_qa_rejects_repeated_and_long_spans_and_duplicate_questions():
    repeated = "abc def " + words(100) + " abc def " + words(100, "q")
    selection, reasons = curate.select_qa(repeated, [qa(span="abc def")], set())
    assert selection is None and reasons == ["span_repeated"]

    context = words(450)
    long_span = " ".join(f"palavra{i}" for i in range(10, 62))  # 52 words
    selection, reasons = curate.select_qa(context, [qa(span=long_span)], set())
    assert selection is None and reasons == ["span_too_long"]

    selection, reasons = curate.select_qa(context, [qa()], {"como configurar o serviço?"})
    assert selection is None and reasons == ["duplicate_question"]


def test_select_qa_falls_through_to_the_next_valid_qa():
    items = [qa(evidence_truncated=True), qa(question="Qual o limite padrão?")]
    selection, reasons = curate.select_qa(words(450), items, set())
    assert selection["question"] == "Qual o limite padrão?"
    assert selection["source_qa_index"] == 1 and reasons == ["evidence_truncated"]


class FakeJudgeClient:
    def __init__(self, accept_ids=None, fail=False):
        self.models = SimpleNamespace(generate_content=self._call)
        self.accept = accept_ids
        self.fail = fail
        self.count = 0

    def _call(self, **kwargs):
        self.count += 1
        if self.fail:
            raise ConnectionError("down")
        ok = self.accept is None or self.count in self.accept
        payload = {
            "is_technical": ok,
            "is_news_opinion_or_marketing": False,
            "answer_supported_by_span": True,
            "question_self_contained": True,
            "reason": "ok",
        }
        return SimpleNamespace(text=json.dumps(payload), model_version="fake-1", usage_metadata=None)


def good_rows(n):
    return [row(i, qa_pairs=[qa(question=f"Pergunta número {i}?", span=f"d{i}x10 d{i}x11")]) for i in range(n)]


def test_collect_filters_judges_dedupes_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(curate.time, "sleep", lambda s: None)
    rows = good_rows(3)
    rows.insert(1, row(99, language="en"))  # cheap reject
    rows.append({**rows[0], "document_id": "dup"})  # same context as rows[0]
    client = FakeJudgeClient(accept_ids={1, 3})  # judge rejects the 2nd judged document
    pool = curate.collect(iter(rows), client, "m", data_dir=tmp_path, sleep=0)
    assert [p["document_id"] for p in pool] == ["doc-0", "doc-2"]
    log = read_jsonl(tmp_path / "curation_log.jsonl")
    decisions = [line.get("decision") for line in log if line["type"] == "candidate"]
    assert decisions == ["accepted", "rejected_by_judge", "accepted"]
    summary = log[-1]
    assert summary["reject_counts"] == {"language": 1, "duplicate_context": 1}
    assert log[0]["revision"] == curate.REVISION and log[0]["seed"] == 42
    assert read_jsonl(tmp_path / "raw" / "accepted_pool.jsonl")[0]["context"].startswith("d0x0")


def test_collect_stops_after_five_judge_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(curate.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="5 judge calls failed"):
        curate.collect(iter(good_rows(10)), FakeJudgeClient(fail=True), "m", data_dir=tmp_path, sleep=0)
    log = read_jsonl(tmp_path / "curation_log.jsonl")
    assert [line["decision"] for line in log if line["type"] == "candidate"] == ["judge_error"] * 5


def test_collect_respects_the_pool_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(curate.time, "sleep", lambda s: None)
    pool = curate.collect(iter(good_rows(40)), FakeJudgeClient(), "m", data_dir=tmp_path, sleep=0)
    assert len(pool) == curate.TARGET + curate.RESERVE


def test_finalize_applies_human_review_and_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(curate.time, "sleep", lambda s: None)
    curate.collect(iter(good_rows(40)), FakeJudgeClient(), "m", data_dir=tmp_path, sleep=0)
    (tmp_path / "human_review.json").write_text(json.dumps({"rejected": ["doc-0", "doc-1"]}))
    selected = curate.finalize(tmp_path)
    assert len(selected) == 25
    assert selected[0]["doc_id"] == "D001" and selected[0]["document_id"] == "doc-2"
    assert selected[-1]["doc_id"] == "D025"
    assert len(read_jsonl(tmp_path / "corpus_v2.jsonl")) == 25
    review = (tmp_path / "review.md").read_text(encoding="utf-8")
    assert "## D001 · doc-2" in review and "**Evidência:**" in review


def test_finalize_fails_when_too_few_documents_remain(tmp_path, monkeypatch):
    monkeypatch.setattr(curate.time, "sleep", lambda s: None)
    curate.collect(iter(good_rows(40)), FakeJudgeClient(), "m", data_dir=tmp_path, sleep=0)
    rejected = [f"doc-{i}" for i in range(7)]
    (tmp_path / "human_review.json").write_text(json.dumps({"rejected": rejected}))
    with pytest.raises(RuntimeError, match="need 25"):
        curate.finalize(tmp_path)


def test_collect_flushes_the_log_when_the_stream_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(curate.time, "sleep", lambda s: None)

    def broken_stream():
        yield from good_rows(1)
        raise OSError("network")

    with pytest.raises(OSError):
        curate.collect(broken_stream(), FakeJudgeClient(), "m", data_dir=tmp_path, sleep=0)
    assert [p["document_id"] for p in read_jsonl(tmp_path / "raw" / "accepted_pool.jsonl")] == ["doc-0"]
    assert read_jsonl(tmp_path / "curation_log.jsonl")[-1]["type"] == "summary"


def test_collect_refuses_to_overwrite_an_existing_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(curate.time, "sleep", lambda s: None)
    curate.collect(iter(good_rows(2)), FakeJudgeClient(), "m", data_dir=tmp_path, sleep=0)
    with pytest.raises(FileExistsError, match="--overwrite"):
        curate.collect(iter(good_rows(2)), FakeJudgeClient(), "m", data_dir=tmp_path, sleep=0)
    curate.collect(iter(good_rows(2)), FakeJudgeClient(), "m", data_dir=tmp_path, sleep=0, overwrite=True)


def test_collect_budget_counts_failed_attempts_and_rows_streamed_is_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(curate.time, "sleep", lambda s: None)
    curate.collect(iter(good_rows(10)), FakeJudgeClient(), "m", data_dir=tmp_path, sleep=0, max_judge_calls=3)
    summary = read_jsonl(tmp_path / "curation_log.jsonl")[-1]
    assert summary["judge_calls"] == 3 and summary["rows_streamed"] == 3


@pytest.mark.parametrize(
    "review,message",
    [
        ({"rejected": ["doc-3x"]}, "not in the pool"),
        ({"reject": ["doc-3"]}, "unknown keys"),
        ({"rejected": [3]}, "not in the pool"),
    ],
)
def test_finalize_fails_closed_on_human_review_mistakes(tmp_path, monkeypatch, review, message):
    monkeypatch.setattr(curate.time, "sleep", lambda s: None)
    curate.collect(iter(good_rows(40)), FakeJudgeClient(), "m", data_dir=tmp_path, sleep=0)
    (tmp_path / "human_review.json").write_text(json.dumps(review))
    with pytest.raises(ValueError, match=message):
        curate.finalize(tmp_path)


class QuotaClient(FakeJudgeClient):
    """Accepts `ok_calls` calls, then fails like an exhausted daily quota."""

    def __init__(self, ok_calls):
        super().__init__()
        self.ok_calls = ok_calls

    def _call(self, **kwargs):
        if self.count >= self.ok_calls:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return super()._call(**kwargs)


def test_resume_continues_after_a_quota_error_without_paying_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(curate.time, "sleep", lambda s: None)
    rows = good_rows(40)
    with pytest.raises(RuntimeError, match="5 judge calls failed"):
        curate.collect(iter(rows), QuotaClient(ok_calls=3), "m1", data_dir=tmp_path, sleep=0)
    first = read_jsonl(tmp_path / "raw" / "accepted_pool.jsonl")
    assert [p["document_id"] for p in first] == ["doc-0", "doc-1", "doc-2"]

    client = FakeJudgeClient()
    pool = curate.collect(iter(rows), client, "m2", data_dir=tmp_path, sleep=0, resume=True)
    assert len(pool) == curate.TARGET + curate.RESERVE
    assert [p["document_id"] for p in pool[:3]] == ["doc-0", "doc-1", "doc-2"]
    assert client.count == curate.TARGET + curate.RESERVE - 3  # the first 3 were not judged again

    log = read_jsonl(tmp_path / "curation_log.jsonl")
    assert [line["judge_model"] for line in log if line["type"] == "run"] == ["m1", "m2"]
    assert any(line["type"] == "previous_summary" for line in log)
    assert sum(1 for line in log if line.get("decision") == "judge_error") == 5
    assert log[-1]["type"] == "summary" and log[-1]["reject_counts"]["already_judged"] == 3


def test_resume_needs_existing_files_and_excludes_overwrite(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing to resume"):
        curate.collect(iter([]), FakeJudgeClient(), "m", data_dir=tmp_path, resume=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        curate.collect(iter([]), FakeJudgeClient(), "m", data_dir=tmp_path, resume=True, overwrite=True)
