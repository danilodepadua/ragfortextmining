import json
import shutil

import pytest

import evaluate
import rag
from common import DATA, RUNS, write_jsonl


HEADER = "g_id,grade,cited_supports,notes\n"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "expected header"),
        ("grade,g_id,cited_supports,notes\nG01,Correct,Yes,\n", "expected header"),
        ("\ufeff" + HEADER + "G01,Correct,Yes,\n", "expected header"),
        (HEADER + "G01,Correct,Yes\n", "expected 4 columns"),
        (HEADER + "G01,Correct,Yes,extra,column\n", "expected 4 columns"),
        (HEADER + "G01,Correct,Yes,\n\n", "expected 4 columns"),
        (HEADER + 'G01,Correct,Yes,"unclosed note\n', "malformed CSV"),
        (HEADER + "G01,Correct,Yes,\nG01,Partial,Yes,\n", "duplicate g_id"),
        (HEADER + "G11,Correct,Yes,\n", "unexpected g_id"),
        (HEADER, "missing g_id"),
        (HEADER + "G01,correct,Yes,\n", "invalid grade"),
        (HEADER + "G01,Correct, Yes,\n", "invalid cited_supports"),
    ],
)
def test_load_grades_rejects_invalid_csv(tmp_path, content, message):
    path = tmp_path / "grades.csv"
    path.write_text(content, encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match=message):
        evaluate.load_grades(path, {"G01"})


def test_join_uses_grading_id_and_preserves_notes(tmp_path):
    jobs = [
        {"g_id": "G01", "qa_id": "Q021", "retriever": "dense"},
        {"g_id": "G02", "qa_id": "Q021", "retriever": "tfidf"},
    ]
    run_path, grades_path = tmp_path / "run.jsonl", tmp_path / "grades.csv"
    write_jsonl(run_path, [
        {**jobs[1], "error": None, "cited_chunk_ids": [], "answer": "O texto não permite responder."},
        {**jobs[0], "error": None, "cited_chunk_ids": ["one"], "answer": "Resposta."},
    ])
    grades_path.write_text(
        HEADER + 'G02,Wrong,N/A,\nG01,Partial,Yes,"Trecho correto,\nmas faltou a ""causa""."\n', encoding="utf-8",
    )
    dense, tfidf = evaluate.join_grades(run_path, grades_path, jobs)
    assert (dense["retriever"], dense["grade"], dense["notes"]) == (
        "dense", "Partial", 'Trecho correto,\nmas faltou a "causa".',
    )
    assert (tfidf["retriever"], tfidf["grade"], tfidf["cited_supports"]) == ("tfidf", "Wrong", "N/A")


@pytest.mark.parametrize(("cited_chunk_ids", "support"), [([], "Yes"), ([], "No"), (["one"], "N/A")])
def test_join_requires_na_iff_there_is_no_mapped_citation(tmp_path, cited_chunk_ids, support):
    job = {"g_id": "G01", "qa_id": "Q021", "retriever": "dense"}
    run_path, grades_path = tmp_path / "run.jsonl", tmp_path / "grades.csv"
    write_jsonl(run_path, [{**job, "error": None, "cited_chunk_ids": cited_chunk_ids, "cited_numbers": [4]}])
    grades_path.write_text(HEADER + f"G01,Correct,{support},\n", encoding="utf-8")
    with pytest.raises(ValueError, match="G01 requires cited_supports = N/A"):
        evaluate.join_grades(run_path, grades_path, [job])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"g_id": "G03"}, "expected one successful run"),
        ({"retriever": "tfidf"}, "does not match its frozen paired job"),
        ({"error": "503"}, "does not match its frozen paired job"),
    ],
)
def test_join_rejects_run_records_that_do_not_match_the_frozen_slot(tmp_path, change, message):
    job = {"g_id": "G01", "qa_id": "Q021", "retriever": "dense"}
    run_path, grades_path = tmp_path / "run.jsonl", tmp_path / "grades.csv"
    write_jsonl(run_path, [{**job, "error": None, **change}])
    grades_path.write_text(HEADER + "G01,Correct,Yes,\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        evaluate.join_grades(run_path, grades_path, [job])


def test_join_rejects_duplicate_run_id(tmp_path):
    path = tmp_path / "run.jsonl"
    write_jsonl(path, [{"g_id": "G01"}, {"g_id": "G01"}])
    with pytest.raises(ValueError, match="duplicate g_id"):
        evaluate.join_grades(path, tmp_path / "grades.csv", [{"g_id": "G01"}])


def test_document_recall_distinguishes_cross_document_and_within_document_misses():
    chunks = {"gold": {"doc_id": "D001"}, "near": {"doc_id": "D001"}, "other": {"doc_id": "D002"}}
    questions = {q: {"doc_id": "D001", "question": q} for q in ("Q001", "Q002")}
    rankings = {
        "Q001": [{"chunk_id": key} for key in ("near", "gold", "other")],
        "Q002": [{"chunk_id": key} for key in ("other", "near", "gold")],
    }
    diagnostic = evaluate.document_diagnostics(rankings, questions, chunks, {"Q001": 2, "Q002": 3})
    assert diagnostic["doc_recall_at_1"] == 0.5
    assert diagnostic["doc_hits_at_1"] == 1
    assert diagnostic["wrong_doc_at_1"] == [
        {"qa_id": "Q002", "question": "Q002", "gold_doc_id": "D001", "ranked_doc_id": "D002",
         "ranked_chunk_id": "other", "first_gold_rank": 3},
    ]
    assert diagnostic["evidence_misses"]["1"] == {
        "target_document_present": ["Q001"], "target_document_absent": ["Q002"],
    }


def test_rrf_uses_ranks_from_both_lists_with_a_stable_tie_break():
    rankings = [
        [{"chunk_id": key, "rank": rank, "score": 0.0} for rank, key in enumerate(order, 1)]
        for order in (("b", "a", "c"), ("a", "b", "c"))
    ]
    fused = evaluate.reciprocal_rank_fusion(rankings)
    assert [item["chunk_id"] for item in fused] == ["a", "b", "c"]
    assert [item["rank"] for item in fused] == [1, 2, 3]
    assert [item["score"] for item in fused] == pytest.approx([1 / 61 + 1 / 62, 1 / 61 + 1 / 62, 2 / 63])
    assert fused == evaluate.reciprocal_rank_fusion(list(reversed(rankings)))


@pytest.mark.parametrize("name", ["corpus_v2.jsonl", "chunks_v2.jsonl", "eval_set_v2.jsonl", "paired_questions_v2.json"])
def test_evaluate_rejects_a_changed_frozen_input(tmp_path, name):
    data = shutil.copytree(DATA, tmp_path / "data")
    with (data / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(RuntimeError, match=f"{name} does not match data/MANIFEST.json"):
        evaluate.evaluate(data)


def test_multimodel_evaluation_and_json_are_reproducible(tmp_path):
    runs = shutil.copytree(RUNS, tmp_path / "runs", ignore=shutil.ignore_patterns("*.npz", "*.md", evaluate.SUMMARY_FILENAME))
    (runs / "m3_grades_example.csv").write_text("a scratch file", encoding="utf-8")
    result = evaluate.evaluate(runs_dir=runs)
    config = rag.load_config()
    summary = result["rag"]
    assert list(summary["models"]) == list(config["models"])
    assert summary["default_models"] == ["gemini-3.5-flash-lite"]
    assert summary["overall"]["n"] == 70
    assert summary["overall"]["grades"] == {"Correct": 66, "Partial": 4, "Wrong": 0}
    assert summary["overall"]["cited_supports"] == {"Yes": 58, "No": 12, "N/A": 0}
    assert summary["overall"]["invalid_citations"] == 3
    assert summary["by_provider"]["gemini"]["cited_supports"]["Yes"] == 30
    assert summary["by_provider"]["ollama"]["correct_without_support"] == 10
    assert [summary["by_retriever"][name]["gold_in_context"] for name in ("tfidf", "dense")] == [35, 28]
    assert summary["highlighted"]["n"] == 42
    assert len(summary["answer_grades"]) == 70 and len(summary["non_perfect"]) == 18
    assert {row["retriever"]: row["rubric_grade"] for row in summary["development_probe"]} == {"tfidf": "Wrong", "dense": None}
    assert [failure["recovered"] for model in summary["models"].values() for failure in model["failed_attempts"]] == [True, True]
    assert summary["models"]["gemma4:e4b"]["overall"]["tokens"]["prompt"]["n"] == 10
    assert summary["models"]["gemma4:e4b"]["overall"]["tokens"]["reported_total"]["n"] == 0
    assert summary["models"]["gemini-3.5-flash-lite"]["overall"]["tokens"]["reported_total"]["n"] == 10

    retrieval = result["retrieval"]
    assert retrieval["retrievers"]["tfidf"]["doc_recall_at_1"] == 0.96
    assert retrieval["retrievers"]["dense"]["doc_recall_at_1"] == 1.0
    assert retrieval["rrf"]["evidence_recall_at_k"] == {"1": 0.64, "3": 0.96, "5": 1.0}
    assert retrieval["rrf"]["mrr"] == pytest.approx(0.8033333333333332)

    output = runs / evaluate.SUMMARY_FILENAME
    evaluate.write_summary(output, result)
    first = output.read_bytes()
    evaluate.write_summary(output, evaluate.evaluate(runs_dir=runs))
    assert output.read_bytes() == first
    assert first.endswith(b"\n") and b"\r" not in first and not first.startswith(b"\xef\xbb\xbf")
    assert json.loads(first) == result


def test_failed_summary_swap_keeps_the_previous_output(tmp_path, monkeypatch):
    output = tmp_path / "summary.json"
    output.write_text('{"complete": true}\n', encoding="utf-8")

    def fail_replace(source, destination):
        assert source.parent == destination.parent
        raise OSError("cannot replace output")

    monkeypatch.setattr(evaluate.os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace output"):
        evaluate.write_summary(output, {"complete": False})
    assert output.read_bytes() == b'{"complete": true}\n'
    assert list(tmp_path.iterdir()) == [output]
