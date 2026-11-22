import hashlib
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

import retrieve
from common import sha256_file, write_jsonl


def words_to_vector(text, dim):
    vec = np.zeros(dim)
    vec[0] = 1e-3  # never an all-zero vector
    for word in text.lower().split():
        vec[int(hashlib.sha256(word.encode()).hexdigest(), 16) % dim] += 1
    return vec


class FakeEmbedClient:
    """Bag-of-words embeddings, so retrieval is meaningful in tests. Counts requests."""

    def __init__(self):
        self.models = SimpleNamespace(embed_content=self._call)
        self.requests = []

    def _call(self, model, contents, config):
        self.requests.append({"task": config.task_type, "texts": list(contents)})
        dim = config.output_dimensionality
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=words_to_vector(t, dim)) for t in contents],
            usage_metadata=None,
        )


def never_a_client():
    raise AssertionError("the client factory must not be called")


CHUNK_TEXTS = [
    "A chave de acesso expira depois de trinta dias e deve ser renovada no painel.",
    "O servidor reinicia automaticamente quando a memória livre chega a zero.",
    "Para exportar relatórios em formato CSV abra o menu de relatórios e escolha exportar.",
    "O backup diário grava uma cópia criptografada dos dados em outro datacenter.",
]
QUESTIONS = [
    "Quando expira a chave de acesso?",
    "Como exportar relatórios em CSV?",
]


def make_data(tmp_path, chunk_texts=CHUNK_TEXTS, questions=QUESTIONS, gold=(0, 2)):
    data = tmp_path / "data"
    chunks = [
        {"chunk_id": f"chk_D001_{i:03d}_{i:08x}", "doc_id": "D001", "chunk_index": i, "text": t}
        for i, t in enumerate(chunk_texts)
    ]
    eval_set = [
        {"qa_id": f"Q{i + 1:03d}", "question": q, "gold_chunk_ids": [chunks[gold[i]]["chunk_id"]]}
        for i, q in enumerate(questions)
    ]
    write_jsonl(data / "chunks_v2.jsonl", chunks)
    write_jsonl(data / "eval_set_v2.jsonl", eval_set)
    manifest = {"files": {"chunks_v2.jsonl": sha256_file(data / "chunks_v2.jsonl")}}
    (data / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data, chunks, eval_set


@pytest.fixture(autouse=True)
def small_dimension(monkeypatch):
    monkeypatch.setattr(retrieve, "EMBED_DIM", 32)


# ---------------------------------------------------------------- scoring


def test_tfidf_ranks_the_planted_answer_first_ignoring_accents_and_case():
    scores, empty = retrieve.tfidf_scores(CHUNK_TEXTS, ["MEMORIA livre do servidor", "exportar RELATORIOS csv"])
    assert empty == []
    assert int(np.argmax(scores[0])) == 1 and int(np.argmax(scores[1])) == 2


def test_tfidf_reports_queries_with_no_known_term():
    scores, empty = retrieve.tfidf_scores(CHUNK_TEXTS, ["xyzzy plugh", "servidor"])
    assert empty == [0] and not scores[0].any()


def test_rank_row_breaks_ties_by_chunk_order_and_rounds_scores():
    ranking = retrieve.rank_row(np.array([0.5, 0.9, 0.5, 0.9000000001]), ["a", "b", "c", "d"])
    assert [r["chunk_id"] for r in ranking] == ["b", "d", "a", "c"]  # b and d tie after rounding
    assert [r["rank"] for r in ranking] == [1, 2, 3, 4]
    assert ranking[0]["score"] == 0.9


def test_rank_row_never_writes_negative_zero():
    ranking = retrieve.rank_row(np.array([-1e-12, 0.0]), ["a", "b"])
    assert "-0.0" not in json.dumps(ranking)


def test_dense_scores_are_cosines_of_unnormalized_vectors():
    chunks = np.array([[3.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float32)
    queries = np.array([[10.0, 0.0]], dtype=np.float32)
    scores = retrieve.dense_scores(chunks, queries)
    assert scores.dtype == np.float64
    assert scores[0].round(6).tolist() == [1.0, 0.0, 0.707107]


# ---------------------------------------------------------------- cache


def test_cache_round_trip_is_byte_stable(tmp_path):
    path = tmp_path / "cache.npz"
    d = retrieve.EMBED_DIM
    cache = {"b": np.full(d, 1.0, dtype=np.float32), "a": np.full(d, 3.0, dtype=np.float32)}
    retrieve.save_cache(path, cache)
    first = path.read_bytes()
    retrieve.save_cache(path, dict(reversed(list(cache.items()))))
    assert path.read_bytes() == first
    loaded = retrieve.load_cache(path)
    assert sorted(loaded) == ["a", "b"] and loaded["a"].tolist() == [3.0] * d


def test_ensure_embeddings_embeds_only_what_is_missing_and_saves_each_batch(tmp_path):
    path = tmp_path / "cache.npz"
    client = FakeEmbedClient()
    items = [(f"k{i}", f"texto {i}", "RETRIEVAL_DOCUMENT") for i in range(5)]
    cache = {}
    calls = retrieve.ensure_embeddings(items, cache, path, lambda: client, pause=0, batch_size=2)
    assert calls == 3 and [len(r["texts"]) for r in client.requests] == [2, 2, 1]
    assert sorted(retrieve.load_cache(path)) == [f"k{i}" for i in range(5)]

    again = retrieve.ensure_embeddings(items, cache, path, never_a_client, pause=0, batch_size=2)
    assert again == 0

    items.append(("k5", "texto novo", "RETRIEVAL_DOCUMENT"))
    client.requests.clear()
    assert retrieve.ensure_embeddings(items, cache, path, lambda: client, pause=0) == 1
    assert client.requests[0]["texts"] == ["texto novo"]


def test_ensure_embeddings_keeps_finished_batches_when_a_later_one_fails(tmp_path):
    path = tmp_path / "cache.npz"

    class Flaky(FakeEmbedClient):
        def _call(self, model, contents, config):
            if self.requests:
                raise RuntimeError("429 quota")
            return super()._call(model, contents, config)

    items = [(f"k{i}", f"texto {i}", "T") for i in range(4)]
    with pytest.raises(RuntimeError, match="429"):
        retrieve.ensure_embeddings(items, {}, path, Flaky, pause=0, batch_size=2)
    assert sorted(retrieve.load_cache(path)) == ["k0", "k1"]


def test_ensure_embeddings_pauses_between_requests_only(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(retrieve.time, "sleep", sleeps.append)
    items = [(f"k{i}", "t", "T") for i in range(5)]
    retrieve.ensure_embeddings(items, {}, tmp_path / "c.npz", FakeEmbedClient, pause=7, batch_size=2)
    assert sleeps == [7, 7]  # 3 requests, no pause before the first


def test_keys_depend_on_task_and_content():
    assert retrieve.query_key("a?") != retrieve.query_key("b?")
    assert retrieve.query_key("a?").startswith("QUESTION_ANSWERING|")
    assert retrieve.chunk_key("chk_1").startswith("RETRIEVAL_DOCUMENT|")


# ---------------------------------------------------------------- metrics


def line(qa_id, retriever, order):
    return {
        "qa_id": qa_id,
        "retriever": retriever,
        "ranking": [{"rank": r + 1, "chunk_id": c, "score": 0.0} for r, c in enumerate(order)],
    }


def test_metrics_by_hand():
    eval_set = [
        {"qa_id": "Q1", "gold_chunk_ids": ["a"]},
        {"qa_id": "Q2", "gold_chunk_ids": ["b", "c"]},
        {"qa_id": "Q3", "gold_chunk_ids": ["d"]},
    ]
    ids = ["a", "b", "c", "d", "e", "f"]
    rankings = [
        line("Q1", "tfidf", ["a", "b", "c", "d", "e", "f"]),  # first gold at rank 1
        line("Q2", "tfidf", ["e", "f", "c", "a", "b", "d"]),  # rank 3 (c)
        line("Q3", "tfidf", ["e", "f", "a", "b", "c", "d"]),  # rank 6
        line("Q1", "dense", ["b", "a", "c", "d", "e", "f"]),  # rank 2
        line("Q2", "dense", ["b", "c", "a", "d", "e", "f"]),  # rank 1
        line("Q3", "dense", ["d", "a", "b", "c", "e", "f"]),  # rank 1
    ]
    m = retrieve.compute_metrics(rankings, eval_set, len(ids))
    tfidf, dense = m["retrievers"]["tfidf"], m["retrievers"]["dense"]
    assert tfidf["first_gold_rank"] == {"Q1": 1, "Q2": 3, "Q3": 6}
    assert tfidf["evidence_recall_at_k"] == {"1": 1 / 3, "3": 2 / 3, "5": 2 / 3}
    assert tfidf["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 6) / 3)
    assert dense["evidence_recall_at_k"] == {"1": 2 / 3, "3": 1.0, "5": 1.0}
    assert m["paired_counts"]["1"] == {"both": 0, "dense_only": 2, "tfidf_only": 1, "neither": 0}
    assert m["paired_counts"]["3"] == {"both": 2, "dense_only": 1, "tfidf_only": 0, "neither": 0}


def test_random_baseline_matches_brute_force():
    n, gold_counts = 6, [1, 2]
    baseline = retrieve.random_baseline(gold_counts, n)
    for g in set(gold_counts):
        for k in retrieve.KS:
            hit = mrr = 0.0
            orders = list(itertools.permutations(range(n)))
            for order in orders:
                ranks = [pos + 1 for pos, chunk in enumerate(order) if chunk < g]
                hit += min(ranks) <= k
                mrr += 1 / min(ranks)
            single = retrieve.random_baseline([g], n)
            assert single["evidence_recall_at_k"][str(k)] == pytest.approx(hit / len(orders))
            assert single["mrr"] == pytest.approx(mrr / len(orders))
    assert set(baseline["evidence_recall_at_k"]) == {"1", "3", "5"}


def test_first_gold_rank_requires_a_gold_chunk():
    with pytest.raises(ValueError):
        retrieve.first_gold_rank([{"rank": 1, "chunk_id": "x"}], {"y"})


# ---------------------------------------------------------------- pipeline


def test_run_end_to_end_writes_full_rankings_and_metrics(tmp_path):
    data, chunks, eval_set = make_data(tmp_path)
    runs = tmp_path / "runs"
    client = FakeEmbedClient()
    result, calls = retrieve.run(data, runs, lambda: client, pause=0)
    assert calls == 2  # one batch of chunks, one of questions
    assert {r["task"] for r in client.requests} == {"RETRIEVAL_DOCUMENT", "QUESTION_ANSWERING"}

    lines = [json.loads(x) for x in (runs / "rankings_v2.jsonl").read_text().splitlines()]
    assert len(lines) == len(eval_set) * 2
    chunk_ids = {c["chunk_id"] for c in chunks}
    for item in lines:
        assert {r["chunk_id"] for r in item["ranking"]} == chunk_ids  # the full ranking
        assert [r["rank"] for r in item["ranking"]] == list(range(1, len(chunks) + 1))
        assert item["chunks_sha256"] == result["chunks_sha256"]
        assert item["retriever"] in {"tfidf", "dense"} and "scope" not in item
    assert lines[0]["config"]["vectorizer"].startswith("TfidfVectorizer")
    assert lines[1]["config"]["model"] == "gemini-embedding-001" and "google-genai" in lines[1]["config"]
    metrics = json.loads((runs / "retrieval_metrics.json").read_text())
    assert metrics["retrievers"]["tfidf"]["evidence_recall_at_k"]["1"] == 1.0  # planted answers
    assert metrics["n_chunks"] == 4 and "random" in metrics
    assert "| tfidf |" in retrieve.format_table(result)


def test_second_run_is_offline_and_byte_identical(tmp_path):
    data, _, _ = make_data(tmp_path)
    runs = tmp_path / "runs"
    retrieve.run(data, runs, FakeEmbedClient, pause=0)
    first = {p.name: p.read_bytes() for p in runs.iterdir()}
    result, calls = retrieve.run(data, runs, never_a_client, pause=0)  # would raise if it asked for a client
    assert calls == 0
    assert {p.name: p.read_bytes() for p in runs.iterdir()} == first


def test_a_changed_chunk_reembeds_only_that_chunk(tmp_path):
    data, chunks, _ = make_data(tmp_path)
    runs = tmp_path / "runs"
    retrieve.run(data, runs, FakeEmbedClient, pause=0)

    # In the real data the chunk id carries a hash of the text, so a changed text is a new id.
    chunks[3] = {**chunks[3], "chunk_id": "chk_D001_003_deadbeef", "text": "O backup semanal grava dados."}
    write_jsonl(data / "chunks_v2.jsonl", chunks)
    manifest = {"files": {"chunks_v2.jsonl": sha256_file(data / "chunks_v2.jsonl")}}
    (data / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    client = FakeEmbedClient()
    _, calls = retrieve.run(data, runs, lambda: client, pause=0)
    assert calls == 1 and client.requests[0]["texts"] == ["O backup semanal grava dados."]


def test_run_refuses_chunks_that_do_not_match_the_manifest(tmp_path):
    data, chunks, _ = make_data(tmp_path)
    (data / "chunks_v2.jsonl").write_text((data / "chunks_v2.jsonl").read_text() + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="MANIFEST"):
        retrieve.run(data, tmp_path / "runs", never_a_client, pause=0)


def test_run_warns_about_queries_without_known_terms(tmp_path, capsys):
    data, _, _ = make_data(tmp_path, questions=["servidor memória", "xyzzy plugh"], gold=(1, 0))
    result, _ = retrieve.run(data, tmp_path / "runs", FakeEmbedClient, pause=0)
    assert result["queries_without_known_terms"] == ["Q002"]
    assert "no TF-IDF term" in capsys.readouterr().err


def test_random_baseline_with_k_at_least_the_corpus_size():
    baseline = retrieve.random_baseline([1], 4)  # k = 5 > 4 chunks: always found
    assert baseline["evidence_recall_at_k"]["5"] == 1.0
    assert baseline["evidence_recall_at_k"]["1"] == pytest.approx(0.25)


def test_zero_or_nan_vectors_and_scores_fail_loudly():
    good = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite and non-zero"):
        retrieve.dense_scores(good, np.array([[0.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="finite and non-zero"):
        retrieve.dense_scores(good, np.array([[np.nan, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        retrieve.rank_row(np.array([0.1, np.nan, 0.3]), ["a", "b", "c"])


def test_load_cache_rejects_vectors_of_another_dimension(tmp_path):
    path = tmp_path / "cache.npz"
    retrieve.save_cache(path, {"k": np.ones(retrieve.EMBED_DIM + 1, dtype=np.float32)})
    with pytest.raises(RuntimeError, match="delete it to rebuild"):
        retrieve.load_cache(path)
    retrieve.save_cache(path, {"k": np.ones(retrieve.EMBED_DIM, dtype=np.float32)})
    assert list(retrieve.load_cache(path)) == ["k"]
