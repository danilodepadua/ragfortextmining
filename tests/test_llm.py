import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import llm


class Answer(BaseModel):
    ok: bool


class FakeClient:
    def __init__(self, text):
        self.models = SimpleNamespace(generate_content=self._call)
        self.calls = []
        self._text = text

    def _call(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(model_dump=lambda exclude_none=True: {"total_token_count": 7})
        return SimpleNamespace(text=self._text, model_version="fake-1", usage_metadata=usage)


def test_generate_json_returns_object_and_metadata():
    client = FakeClient(json.dumps({"ok": True}))
    obj, meta = llm.generate_json(client, "m", "prompt", Answer)
    assert obj.ok is True
    assert meta["model_version"] == "fake-1" and meta["model_requested"] == "m"
    assert meta["usage"] == {"total_token_count": 7}
    config = client.calls[0]["config"]
    assert config.temperature == 0.0 and config.seed == 42


def test_generate_json_rejects_invalid_payload():
    with pytest.raises(Exception):
        llm.generate_json(FakeClient("not json"), "m", "p", Answer)


def test_make_client_requires_key(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "  ")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.make_client()


def test_generate_json_explains_an_empty_response():
    with pytest.raises(RuntimeError, match="empty response"):
        llm.generate_json(FakeClient(None), "m", "p", Answer)


class FakeEmbedClient:
    def __init__(self, vectors_per_call=None, dim=4):
        self.models = SimpleNamespace(embed_content=self._call)
        self.calls = []
        self.dim = dim
        self.vectors_per_call = vectors_per_call

    def _call(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        n = len(contents) if self.vectors_per_call is None else self.vectors_per_call
        embeddings = [SimpleNamespace(values=[float(i + 1)] * self.dim) for i in range(n)]
        return SimpleNamespace(embeddings=embeddings, usage_metadata=None)


def test_embed_returns_raw_vectors_and_metadata():
    client = FakeEmbedClient()
    vectors, meta = llm.embed(client, "m", ["a", "b", "c"], "RETRIEVAL_DOCUMENT", 4)
    assert vectors.shape == (3, 4) and vectors.dtype.name == "float32"
    assert vectors[2, 0] == 3.0  # raw values, not normalized
    assert meta == {"model_requested": "m", "task_type": "RETRIEVAL_DOCUMENT", "dim": 4, "usage": None}
    config = client.calls[0]["config"]
    assert config.task_type == "RETRIEVAL_DOCUMENT" and config.output_dimensionality == 4


def test_embed_rejects_a_wrong_vector_count():
    with pytest.raises(RuntimeError, match="1 embeddings for 3 texts"):
        llm.embed(FakeEmbedClient(vectors_per_call=1), "gemini-embedding-2", ["a", "b", "c"], "T", 4)


def test_embed_rejects_a_wrong_dimension():
    with pytest.raises(RuntimeError, match="expected 8"):
        llm.embed(FakeEmbedClient(dim=4), "m", ["a"], "T", 8)
