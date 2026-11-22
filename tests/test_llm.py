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
