import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def test_generate_json_failures_keep_the_raw_text_and_finish_reason():
    client = FakeClient("{not json")
    with pytest.raises(llm.GenerationError, match="invalid JSON") as info:
        llm.generate_json(client, "m", "p", Answer)
    assert info.value.meta["raw_text"] == "{not json" and info.value.meta["model_version"] == "fake-1"
    assert isinstance(info.value, RuntimeError)

    empty = FakeClient(None)
    with pytest.raises(llm.GenerationError, match="empty response") as info:
        llm.generate_json(empty, "m", "p", Answer)
    assert info.value.meta["raw_text"] is None


def test_generate_json_reports_the_finish_reason_when_the_api_gives_one():
    client = FakeClient(json.dumps({"ok": True}))
    original = client._call

    def with_candidates(**kwargs):
        response = original(**kwargs)
        response.candidates = [SimpleNamespace(finish_reason="STOP")]
        return response

    client.models = SimpleNamespace(generate_content=with_candidates)
    obj, meta = llm.generate_json(client, "m", "p", Answer)
    assert meta["finish_reason"] == "STOP" and meta["raw_text"] == '{"ok": true}'
    assert llm.generate_json(FakeClient(json.dumps({"ok": True})), "m", "p", Answer)[1]["finish_reason"] is None


# ---------------------------------------------------------------- Ollama (local server)


class FakeOllama:
    def __init__(self, message=None, done_reason="stop"):
        self.digest, self.version, self.base_url = "sha256:abc", "0.40.18", "http://fake:11434"
        self.bodies = []
        self._message = {"content": json.dumps({"ok": True})} if message is None else message
        self._done = done_reason

    def request(self, method, path, body=None):
        self.bodies.append((method, path, body))
        return {"message": self._message, "done_reason": self._done, "prompt_eval_count": 11, "eval_count": 3}


def test_ollama_request_is_structured_unthinking_and_never_truncates():
    client = FakeOllama({"content": json.dumps({"ok": True}), "thinking": "hm"})
    obj, meta = llm.generate_json_ollama(client, "llama3.1:8b", "prompt", Answer, 0.0, 42, num_ctx=4096, num_predict=99, think=False)
    assert obj.ok is True
    method, path, body = client.bodies[0]
    assert (method, path) == ("POST", "/api/chat")
    assert body["format"] == Answer.model_json_schema() and body["stream"] is False
    assert body["think"] is False and body["truncate"] is False and body["shift"] is False
    assert body["options"] == {"temperature": 0.0, "seed": 42, "num_ctx": 4096, "num_predict": 99}
    assert meta["model_version"] == "sha256:abc" and meta["server_version"] == "0.40.18"
    assert meta["finish_reason"] == "stop" and meta["thinking"] == "hm" and meta["usage"]["prompt_eval_count"] == 11


@pytest.mark.parametrize(
    "message, done, expected",
    [({"content": "not json"}, "stop", "invalid JSON"), ({"content": ""}, "stop", "empty response"),
     ({"content": "{\"ok\": tr"}, "length", "output limit")],
)
def test_ollama_failures_keep_the_raw_text_and_say_why(message, done, expected):
    with pytest.raises(llm.GenerationError, match=expected) as exc:
        llm.generate_json_ollama(FakeOllama(message, done), "m", "p", Answer, num_ctx=4096, num_predict=99, think=False)
    assert exc.value.meta["finish_reason"] == done and exc.value.meta["raw_text"] == message["content"]


class Handler(BaseHTTPRequestHandler):
    routes: dict = {}

    def log_message(self, *args):
        pass

    def _serve(self):
        status, payload, delay = self.routes.get(self.path, (404, {"error": "no such route"}, 0))
        time.sleep(delay)
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = do_POST = _serve


@pytest.fixture
def server():
    handler = type("H", (Handler,), {"routes": {}})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield handler.routes, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_make_ollama_client_reads_version_and_digest_and_normalises_latest(server):
    routes, url = server
    routes["/api/version"] = (200, {"version": "0.40.18"}, 0)
    routes["/api/tags"] = (200, {"models": [{"name": "llama3.1:8b", "digest": "sha256:aa"}, {"name": "tiny:latest", "digest": "sha256:bb"}]}, 0)
    client = llm.make_ollama_client("llama3.1:8b", base_url=url)
    assert client.version == "0.40.18" and client.digest == "sha256:aa"
    assert llm.make_ollama_client("tiny", base_url=url).digest == "sha256:bb"


def test_make_ollama_client_says_what_to_run_when_the_model_is_not_pulled(server):
    routes, url = server
    routes["/api/version"] = (200, {"version": "0.40.18"}, 0)
    routes["/api/tags"] = (200, {"models": []}, 0)
    with pytest.raises(RuntimeError, match="ollama pull qwen3:8b"):
        llm.make_ollama_client("qwen3:8b", base_url=url)


@pytest.mark.parametrize(
    "version, tags",
    [((404, {"error": "no such route"}, 0), (200, {"models": []}, 0)),  # wrong URL
     ((200, {"version": "x"}, 0), (200, {"models": None}, 0)),  # not an Ollama answer
     ((200, ["not", "a", "dict"], 0), (200, {"models": []}, 0))],
)
def test_make_ollama_client_hints_at_the_fix_for_a_wrong_or_odd_server(server, version, tags):
    routes, url = server
    routes["/api/version"], routes["/api/tags"] = version, tags
    with pytest.raises(RuntimeError, match="docker compose -f local-llm/compose.yaml up -d"):
        llm.make_ollama_client("m", base_url=url)


def test_make_ollama_client_says_how_to_start_the_server_when_it_is_unreachable():
    with pytest.raises(RuntimeError, match="docker compose -f local-llm/compose.yaml up -d"):
        llm.make_ollama_client("m", base_url="http://127.0.0.1:9")


def test_make_ollama_client_takes_the_url_from_the_environment(server, monkeypatch):
    routes, url = server
    routes["/api/version"] = (200, {"version": "x"}, 0)
    routes["/api/tags"] = (200, {"models": [{"name": "m:latest", "digest": "d"}]}, 0)
    monkeypatch.setenv("OLLAMA_BASE_URL", url + "/")
    assert llm.make_ollama_client("m").base_url == url


def test_an_http_error_carries_the_servers_message(server):
    routes, url = server
    routes["/api/chat"] = (400, {"error": "the prompt is longer than the context length"}, 0)
    client = llm.OllamaClient(url, timeout=5, digest="d")
    with pytest.raises(RuntimeError, match="HTTP 400.*longer than the context length"):
        llm.generate_json_ollama(client, "m", "p", Answer, num_ctx=4096, num_predict=99, think=False)


def test_a_slow_server_times_out(server):
    routes, url = server
    routes["/api/chat"] = (200, {"message": {"content": "{}"}}, 1.0)
    client = llm.OllamaClient(url, timeout=0.2, digest="d")
    with pytest.raises(OSError):  # socket.timeout is a TimeoutError
        llm.generate_json_ollama(client, "m", "p", Answer, num_ctx=4096, num_predict=99, think=False)
