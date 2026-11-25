"""Thin wrappers over Gemini and over a local Ollama server. The client is injected so tests can use a fake."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError


OLLAMA_DEFAULT_URL = "http://ragfortextmining-ollama:11434"
OLLAMA_TIMEOUT = 900  # seconds per call: a local CPU model can take minutes per answer


class GenerationError(RuntimeError):
    """A call that returned no usable JSON. `meta` keeps the raw text, finish reason and usage."""

    def __init__(self, message: str, meta: dict):
        super().__init__(message)
        self.meta = meta


def make_client(attempts: int = 5) -> genai.Client:
    """Build a client from GEMINI_API_KEY (environment or .env).

    The SDK retries 408/429/5xx with exponential backoff.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set: put it in .env (see .env.example)")
    retry = types.HttpRetryOptions(attempts=attempts)
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=retry))


def generate_json(
    client,
    model: str,
    prompt: str,
    schema: type[BaseModel],
    temperature: float = 0.0,
    seed: int = 42,
) -> tuple[BaseModel, dict]:
    """One structured call. Returns the validated object and metadata for the log."""
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    usage = getattr(response, "usage_metadata", None)
    candidates = getattr(response, "candidates", None) or []
    finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    meta = {
        "model_requested": model,
        "model_version": getattr(response, "model_version", None),
        "temperature": temperature,
        "seed": seed,
        "finish_reason": str(finish_reason) if finish_reason is not None else None,
        "raw_text": response.text,
        "usage": usage.model_dump(exclude_none=True) if usage is not None else None,
    }
    if not response.text:
        feedback = getattr(response, "prompt_feedback", None)
        raise GenerationError(f"empty response from {model} (prompt_feedback={feedback})", meta)
    try:
        return schema.model_validate_json(response.text), meta
    except ValidationError as exc:
        raise GenerationError(f"invalid JSON from {model}: {exc}", meta) from exc


class OllamaClient:
    """A local Ollama server, reached over HTTP with the standard library (no extra dependency)."""

    def __init__(self, base_url: str, timeout: float, version: str | None = None, digest: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.version = version
        self.digest = digest

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:  # the server's own explanation, for example a prompt over the context
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"Ollama answered HTTP {exc.code} for {path}: {detail}") from exc


def make_ollama_client(model: str, base_url: str | None = None) -> OllamaClient:
    """Connect to the Ollama server (OLLAMA_BASE_URL, default the local-llm container) and check the model.

    Fails with the command to run when the server is down or the model has not been pulled. The digest of the
    pulled model and the server version are kept on the client so that every record can log them.
    """
    load_dotenv()
    url = base_url or os.environ.get("OLLAMA_BASE_URL", "").strip() or OLLAMA_DEFAULT_URL
    client = OllamaClient(url, OLLAMA_TIMEOUT)
    try:
        client.version = client.request("GET", "/api/version").get("version")
        tags = client.request("GET", "/api/tags").get("models", [])
        if not isinstance(tags, list):
            raise ValueError("unexpected answer from /api/tags")
    except (urllib.error.URLError, OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
        # down, wrong URL (HTTP error) or an answer that is not an Ollama server's: say what to run either way
        raise RuntimeError(
            f"cannot use the Ollama server at {client.base_url} ({exc}): start it with "
            "`docker compose -f local-llm/compose.yaml up -d` and, from the devcontainer, attach it with "
            "`local-llm/connect-devcontainer.sh` (run on the host)"
        ) from exc
    wanted = model if ":" in model else f"{model}:latest"
    for entry in tags:
        if entry.get("name") == wanted or entry.get("model") == wanted:
            client.digest = entry.get("digest")
            return client
    raise RuntimeError(f"model {model} is not on {client.base_url}: run `docker exec ragfortextmining-ollama ollama pull {model}`")


def generate_json_ollama(
    client: OllamaClient,
    model: str,
    prompt: str,
    schema: type[BaseModel],
    temperature: float = 0.0,
    seed: int = 42,
    *,
    num_ctx: int,
    num_predict: int,
    think: bool,
) -> tuple[BaseModel, dict]:
    """One structured call to Ollama. Same contract as generate_json: (validated object, metadata for the log).

    num_ctx, num_predict and think have no defaults: they decide the answers and are logged as settings,
    so the caller (src/rag.py) states them once.

    `truncate` and `shift` are off, so a prompt that does not fit the context is an error from the server
    instead of silently losing retrieved chunks, and a full context ends the answer with finish_reason "length".
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": schema.model_json_schema(),
        "think": think,
        "truncate": False,
        "shift": False,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    response = client.request("POST", "/api/chat", body)
    message = response.get("message") or {}
    text = message.get("content")
    usage_keys = ("prompt_eval_count", "eval_count", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
    meta = {
        "provider": "ollama",
        "model_requested": model,
        "model_version": client.digest,
        "server_version": client.version,
        "base_url": client.base_url,
        "temperature": temperature,
        "seed": seed,
        "finish_reason": response.get("done_reason"),
        "raw_text": text,
        "thinking": message.get("thinking"),
        "usage": {key: response[key] for key in usage_keys if key in response},
    }
    if not text:
        raise GenerationError(f"empty response from {model}", meta)
    if response.get("done_reason") == "length":
        raise GenerationError(f"{model} reached the output limit (num_predict={num_predict}) before finishing", meta)
    try:
        return schema.model_validate_json(text), meta
    except ValidationError as exc:
        raise GenerationError(f"invalid JSON from {model}: {exc}", meta) from exc


def embed(client, model: str, texts: list[str], task_type: str, dim: int) -> tuple[np.ndarray, dict]:
    """One embedding request. Returns the raw vectors (float32) and metadata for the log.

    The vectors are exactly what the API returned: normalize where cosine is computed.
    Fails loudly unless there is exactly one vector per text (a multimodal embedding model
    silently returns a single embedding for a list of strings).
    """
    response = client.models.embed_content(
        model=model,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim),
    )
    vectors = [list(e.values) for e in (response.embeddings or [])]
    if len(vectors) != len(texts):
        raise RuntimeError(f"{model} returned {len(vectors)} embeddings for {len(texts)} texts")
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape[1] != dim:
        raise RuntimeError(f"{model} returned {array.shape[1]} dimensions, expected {dim}")
    usage = getattr(response, "usage_metadata", None)
    meta = {
        "model_requested": model,
        "task_type": task_type,
        "dim": dim,
        "usage": usage.model_dump(exclude_none=True) if usage is not None else None,
    }
    return array, meta
