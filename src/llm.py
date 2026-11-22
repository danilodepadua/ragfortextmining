"""Thin Gemini wrapper. The client is injected so tests can use a fake."""

from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError


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
