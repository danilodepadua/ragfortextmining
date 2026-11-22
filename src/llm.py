"""Thin Gemini wrapper. The client is injected so tests can use a fake."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel


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
    if not response.text:
        feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(f"empty response from {model} (prompt_feedback={feedback})")
    usage = getattr(response, "usage_metadata", None)
    meta = {
        "model_requested": model,
        "model_version": getattr(response, "model_version", None),
        "temperature": temperature,
        "seed": seed,
        "usage": usage.model_dump(exclude_none=True) if usage is not None else None,
    }
    return schema.model_validate_json(response.text), meta
