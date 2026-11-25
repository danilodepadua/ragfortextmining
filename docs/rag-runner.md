# RAG runner (M3)

`python src/rag.py` answers the 5 paired questions with both retrievers (10 answers per model) from retrieved chunks only. It reads `data/` (checked against `data/MANIFEST.json`) and `runs/rankings_v2.jsonl`, and writes only to `runs/`, so it also runs in the devcontainer. It needs `GEMINI_API_KEY` in `.env` for Gemini models that still have answers to make, and a running Ollama server for local models. The experiment design is in [rag-experiment.md](rag-experiment.md).

## Models

`rag_models.toml` has a required `run` list (the models answered by default) and one `[models."<name>"]` table per model with `temperature` (0 to 2), `seed`, `pause` (seconds between calls) and optionally `provider` (`gemini`, the default, or `ollama`). Unknown keys, a missing `run` or a `run` entry without a table are errors. Options:
- `--model a,b` runs a subset; the names must exist in the config, so a typo fails before any call.
- `--config PATH` uses another config file.
- `--pause` overrides the config's pause.
- `--dev` and `--dev-k` run development questions (see [rag-experiment.md](rag-experiment.md)).

To add a model, add a table with its API name (the Ollama tag for local models) and list it in `run` or pass `--model <name>`.

The committed config defines seven models, all with temperature 0 and seed 42. Only `gemini-3.5-flash-lite` is in `run`, so a bare `python src/rag.py` runs only that one; the others run with `--model`.

| Model | Provider | Pause |
| --- | --- | --- |
| `gemini-3.5-flash-lite` (default) | Gemini | 4 s |
| `gemini-3.1-flash-lite` | Gemini | 4 s |
| `gemma-4-26b-a4b-it` | Gemini | 4 s |
| `llama3.1:8b`, `qwen3:8b`, `gemma4:e4b`, `llama3.2:3b` | Ollama | 0 |

`gemini-2.5-flash-lite` was tried, but the API answers 404 "no longer available to new users", so it is not defined. Final runs (10 answers, no answer left failed) exist for all seven models. Gemini calls go through `google-genai`, which retries 408, 429 and 5xx errors; one call to `gemini-3.5-flash-lite` (Q024, TF-IDF) and one to `gemma-4-26b-a4b-it` (Q004, dense) still failed with a 503, were retried by a rerun and are kept in the `m3_runs_failures` files.

## Local models (Ollama)

The server in `local-llm/` serves open models on your own machine; setup, commands and warnings are in [../local-llm/README.md](../local-llm/README.md). Pull the tag first: the runner fails with the command to run when the server is down or the tag is missing. The server address comes from `OLLAMA_BASE_URL` (default `http://ragfortextmining-ollama:11434`).
- Fixed in `src/rag.py` and logged as settings: `num_ctx` 8192, `num_predict` 1024, thinking off. The request also sets `truncate` and `shift` off, so a prompt that does not fit the context is an error instead of silently dropping chunks, and an answer cut at `num_predict` is a failed attempt.
- The log records the model digest (`model_version`), the server version, usage (token counts and durations) and the thinking text if the model returns any.
- Digest check: at the start of a run, if the tag now points to other weights than the answers already in the log, the run refuses and asks you to delete the file, so one log never mixes two digests. The digest is checked once: a tag re-pulled in the middle of a run is not detected.
- Each local model was probed once on a non-paired question (`--dev Q006`, 2 answers, valid JSON, thinking off). The final runs took about 5 to 17 minutes per model (sum of the logged call latencies) on a CPU-only laptop with other applications open, so treat the speed as indicative only.
- The `done_reason=length` path (an answer cut at `num_predict`) is covered by unit tests only; no live call has hit it.

## Outputs

Every file under `runs/` carries the model name with characters outside `A-Za-z0-9._-` replaced by `_`; for example `m3_runs.gemini-3.5-flash-lite.jsonl`, and `llama3.1:8b` gives `m3_runs.llama3.1_8b.jsonl`. The full list is in [data-and-run-outputs.md](data-and-run-outputs.md). Each record of `m3_runs.<model>.jsonl` has the prompt text, retrieved chunks, raw response, finish reason, usage, citation checks and provenance (git HEAD, hashes of `src/rag.py` and `src/llm.py`, input file hashes).

## Rerun rules

Records are keyed by (`qa_id`, retriever).
- Per model: if all 10 answers succeeded, a rerun makes 0 calls (the Gemini `settings` keep exactly 5 keys, so the committed logs rerun with 0 calls); otherwise it calls only the missing or failed ones.
- A model refuses to run (delete its `runs/m3_runs.<model>.jsonl` to start over) if an existing record used a different prompt, settings, `g_id` or input files.
- A failed attempt is appended to the failures file and the log keeps the retry.
- A model that fails or is refused is reported and the other models still run; the exit code is 1 if any model failed.
- Five failed calls in a row abort that model's run; finished answers stay in the log. With a local server that hangs, that can take up to 5 x 900 s (the timeout per call is 900 s).

The step-by-step procedure is [runbooks/run-m3-models.md](runbooks/run-m3-models.md).

## Validation

`tests/test_rag.py` and `tests/test_llm.py` cover the runner and the wrappers; `python src/rag.py --help` lists the options above.
