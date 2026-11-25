# Reproduce the committed results

Rebuilds the retrieval and RAG runner outputs from a fresh clone using the committed embedding cache and run logs, so no API call is made.

## Prerequisites and context

- Context: devcontainer (or a virtual environment with the dependencies from [README.md](../../README.md)), repository root as working directory.
- `GEMINI_API_KEY` is not needed. No Ollama server is needed: logs with all 10 answers make 0 calls.

## Steps

1. Run the tests: `python -m pytest`. Validation: all tests pass. Stop if any fails.
2. Rank the chunks: `python src/retrieve.py`. Validation: the output ends with `embedding API calls this run: 0` and `git status --short runs/` shows no change (the outputs reproduce byte for byte).
3. Run every model: `python src/rag.py --model gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemma-4-26b-a4b-it,llama3.1:8b,qwen3:8b,gemma4:e4b,llama3.2:3b`. Validation: each model prints `API calls this run: 0` and the exit code is 0 (`echo $?`); `git status --short runs/` shows no change.

## Stop conditions

- `retrieve.py` asks for `GEMINI_API_KEY`: the cache lacks entries, so the checkout or `data/` is not the committed one. Stop and compare with `git status`.
- `rag.py` refuses a model because of a different prompt, settings or input files: stop; see [../rag-runner.md](../rag-runner.md) (rerun rules).

## Expected result

All tests pass and `runs/` is unchanged.

## Recovery

Restore changed outputs with `git restore runs/`.

## References

[../retrieval.md](../retrieval.md), [../rag-runner.md](../rag-runner.md).
