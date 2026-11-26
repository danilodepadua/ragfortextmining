# Reproduce the committed results

Reproduces retrieval and RAG outputs from the saved cache and logs, then regenerates the M4 summary and notebook figures from the returned blind grades. The workflow uses zero model API calls.

## Prerequisites and context

- Context: devcontainer (or a virtual environment with the dependencies from [README.md](../../README.md)), repository root as working directory.
- The saved embedding cache and seven complete answer logs are sufficient for replay. Jupyter and the simulated Ollama tests use local loopback sockets.

## Steps

1. Validate the frozen corpus with `python src/build_dataset.py check`.
2. To replay M2, run `python src/retrieve.py` from a disposable copy of the repository. Confirm `embedding API calls this run: 0` and compare the regenerated rankings and metrics byte for byte with the archived files.
3. In the same copy, replay M3 with `python src/rag.py --model gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemma-4-26b-a4b-it,llama3.1:8b,qwen3:8b,gemma4:e4b,llama3.2:3b`. Confirm every model reports `API calls this run: 0` and its grading sheet reproduces byte for byte.
4. Run `python src/evaluate.py` in the project checkout. It evaluates the seven configured models and writes `runs/m4_evaluation_summary.json`.
5. Run `python -m pytest` and `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/rag_evaluation.ipynb`. The executed cells regenerate the three PNG figures under `report/figures/`.
6. Optionally rebuild the two-page PDF with `python src/build_report.py` (enforces `<= 2` pages) and verify with `pdfinfo report/report.pdf`. [Report source](../../report/report.md)

## Stop conditions

- `retrieve.py` asks for `GEMINI_API_KEY`: the cache lacks entries, so the checkout or `data/` is not the committed one. Stop and compare with `git status`.
- `rag.py` refuses a model because of a different prompt, settings or input files: stop; see [../rag-runner.md](../rag-runner.md) (rerun rules).
- `evaluate.py` rejects a file whose hash, paired jobs or grades fail validation. Check the named file against the archived input or Member B's returned CSV.

## Expected result

All tests pass. The evaluator reports 70 grades across seven models, reproduces the summary byte for byte and writes only its M4 JSON file under `runs/`.

## Recovery

Compare any unexpected M2 or M3 differences with the archive in a fresh clone.

## References

[Retrieval](../retrieval.md) · [RAG runner](../rag-runner.md) · [M4 notebook](../../notebooks/rag_evaluation.ipynb)
