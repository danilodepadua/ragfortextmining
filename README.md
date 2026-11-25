# RAG for text mining

Evaluation of retrieval for question answering over 25 Brazilian Portuguese technical documents. The curated corpus, chunks and evaluation set are committed and frozen. The two retrievers and the RAG runner are implemented and merged; the RAG runner has 7 final runs (10 answers each) that are not graded yet (see [docs/grading.md](docs/grading.md)).

## Quick start

Run every command from the repository root (`python src/...`, not from `src`). Python 3.12 or newer is required (the pinned `numpy` needs it, and `rag_models.toml` is read with `tomllib`); the devcontainer has 3.12.

1. Create the environment, either a virtual environment or the devcontainer.
   - Virtual environment: `python -m venv venv`, activate it (`source venv/bin/activate` on macOS and Linux, `venv\Scripts\activate` on Windows), then `pip install -r requirements.txt -r .devcontainer/requirements-dev.txt`.
   - Devcontainer: open the repository in VS Code and choose **Reopen in Container**, or run `devcontainer up --workspace-folder .`. Python 3.12 and all dependencies are preinstalled. See [docs/devcontainer.md](docs/devcontainer.md).
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (Gemini replaced OpenAI). The key is needed only by `python src/curate.py collect`, by `python src/retrieve.py` when the embedding cache lacks entries (the committed cache does not), and by `python src/rag.py` for a Gemini model that still has answers to make. Local Ollama models need no key. `OLLAMA_BASE_URL` is optional (see [local-llm/README.md](local-llm/README.md)).
3. Check the setup with `python -m pytest`.

To reproduce the committed results from a fresh clone, follow [docs/runbooks/reproduce-results.md](docs/runbooks/reproduce-results.md).

## Documentation

| Document | Content |
| --- | --- |
| [docs/curation.md](docs/curation.md) | Document selection, filters, judge, human review, chunks, gold, evaluation protocol |
| [docs/retrieval.md](docs/retrieval.md) | TF-IDF and dense retrievers, rankings, metrics, results |
| [docs/rag-runner.md](docs/rag-runner.md) | `python src/rag.py`, models config, local models, rerun rules, outputs |
| [docs/rag-experiment.md](docs/rag-experiment.md) | Controlled experiment, prompt, protocol, citation checks, paired questions, caveats |
| [docs/grading.md](docs/grading.md) | Grading handoff, rubric and status |
| [docs/data-and-run-outputs.md](docs/data-and-run-outputs.md) | Files under `data/` and `runs/` |
| [docs/devcontainer.md](docs/devcontainer.md) | Devcontainer behavior, read-only `data/`, execution contexts |
| [docs/runbooks/README.md](docs/runbooks/README.md) | Index of step-by-step procedures |
| [local-llm/README.md](local-llm/README.md) | Runbook: optional Ollama server for local models |

## Layout

| Path | Content |
| --- | --- |
| `src/curate.py` | Corpus curation: stream, filter, judge (`collect`), apply the human review (`finalize`) |
| `src/build_dataset.py` | Chunks, eval set, paired questions, validation and manifest (`build`, `check`, `manifest`) |
| `src/retrieve.py` | TF-IDF and dense retrieval, rankings and metrics |
| `src/rag.py` | RAG runner, citation checks and grading sheets |
| `src/llm.py`, `src/common.py` | Gemini and Ollama wrappers; paths, JSONL I/O, chunking, manifest check |
| `rag_models.toml` | Models for the RAG runner and their settings |
| `tests/` | `pytest` tests (`pytest.ini` sets `pythonpath = src`) |
| `data/` | Frozen corpus and eval set |
| `runs/` | Retrieval and RAG runner outputs, tracked in git |
| `local-llm/` | Optional Ollama server for local models |
| `.devcontainer/` | Devcontainer configuration |

## Common commands

```
python src/retrieve.py                            # rank all chunks with both retrievers, write runs/ (key only if the cache lacks entries)
python src/rag.py                                 # answer the 5 paired questions with both retrievers (default model from rag_models.toml)
python src/rag.py --model llama3.1:8b             # a chosen model, here a local one (server and pulled tag needed)
python src/rag.py --dev Q006 --model llama3.1:8b  # development run on a non-paired question
python -m pytest                                  # tests
```

The data-building commands (`python src/curate.py ...`, `python src/build_dataset.py ...`) write to the frozen `data/` and run on the host; see [docs/runbooks/rebuild-data.md](docs/runbooks/rebuild-data.md). The earlier English dataset (RAGBench TechQA) and its scripts were removed; recover any of its files with `git show 046dcd2:<path>`, for example `git show 046dcd2:data/corpus_v1.jsonl`.
