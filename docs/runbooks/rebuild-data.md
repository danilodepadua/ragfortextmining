# Rebuild the frozen data

Rebuilds the corpus, chunks, eval set and manifest under `data/`. The corpus is frozen and the rebuild changes the sha256 values in `data/MANIFEST.json`, so do it only deliberately. Selection rules are in [../curation.md](../curation.md).

## Prerequisites and context

- Context: the host (or a virtual environment with a writable `data/`), repository root as working directory. The devcontainer mounts `data/` read-only, so the write steps fail there ([../devcontainer.md](../devcontainer.md)).
- `GEMINI_API_KEY` in `.env` for `collect` only; network access to the dataset.
- Python 3.12 or newer (the pinned `numpy` requires it) with `pip install -r requirements.txt`.
- Commit or stash local work first: this procedure rewrites tracked files.

## Steps

1. Select documents: `python src/curate.py collect`. Options: `--resume` (continue an existing pool and log, for example after a quota error), `--overwrite` (replace them), `--pool-size` (accepted documents to keep, default 30), `--include-doc-type` (extra document type, repeatable), `--max-rows` (default 50,000), `--max-judge-calls` (default 100), `--sleep` (seconds between judge calls, default 4), `--model` (judge model, default `gemini-3.5-flash-lite`). Without `--resume` or `--overwrite` it refuses to run if a pool or log exists. Validation: the output says `accepted N documents` and `data/curation_log.jsonl` has the summary.
2. Write the selection (offline): `python src/curate.py finalize`. It takes the first 25 documents of the pool, writes `data/raw/selected_documents.jsonl`, `data/corpus_v2.jsonl` and `data/review.md`. Validation: the output says `wrote 25 documents`.
3. Read `data/review.md`. To reject a document, list its `document_id` in `data/human_review.json` (`{"rejected": ["doc-123"], "reasons": {"doc-123": "why"}}`) and run `python src/curate.py finalize` again. It fails if an id is not in the pool or fewer than 25 documents remain.
4. Write chunks, eval set and paired questions: `python src/build_dataset.py build`.
5. Validate: `python src/build_dataset.py check`. Stop if it fails.
6. Write the manifest: `python src/build_dataset.py manifest` (runs `check` first).
7. In the devcontainer, run `python -m pytest`. Inference: a changed corpus changes the chunks, so `python src/retrieve.py` and the RAG logs in `runs/` would have to be regenerated (the RAG runner refuses logs whose input file hashes differ).

## Stop conditions

- A quota error during `collect`: wait and use `--resume`.
- `check` fails: do not write the manifest.
- `collect` refuses because a pool or log exists: choose `--resume` or `--overwrite` knowingly.

## Expected result

`python src/build_dataset.py check` passes and `data/MANIFEST.json` matches the files.

## Recovery

`git restore data/` (on the host) returns the committed frozen files. The earlier English dataset can be recovered with `git show 046dcd2:<path>`.

## References

[../curation.md](../curation.md), [../data-and-run-outputs.md](../data-and-run-outputs.md).
