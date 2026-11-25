# Run or rerun the RAG runner for chosen models

Answers the 5 paired questions with both retrievers for one or more models and writes the logs and grading sheets under `runs/`.

## Prerequisites and context

- Context: devcontainer, repository root as working directory.
- Gemini models: `GEMINI_API_KEY` in `.env`. Free-tier limits apply; the pause comes from `rag_models.toml`.
- Local models: the Ollama server is running, the tag is pulled and the devcontainer is connected to its network; follow [../../local-llm/README.md](../../local-llm/README.md) first.
- The model has a `[models."<name>"]` table in `rag_models.toml` ([../rag-runner.md](../rag-runner.md)).

## Steps

1. For a new model or prompt change, probe one non-paired question: `python src/rag.py --dev Q006 --model <name>`. Validation: 2 answers in `runs/m3_dev_runs.<model>.jsonl` with valid JSON. `--dev` refuses paired ids.
2. Run the final answers: `python src/rag.py --model <name>` (comma-separate several names). Validation: the summary line reads `10 answers requested, written to m3_runs.<model>.jsonl` and `runs/m3_runs.<model>.jsonl` has 10 records.
3. Check `runs/m3_grading_sheet.<model>.md` exists. It is written only when all 10 answers succeeded.
4. If some answers failed, run step 2 again: only missing or failed answers are called. Failed attempts stay in `runs/m3_runs_failures.<model>.jsonl`.

## Stop conditions

- The runner refuses to run because the existing log used a different prompt, settings, `g_id` or input files, or (local models) the model digest changed. Decide whether to discard the old results; to start over delete `runs/m3_runs.<model>.jsonl`.
- Five failed calls in a row abort the model's run (up to 5 x 900 s with a hung local server). Finished answers stay in the log. Fix the cause (quota, server) and rerun.
- Exit code 1 means at least one model failed; the other models still ran.

## Expected result

Per model, `runs/m3_runs.<model>.jsonl` with 10 records and `runs/m3_grading_sheet.<model>.md`. Hand the sheets over as described in [../grading.md](../grading.md).

## Recovery

Delete `runs/m3_runs.<model>.jsonl` (and its failures file) and rerun to start a model over. Restore tracked outputs with `git restore runs/`.

## References

[../rag-runner.md](../rag-runner.md), [../rag-experiment.md](../rag-experiment.md).
