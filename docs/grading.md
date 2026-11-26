# Grading handoff and rubric

The RAG answers are graded by hand by a second person (the grader) from blind sheets, one per model. This page defines the sheet, the return format and the rubric. The answers come from [rag-runner.md](rag-runner.md).

## Status

Member B returned completed blind grades for all seven models, ten answers per model. `python src/evaluate.py` validates the exact CSV header, the G01–G10 coverage and the N/A rule before joining by `g_id`. The evaluation resolves the grade files through `rag_models.toml`. [Summary](../runs/m4_evaluation_summary.json) · [Executed notebook](../notebooks/rag_evaluation.ipynb)

## Handoff

The grader uses `runs/m3_grading_sheet.<model>.md`: answers labelled with grading ids `G01` to `G10` (the same ids for every model), the two answers to a question side by side in a seeded random order, with the question, the answer, the cited chunks and the gold answer and evidence span. The retriever is hidden; `g_id` is the key in the log. A sheet is written when all 10 answers of the model succeeded.

Blinding holds by trust: the grader must not open the log or read `src/rag.py` until the grades are returned. Grades come back in a separate file per model, `runs/m3_grades.<model>.csv`, with columns `g_id,grade,cited_supports,notes`.

## Rubric

Fixed before the paired answers were generated.
- Correct: same substance as the gold answer and no wrong claim.
- Partial: right but incomplete, or with an unsupported or wrong extra claim.
- Wrong: contradicts the gold answer, answers another question, or abstains. An abstention is never Correct; `gold_in_context` in the log says whether retrieval is to blame.
- `cited_supports`: Yes if the cited chunks together contain what the answer states, No otherwise, N/A when nothing is cited.

The blind K = 3 grades include 66 Correct, four Partial and zero Wrong. The report also discusses a Q019 development abstention at K = 1, which qualifies as Wrong under the rubric and sits outside the blind sample. [Failure analyses](../report/report.md#failure-analyses)

## Related

- [rag-experiment.md](rag-experiment.md) for the citation checks and what "paired" means.
- [data-and-run-outputs.md](data-and-run-outputs.md) for the file list.
