# Grading handoff and rubric

The RAG answers are graded by hand by a second person (the grader) from blind sheets, one per model. This page defines the sheet, the return format and the rubric. The answers come from [rag-runner.md](rag-runner.md).

## Status

Sheets exist for all 7 models, but no grades have been returned yet, and which of the 7 models are graded is still to be agreed.

## Handoff

The grader uses `runs/m3_grading_sheet.<model>.md`: answers labelled with grading ids `G01` to `G10` (the same ids for every model), the two answers to a question side by side in a seeded random order, with the question, the answer, the cited chunks and the gold answer and evidence span. The retriever is hidden; `g_id` is the key in the log. A sheet is written when all 10 answers of the model succeeded.

Blinding holds by trust: the grader must not open the log or read `src/rag.py` until the grades are returned. Grades come back in a separate file per model, `runs/m3_grades.<model>.csv`, with columns `g_id,grade,cited_supports,notes`.

## Rubric

Written before any paired answer exists; for the team to confirm.
- Correct: same substance as the gold answer and no wrong claim.
- Partial: right but incomplete, or with an unsupported or wrong extra claim.
- Wrong: contradicts the gold answer, answers another question, or abstains. An abstention is never Correct; `gold_in_context` in the log says whether retrieval is to blame.
- `cited_supports`: Yes if the cited chunks together contain what the answer states, No otherwise, N/A when nothing is cited.

## Related

- [rag-experiment.md](rag-experiment.md) for the citation checks and what "paired" means.
- [data-and-run-outputs.md](data-and-run-outputs.md) for the file list.
