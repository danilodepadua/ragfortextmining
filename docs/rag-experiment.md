# M3 experiment design

The M3 runner ([rag-runner.md](rag-runner.md)) is a controlled experiment: the prompt, K = 3, the retrievers and the questions are frozen in `src/rag.py`; only the model, with its logged temperature and seed, varies. This page records the prompt, the protocol that froze it, the citation checks and the limits of the comparison.

## Controlled experiment

Disclosure: the prompt was developed on `gemini-3.5-flash-lite` only and is applied unchanged to every other model (including the local ones), so differences between models mix model quality with prompt fit.

## Prompt and settings

The top K = 3 chunks of each retriever go into the prompt in rank order, labelled `[1]` to `[3]`; the code maps the numbers back to chunk ids. One Portuguese prompt template serves both retrievers. The model returns structured output with `answerable`, `cited` (chunk numbers) and `answer`, in that order. The prompt version is a hash of the template and the schema (`13c74fac6aa7`); every committed run uses it.

## Protocol

The prompt and settings were developed only on non-paired questions and frozen before the paired run: `python src/rag.py --dev Q006,Q002,Q012`, plus an abstention probe, `--dev Q019 --dev-k 1`, to exercise `answerable = false`. `--dev` refuses paired ids and is the only place a k other than 3 is allowed (`--dev-k`). The development log `runs/m3_dev_runs.gemini-3.5-flash-lite.jsonl` is committed as evidence. K = 3 was chosen on protocol grounds (the middle of 1, 3 and 5). Disclosure: the author had already looked at the ranks of the paired questions while sizing the options.

## Citation checks

Each record has `cited_numbers` mapped to `cited_chunk_ids`, `invalid_cited_numbers`, `cites_gold`, `gold_in_context` and `consistent` (an answer with no valid citation, or a refusal that cites, is inconsistent). These are auxiliary signals only; failure classification in M4 rests on the human grade ([grading.md](grading.md)).

## What "paired" means here

- Q009: both retrievers give the same chunks in the same order, so the prompts are byte-identical (a determinism check).
- Q004: same chunks, different order.
- Only Q021, Q001 and Q024 compare retrieval. The highlighted questions are Q021, Q004 and Q001.
- With n = 5 the comparison is qualitative only; nothing is statistical.

## Caveats

- Temperature 0 was chosen for determinism, although Google's Gemini 3 guidance prefers the default temperature 1.0.
- Hosted LLMs are not bit-reproducible, even with temperature 0 and a seed. The experiment is controlled (same prompt, model and settings, all logged), not exactly repeatable. The log records the served model version for the `gemini-3.5-flash-lite` alias.
