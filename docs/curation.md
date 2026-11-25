# Corpus curation and evaluation protocol (M1)

This page describes how the 25 documents were chosen and how retrieval and answers are measured. The results are committed under `data/` and frozen; the files are listed in [data-and-run-outputs.md](data-and-run-outputs.md). The code is `src/curate.py` (selection) and `src/build_dataset.py` (chunks and eval set); the procedure to rebuild is [runbooks/rebuild-data.md](runbooks/rebuild-data.md).

## Which documents we use

- Dataset: `Madras1/rag-qa-fulltext-ptbr` (Brazilian Portuguese, licensed ODC-By: attribution required), configuration `documents`, pinned to revision `a9546f0d67416f3f2df57007f0ae60fde0cc0fab`.
- Rows are read by streaming, shuffled with seed 42 (buffer 10,000), so the pick is reproducible.
- We use 25 documents, IDs D001-D025, in `data/corpus_v2.jsonl`. They do not change after M1.

**Cheap filters** (applied before any LLM call). A row is rejected if it:
- is not `pt-br`, or its domain is not `tecnologia`
- has a `doc_type` other than tutorial, manual, guia or faq (`collect --include-doc-type` adds more)
- has fewer than 400 words (at least 2 chunks) or more than 1,500
- is empty or broken (replacement character, or under 60% letters, digits and spaces)
- is code heavy (more than 5 code-like tokens per 100 words)
- has the same text as an earlier document

**One QA per document.** The first QA pair of the row that has all of:
- a single evidence span, not truncated, with an answer of at least 8 characters
- a span that appears exactly once in the document, literally (text is NFC-normalized), and touches at most 50 words (equal to the chunk overlap, so a span always fits inside one chunk)
- a non-deictic question: it does not say "segundo o texto", "neste documento" and similar
- a question not used by an earlier document

**LLM judge.** Gemini (`gemini-3.5-flash-lite`, temperature 0, seed 42) must answer yes to: the document is technical, it is not news, opinion or marketing, the span supports the answer, and the question makes sense alone. The default model is `gemini-3.5-flash-lite` (free tier: 500 requests per day, 15 per minute) because the free tier of `gemini-3.6-flash` allows only 20 requests per day, which stopped the first run. Every judge call is logged in `data/curation_log.jsonl`.

**Human review gate.** The judge keeps a pool of 36 accepted documents. We read `data/review.md` and rejected 9 of them in `data/human_review.json`, each with a reason (marketing, answer not in the document, garbled text, and so on). `finalize` drops those and keeps the first 25.

## How we measure

**Chunks:** 300 words with 50 words of overlap, stored as character ranges of the NFC-normalized document text. Chunk ids are stable: `chk_<doc_id>_<index>_<first 8 hex of the sha256 of the chunk text>`.

**Gold:** every chunk of the document that contains the whole evidence span (`gold_chunk_ids`).

**Retrieval:** compare TF-IDF and dense search (same chunks, same k). The search covers the whole corpus (all chunks of all 25 documents), and the query is the question only. Score = evidence recall@k, for k = 1, 3, 5: whether a gold chunk is in the top k. With one evidence span per question this equals hit@k. MRR is reported in M2 ([retrieval.md](retrieval.md)).

**Answers:** grade 10 answers by hand: Correct / Partial / Wrong, and whether the cited chunk supports the answer (Yes / No). See [grading.md](grading.md).

**Paired questions:** 5 questions, answered with both retrievers. Chosen randomly (seed 42) with `random.Random(42).sample(sorted(qa_ids), 5)`: Q021, Q004, Q001, Q024, Q009 (`data/paired_questions_v2.json`). The first 3 are highlighted: Q021, Q004, Q001.

**Logs:** save question, retriever, chunk IDs, prompt, model settings and answer for every run.

**Failures:** analyze 2 wrong answers. Name 1 search risk and 1 fix.

## Validation

`python src/build_dataset.py check` validates the files and that a rebuild is identical; `tests/test_curate.py` and `tests/test_build_dataset.py` cover the rules above.

## Related

- [retrieval.md](retrieval.md), [rag-runner.md](rag-runner.md), [devcontainer.md](devcontainer.md).
