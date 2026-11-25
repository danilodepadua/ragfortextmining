# Data files and run outputs

This page lists the committed files under `data/` and `runs/`. `data/` is the frozen corpus and eval set: `data/MANIFEST.json` records the sha256 of every file, and the devcontainer mounts the folder read-only ([devcontainer.md](devcontainer.md)). `runs/` holds the retrieval and RAG runner outputs, tracked in git.

## Data files

| File | Content |
| --- | --- |
| `data/MANIFEST.json` | Dataset, revision, seed, judge models, chunk settings and sha256 of every file below |
| `data/raw/accepted_pool.jsonl` | The 36 documents accepted by the judge, with context, QA and span |
| `data/raw/selected_documents.jsonl` | The 25 final documents (D001-D025) with full text |
| `data/corpus_v2.jsonl` | Metadata of the 25 documents (no text) |
| `data/chunks_v2.jsonl` | Chunks: id, document, index, character range, word count, text |
| `data/eval_set_v2.jsonl`, `data/eval_set_v2.csv` | One question per document (Q001-Q025): gold answer, evidence span and offsets, gold chunk ids |
| `data/paired_questions_v2.json` | The 5 paired and 3 highlighted question ids |
| `data/curation_log.jsonl` | Run settings, every judge decision with model metadata, summary |
| `data/review.md` | Selected documents laid out for human review |
| `data/human_review.json` | Documents rejected in review, with reasons |

Rebuilding changes the hashes in `data/MANIFEST.json`: the corpus is frozen, so only do it deliberately ([runbooks/rebuild-data.md](runbooks/rebuild-data.md)).

## Run outputs

| File | Content |
| --- | --- |
| `runs/rankings_v2.jsonl` | Full ranking of all chunks per question and retriever (50 lines), with config and `chunks_sha256` |
| `runs/retrieval_metrics.json` | Evidence recall@k, MRR, first gold rank per question, exact random baseline, paired counts |
| `runs/embeddings_gemini-embedding-001_3072.npz` | Embedding cache (chunks and questions), content keyed |
| `runs/m3_runs.<model>.jsonl` | Full log per model: prompt, retrieved chunks, raw response, checks, provenance (10 records per model; 7 committed) |
| `runs/m3_runs_failures.<model>.jsonl` | Failed attempts, kept even after a retry succeeds (committed for `gemini-3.5-flash-lite` and `gemma-4-26b-a4b-it`; created on the first failure) |
| `runs/m3_grading_sheet.<model>.md` | Blind grading sheet per model (grading ids G01-G10), written when all 10 answers succeeded (7 committed) |
| `runs/m3_dev_runs.<model>.jsonl` | Development runs on non-paired questions (committed for `gemini-3.5-flash-lite` and the four local models) |
| `runs/m3_dev_runs_failures.<model>.jsonl` | Failed attempts of development runs (none committed) |
| `runs/m3_grades.<model>.csv` | Grades from the grader, per model: `g_id,grade,cited_supports,notes` (none yet) |

`<model>` is the model name made filename-safe (characters outside `A-Za-z0-9._-` become `_`). The 7 committed models are `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemma-4-26b-a4b-it`, `llama3.1_8b`, `qwen3_8b`, `gemma4_e4b` and `llama3.2_3b`.

Related: [retrieval.md](retrieval.md), [rag-runner.md](rag-runner.md), [grading.md](grading.md).
