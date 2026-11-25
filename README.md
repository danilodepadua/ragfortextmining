# RAG for text mining

Evaluation of retrieval for question answering over 25 Brazilian Portuguese technical documents. The v2 foundation (curated corpus, chunks, evaluation set) is committed and frozen. The two retrievers (M2) and the RAG runner (M3) are implemented and merged; M3 has 7 final runs (10 answers each) that are not graded yet (see Grading handoff).

# Layout

| Path | Content |
| --- | --- |
| `src/curate.py` | Corpus curation: stream, filter, judge (`collect`), apply the human review (`finalize`) |
| `src/build_dataset.py` | Chunks, eval set, paired questions, validation and manifest (`build`, `check`, `manifest`) |
| `src/retrieve.py` | TF-IDF and dense retrieval, rankings and metrics (M2) |
| `src/rag.py` | RAG runner, citation checks and grading sheets (M3) |
| `src/llm.py`, `src/common.py` | Gemini and Ollama wrappers; paths, JSONL I/O, chunking, manifest check |
| `rag_models.toml` | Models for M3 and their settings |
| `tests/` | `pytest` tests (`pytest.ini` sets `pythonpath = src`) |
| `data/` | Frozen v2 corpus and eval set (see Data files) |
| `runs/` | Retrieval and M3 outputs, tracked in git (see Run outputs) |
| `local-llm/` | Optional Ollama server for local models ([README](local-llm/README.md)) |
| `.devcontainer/` | Devcontainer (see below) |

# Configuring the environment

Run every command from the repository root (`python src/...`, not from `src`). Python 3.11 or newer is required (`rag_models.toml` is read with `tomllib`); the devcontainer has 3.12.

## Step 1: Create a virtual environment
```python -m venv venv```

## Step 2: Activate the virtual environment
- On macOS and Linux:
```source venv/bin/activate```

- On Windows (Command Prompt or PowerShell):
```venv\Scripts\activate```

## Step 3: Install the dependencies
```pip install -r requirements.txt```

To run the tests, also install the development dependencies:
```pip install -r .devcontainer/requirements-dev.txt```

## Step 4: Configure your environment variables
Copy `.env.example` to `.env` in the repository root and set your Gemini API key (Gemini replaced OpenAI):
```GEMINI_API_KEY=your_actual_gemini_api_key_here```

The key is needed only by `python src/curate.py collect`, by `python src/retrieve.py` when the embedding cache lacks entries (the committed cache does not), and by `python src/rag.py` for a Gemini model that still has answers to make. Local Ollama models need no key. `OLLAMA_BASE_URL` in `.env` is optional (see [local-llm/README.md](local-llm/README.md)).

## Step 5: Check the setup
```python -m pytest```

# Devcontainer (optional)

An alternative to the virtual environment above: Python 3.12 with all dependencies, `pytest`, `ipykernel` and `nbconvert` preinstalled.

- Open the repository in VS Code and choose **Reopen in Container**, or run `devcontainer up --workspace-folder .`.
- Put `GEMINI_API_KEY` in `.env` (see Step 4). The key is not forwarded from the host environment, so an empty host variable can never shadow the `.env` value.
- `data/` is mounted **read-only** inside the container: the v2 corpus is frozen (`data/MANIFEST.json` records the sha256 of every file), so `src/curate.py` and `src/build_dataset.py`, which write there, fail instead of changing it. To rebuild the corpus, run them on the host or remove the `data` entry from `mounts` in `.devcontainer/devcontainer.json`.
- Run git commands that change files under `data/` (`pull`, `checkout`, `merge`, `stash`) on the host: inside the container they fail on the read-only mount and can leave the working tree half-updated.
- Write run outputs (logs, answers) to `runs/`, which is writable and tracked in git.
- Run the tests in the container with `python -m pytest`.
- The optional Ollama server runs on the host, outside the devcontainer (see [local-llm/README.md](local-llm/README.md)).

# Protocol of document curating

## 1. Which documents we use

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

## 2. How we measure

**Chunks:** 300 words with 50 words of overlap, stored as character ranges of the NFC-normalized document text. Chunk ids are stable: `chk_<doc_id>_<index>_<first 8 hex of the sha256 of the chunk text>`.

**Gold:** every chunk of the document that contains the whole evidence span (`gold_chunk_ids`).

**Retrieval:** compare TF-IDF and dense search (same chunks, same k). The search covers the whole corpus (all chunks of all 25 documents), and the query is the question only.
Score = evidence recall@k, for k = 1, 3, 5: whether a gold chunk is in the top k. With one evidence span per question this equals hit@k. MRR is reported in M2.

**Answers:** grade 10 answers by hand.
- Correct / Partial / Wrong
- Does the cited chunk support the answer? Yes / No

**Paired questions:** 5 questions, answered with both retrievers.
Chosen randomly (seed 42) with `random.Random(42).sample(sorted(qa_ids), 5)`: Q021, Q004, Q001, Q024, Q009 (`data/paired_questions_v2.json`). The first 3 are highlighted: Q021, Q004, Q001.

**Logs:** save question, retriever, chunk IDs, prompt, model settings and answer for every run.

**Failures:** analyze 2 wrong answers. Name 1 search risk and 1 fix.

# M2: retrieval

`python src/retrieve.py` reads `data/` (chunks and eval set, checked against `data/MANIFEST.json`) and writes three files under `runs/`: `rankings_v2.jsonl`, `retrieval_metrics.json` and the embedding cache `embeddings_gemini-embedding-001_3072.npz`.

**Retrievers.** Settings were fixed before any result was seen and never tuned on the eval set.
- TF-IDF: scikit-learn `TfidfVectorizer(strip_accents='unicode', lowercase=True)`, fit on the chunks only, no stop words.
- Dense: `gemini-embedding-001`, 3072 dimensions. Chunks use task type `RETRIEVAL_DOCUMENT`, questions use `QUESTION_ANSWERING`. Cosine similarity in float64 after normalizing the vectors.

**Ranking file.** `runs/rankings_v2.jsonl` holds the full ranking of all 83 chunks for each of the 25 questions and both retrievers (50 lines). Each line has the question, the ranking (rank, chunk id, score), the retriever `config` and `chunks_sha256`. M3 reads it.

**Metrics** (`runs/retrieval_metrics.json`):
- Evidence recall@k for k = 1, 3, 5: a gold chunk is in the top k (equals hit@k).
- MRR over the full ranking (reciprocal rank of the first gold chunk).
- Random baseline: the exact expected value for a random ranking, not a simulation.
- Paired counts per k: questions hit by both, by dense only, by TF-IDF only, by neither.

**Results** (25 questions, 83 chunks):

| Retriever | recall@1 | recall@3 | recall@5 | MRR |
| --- | --- | --- | --- | --- |
| TF-IDF | 0.640 | 1.000 | 1.000 | 0.800 |
| Dense | 0.640 | 0.960 | 1.000 | 0.803 |
| Random (exact) | 0.013 | 0.039 | 0.065 | 0.063 |

Paired at k = 1: 13 both, 3 dense only, 3 TF-IDF only, 6 neither. At k = 3: 24 both, 1 TF-IDF only. At k = 5: all 25 both.

Read these numbers with care:
- With n = 25, a difference of 1 to 3 questions is noise. The two retrievers are not separated by this data.
- Recall@5 is 1.000 for both, and k = 5 returns only about 6% of the 83 chunks, so the comparison shows mostly at k = 1 and in MRR.

**Cache and API use.** Cache entries are keyed by content (task type plus chunk id, or the hash of the question) and the cache is committed. A rerun makes 0 API calls and reproduces the three files byte for byte (`.gitattributes` keeps their line endings), so `GEMINI_API_KEY` is only needed to build or extend the cache. Missing texts are embedded in batches of 20, with a pause between requests (`--pause`, default 15 seconds) to stay under the free-tier token limit. The cache is saved after every batch.

# M3: RAG answers

`python src/rag.py` answers the 5 paired questions with both retrievers (10 answers per model) from retrieved chunks only. It reads `data/` (checked against `data/MANIFEST.json`) and `runs/rankings_v2.jsonl`. It needs `GEMINI_API_KEY` in `.env` for Gemini models that still have answers to make, and a running Ollama server for local models.

**Models.** `rag_models.toml` has a required `run` list (the models answered by default) and one `[models."<name>"]` table per model with `temperature` (0 to 2), `seed`, `pause` (seconds between calls) and optionally `provider` (`gemini`, the default, or `ollama`). Unknown keys, a missing `run` or a `run` entry without a table are errors. Options:
- `--model a,b` runs a subset; the names must exist in the config, so a typo fails before any call.
- `--config PATH` uses another config file.
- `--pause` overrides the config's pause.

To add a model, add a table with its API name (the Ollama tag for local models) and list it in `run` or pass `--model <name>`.

The committed config defines seven models, all with temperature 0 and seed 42. Only `gemini-3.5-flash-lite` is in `run`, so a bare `python src/rag.py` runs only that one; the others run with `--model`.

| Model | Provider | Pause |
| --- | --- | --- |
| `gemini-3.5-flash-lite` (default) | Gemini | 4 s |
| `gemini-3.1-flash-lite` | Gemini | 4 s |
| `gemma-4-26b-a4b-it` | Gemini | 4 s |
| `llama3.1:8b`, `qwen3:8b`, `gemma4:e4b`, `llama3.2:3b` | Ollama | 0 |

`gemini-2.5-flash-lite` was tried, but the API answers 404 "no longer available to new users", so it is not defined. Final runs (10 answers, no answer left failed) exist for all seven models. Gemini calls go through `google-genai`, which retries 408, 429 and 5xx errors; one call to `gemini-3.5-flash-lite` (Q024, TF-IDF) and one to `gemma-4-26b-a4b-it` (Q004, dense) still failed with a 503, were retried by a rerun and are kept in the `m3_runs_failures` files.

**Local models (Ollama).** The server in `local-llm/` serves open models on your own machine; setup, commands and warnings are in [local-llm/README.md](local-llm/README.md). Pull the tag first: the runner fails with the command to run when the server is down or the tag is missing. The server address comes from `OLLAMA_BASE_URL` (default `http://ragfortextmining-ollama:11434`).
- Fixed in `src/rag.py` and logged as settings: `num_ctx` 8192, `num_predict` 1024, thinking off. The request also sets `truncate` and `shift` off, so a prompt that does not fit the context is an error instead of silently dropping chunks, and an answer cut at `num_predict` is a failed attempt.
- The log records the model digest (`model_version`), the server version, usage (token counts and durations) and the thinking text if the model returns any.
- Digest check: at the start of a run, if the tag now points to other weights than the answers already in the log, the run refuses and asks you to delete the file, so one log never mixes two digests. The digest is checked once: a tag re-pulled in the middle of a run is not detected.
- Each local model was probed once on a non-paired question (`--dev Q006`, 2 answers, valid JSON, thinking off). The final runs took about 5 to 17 minutes per model (sum of the logged call latencies) on a CPU-only laptop with other applications open, so treat the speed as indicative only.
- The `done_reason=length` path (an answer cut at `num_predict`) is covered by unit tests only; no live call has hit it.

**Controlled experiment.** The prompt, K = 3, the retrievers and the questions are frozen in the code; only the model, with its logged temperature and seed, varies. Disclosure: the prompt was developed on `gemini-3.5-flash-lite` only and is applied unchanged to every other model (including the local ones), so differences between models mix model quality with prompt fit.

**Prompt and settings.** The top K = 3 chunks of each retriever go into the prompt in rank order, labelled `[1]` to `[3]`; the code maps the numbers back to chunk ids. One Portuguese prompt template serves both retrievers. The model returns structured output with `answerable`, `cited` (chunk numbers) and `answer`, in that order. The prompt version is a hash of the template and the schema (`13c74fac6aa7`); every committed run uses it.

**Outputs** (under `runs/`). Every file carries the model name with characters outside `A-Za-z0-9._-` replaced by `_`; for example `m3_runs.gemini-3.5-flash-lite.jsonl`, and `llama3.1:8b` gives `m3_runs.llama3.1_8b.jsonl`. The full list is in Run outputs. Each record of `m3_runs.<model>.jsonl` has the prompt text, retrieved chunks, raw response, finish reason, usage, citation checks and provenance (git HEAD, hashes of `src/rag.py` and `src/llm.py`, input file hashes).

**Rerun.** Records are keyed by (`qa_id`, retriever). Per model: if all 10 answers succeeded, a rerun makes 0 calls (the Gemini `settings` keep exactly 5 keys, so the committed logs rerun with 0 calls); otherwise it calls only the missing or failed ones. A model refuses to run (delete its `runs/m3_runs.<model>.jsonl` to start over) if an existing record used a different prompt, settings, `g_id` or input files. A failed attempt is appended to the failures file and the log keeps the retry. A model that fails or is refused is reported and the other models still run; the exit code is 1 if any model failed. Five failed calls in a row abort that model's run; finished answers stay in the log. With a local server that hangs, that can take up to 5 x 900 s (the timeout per call is 900 s).

**Protocol.** The prompt and settings were developed only on non-paired questions and frozen before the paired run: `python src/rag.py --dev Q006,Q002,Q012`, plus an abstention probe, `--dev Q019 --dev-k 1`, to exercise `answerable = false`. `--dev` refuses paired ids and is the only place a k other than 3 is allowed (`--dev-k`). The development log `runs/m3_dev_runs.gemini-3.5-flash-lite.jsonl` is committed as evidence. K = 3 was chosen on protocol grounds (the middle of 1, 3 and 5). Disclosure: the author had already looked at the ranks of the paired questions while sizing the options.

**Citation checks** in each record: `cited_numbers` mapped to `cited_chunk_ids`, `invalid_cited_numbers`, `cites_gold`, `gold_in_context` and `consistent` (an answer with no valid citation, or a refusal that cites, is inconsistent). These are auxiliary signals only; failure classification in M4 rests on the human grade.

**What "paired" means here.**
- Q009: both retrievers give the same chunks in the same order, so the prompts are byte-identical (a determinism check).
- Q004: same chunks, different order.
- Only Q021, Q001 and Q024 compare retrieval. The highlighted questions are Q021, Q004 and Q001.
- With n = 5 the comparison is qualitative only; nothing is statistical.

**Grading handoff.** The grader (Person B) uses `runs/m3_grading_sheet.<model>.md`: answers `G01` to `G10` (the same ids for every model), the two answers to a question side by side in a seeded random order, with the question, the answer, the cited chunks and the gold answer and evidence span. The retriever is hidden; `g_id` is the key in the log. Blinding holds by trust: the grader must not open the log or read `src/rag.py` until the grades are returned. Grades come back in a separate file per model, `runs/m3_grades.<model>.csv`, with columns `g_id,grade,cited_supports,notes`. Status: sheets exist for all 7 models, but no grades have been returned yet, and which of the 7 models are graded is still to be agreed.

**Rubric** (written before any paired answer exists; for the team to confirm):
- Correct: same substance as the gold answer and no wrong claim.
- Partial: right but incomplete, or with an unsupported or wrong extra claim.
- Wrong: contradicts the gold answer, answers another question, or abstains. An abstention is never Correct; `gold_in_context` in the log says whether retrieval is to blame.
- `cited_supports`: Yes if the cited chunks together contain what the answer states, No otherwise, N/A when nothing is cited.

**Caveats.**
- Temperature 0 was chosen for determinism, although Google's Gemini 3 guidance prefers the default temperature 1.0.
- Hosted LLMs are not bit-reproducible, even with temperature 0 and a seed. The experiment is controlled (same prompt, model and settings, all logged), not exactly repeatable. The log records the served model version for the `gemini-3.5-flash-lite` alias.

# Workflow

```
python src/curate.py collect         # stream, filter, judge (network + GEMINI_API_KEY)
python src/curate.py finalize        # offline: apply data/human_review.json, write the corpus
python src/build_dataset.py build    # write chunks, eval set and paired questions
python src/build_dataset.py check    # validate the files and that a rebuild is identical
python src/build_dataset.py manifest # write data/MANIFEST.json (runs check first)
python src/retrieve.py               # rank all chunks with both retrievers, write runs/ (key only if the cache lacks entries)
python src/rag.py                    # answer the 5 paired questions with both retrievers (default model from rag_models.toml)
python src/rag.py --model llama3.1:8b          # a chosen model, here a local one (server and pulled tag needed)
python src/rag.py --dev Q006 --model llama3.1:8b  # development run on a non-paired question
python -m pytest                     # tests
```

- `collect` options: `--resume` (continue an existing pool and log, for example after a quota error), `--overwrite` (replace them), `--pool-size` (accepted documents to keep, default 30), `--include-doc-type` (extra document type, repeatable), `--max-rows` (default 50,000), `--max-judge-calls` (default 100), `--sleep` (seconds between judge calls, default 4) and `--model` (judge model, default `gemini-3.5-flash-lite`). Without `--resume` or `--overwrite` it refuses to run if a pool or log exists.
- `data/` is read-only in the devcontainer, so `collect`, `finalize`, `build` and `manifest` run on the host. `check` and `python -m pytest` only read, so they also work in the container.
- `retrieve.py` option: `--pause` (seconds between embedding requests, default 15). It only reads `data/` and writes `runs/`, so it also runs in the container; it needs `GEMINI_API_KEY` only when the cache is missing entries.
- `rag.py` options are described in M3: `--model`, `--config`, `--pause`, `--dev`, `--dev-k`. It writes only to `runs/`, so it also runs in the container.
- Rebuilding changes the hashes in `data/MANIFEST.json`: the corpus is frozen, so only do it deliberately.

To reproduce the committed results from a fresh clone, run `python -m pytest`, `python src/retrieve.py` and `python src/rag.py --model <each of the seven>`: with the committed cache and logs they make 0 API calls.

# Data files

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

# Run outputs

| File | Content |
| --- | --- |
| `runs/rankings_v2.jsonl` | Full ranking of all chunks per question and retriever (50 lines), with config and `chunks_sha256` |
| `runs/retrieval_metrics.json` | Evidence recall@k, MRR, first gold rank per question, exact random baseline, paired counts |
| `runs/embeddings_gemini-embedding-001_3072.npz` | Embedding cache (chunks and questions), content keyed |
| `runs/m3_runs.<model>.jsonl` | M3 full log per model: prompt, retrieved chunks, raw response, checks, provenance (10 records per model; 7 committed) |
| `runs/m3_runs_failures.<model>.jsonl` | M3 failed attempts, kept even after a retry succeeds (committed for `gemini-3.5-flash-lite` and `gemma-4-26b-a4b-it`; created on the first failure) |
| `runs/m3_grading_sheet.<model>.md` | M3 blind grading sheet per model (G01-G10), written when all 10 answers succeeded (7 committed) |
| `runs/m3_dev_runs.<model>.jsonl` | M3 development runs on non-paired questions (committed for `gemini-3.5-flash-lite` and the four local models) |
| `runs/m3_dev_runs_failures.<model>.jsonl` | Failed attempts of development runs (none committed) |
| `runs/m3_grades.<model>.csv` | Grades from the grader, per model: `g_id,grade,cited_supports,notes` (none yet) |

`<model>` is the model name made filename-safe (see M3: Outputs). The 7 committed models are `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemma-4-26b-a4b-it`, `llama3.1_8b`, `qwen3_8b`, `gemma4_e4b` and `llama3.2_3b`.

The v1 data and scripts (English, RAGBench TechQA) were removed. Recover any v1 file with `git show 046dcd2:<path>`, for example `git show 046dcd2:data/corpus_v1.jsonl`.
