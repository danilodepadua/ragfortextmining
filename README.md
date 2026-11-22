# RAG for text mining

Evaluation of retrieval for question answering over 25 Brazilian Portuguese technical documents. The v2 foundation (curated corpus, chunks, evaluation set) is committed and frozen. The two retrievers (M2) are implemented; the RAG pipeline (M3) is not.

# Configuring the environment

Run every command from the repository root (`python src/...`, not from `src`).

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

Only `python src/curate.py collect` and building or extending the embedding cache in `python src/retrieve.py` need the key (see M2: retrieval).

# Devcontainer (optional)

An alternative to the virtual environment above: Python 3.12 with all dependencies, `pytest`, `ipykernel` and `nbconvert` preinstalled.

- Open the repository in VS Code and choose **Reopen in Container**, or run `devcontainer up --workspace-folder .`.
- Put `GEMINI_API_KEY` in `.env` (see Step 4). The key is not forwarded from the host environment, so an empty host variable can never shadow the `.env` value.
- `data/` is mounted **read-only** inside the container: the v2 corpus is frozen (`data/MANIFEST.json` records the sha256 of every file), so `src/curate.py` and `src/build_dataset.py`, which write there, fail instead of changing it. To rebuild the corpus, run them on the host or remove the `data` entry from `mounts` in `.devcontainer/devcontainer.json`.
- Run git commands that change files under `data/` (`pull`, `checkout`, `merge`, `stash`) on the host: inside the container they fail on the read-only mount and can leave the working tree half-updated.
- Write run outputs (logs, answers) to `runs/`, which is writable and tracked in git.
- Run the tests in the container with `python -m pytest`.

# Protocol of document curating

## 1. Which documents we use

- Dataset: `Madras1/rag-qa-fulltext-ptbr` (Brazilian Portuguese, licensed ODC-By: attribution required), configuration `documents`, pinned to revision `a9546f0d67416f3f2df57007f0ae60fde0cc0fab`.
- Rows are read by streaming, shuffled with seed 42 (buffer 10,000), so the pick is reproducible.
- We use 25 documents, IDs D001-D025, in `data/corpus_v2.jsonl`. They do not change after M1.

**Cheap filters** (applied before any LLM call). A row is rejected if it:
- is not `pt-br`, or its domain is not `tecnologia`
- has a `doc_type` other than tutorial, manual, guia or faq (`--include-doc-type` adds more)
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

**Ranking file.** `runs/rankings_v2.jsonl` holds the full ranking of all 83 chunks for each of the 25 questions and both retrievers (50 lines). Each line has the question, the ranking (rank, chunk id, score), the retriever `config` and `chunks_sha256`. M3 will read it.

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

# Workflow

```
python src/curate.py collect         # stream, filter, judge (network + GEMINI_API_KEY)
python src/curate.py finalize        # offline: apply data/human_review.json, write the corpus
python src/build_dataset.py build    # write chunks, eval set and paired questions
python src/build_dataset.py check    # validate the files and that a rebuild is identical
python src/build_dataset.py manifest # write data/MANIFEST.json (runs check first)
python src/retrieve.py               # rank all chunks with both retrievers, write runs/ (key only if the cache lacks entries)
python -m pytest                     # tests
```

- `collect` options: `--resume` (continue an existing pool and log, for example after a quota error), `--overwrite` (replace them), `--pool-size` (accepted documents to keep, default 30), `--include-doc-type` (extra document type, repeatable). Without `--resume` or `--overwrite` it refuses to run if a pool or log exists.
- `data/` is read-only in the devcontainer, so `collect`, `finalize`, `build` and `manifest` run on the host. `check` and `python -m pytest` only read, so they also work in the container.
- `retrieve.py` option: `--pause` (seconds between embedding requests, default 15). It only reads `data/` and writes `runs/`, so it also runs in the container; it needs `GEMINI_API_KEY` only when the cache is missing entries.
- Rebuilding changes the hashes in `data/MANIFEST.json`: the corpus is frozen, so only do it deliberately.

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

The v1 data and scripts (English, RAGBench TechQA) were removed. Recover any v1 file with `git show 046dcd2:<path>`, for example `git show 046dcd2:data/corpus_v1.jsonl`.
