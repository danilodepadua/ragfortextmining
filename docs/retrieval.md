# Retrieval (M2)

`python src/retrieve.py` ranks all chunks for each question with two retrievers, TF-IDF and dense, and writes the rankings and metrics. It reads `data/` (chunks and eval set, checked against `data/MANIFEST.json`) and writes three files under `runs/`: `rankings_v2.jsonl`, `retrieval_metrics.json` and the embedding cache `embeddings_gemini-embedding-001_3072.npz`. The protocol it implements is in [curation.md](curation.md).

## Retrievers

Settings were fixed before any result was seen and never tuned on the eval set.
- TF-IDF: scikit-learn `TfidfVectorizer(strip_accents='unicode', lowercase=True)`, fit on the chunks only, no stop words.
- Dense: `gemini-embedding-001`, 3072 dimensions. Chunks use task type `RETRIEVAL_DOCUMENT`, questions use `QUESTION_ANSWERING`. Cosine similarity in float64 after normalizing the vectors.

## Outputs

`runs/rankings_v2.jsonl` holds the full ranking of all 83 chunks for each of the 25 questions and both retrievers (50 lines). Each line has the question, the ranking (rank, chunk id, score), the retriever `config` and `chunks_sha256`. M3 ([rag-runner.md](rag-runner.md)) reads it.

`runs/retrieval_metrics.json` holds:
- Evidence recall@k for k = 1, 3, 5: a gold chunk is in the top k (equals hit@k).
- MRR over the full ranking (reciprocal rank of the first gold chunk).
- Random baseline: the exact expected value for a random ranking, not a simulation.
- Paired counts per k: questions hit by both, by dense only, by TF-IDF only, by neither.

## Results

25 questions, 83 chunks:

| Retriever | recall@1 | recall@3 | recall@5 | MRR |
| --- | --- | --- | --- | --- |
| TF-IDF | 0.640 | 1.000 | 1.000 | 0.800 |
| Dense | 0.640 | 0.960 | 1.000 | 0.803 |
| Random (exact) | 0.013 | 0.039 | 0.065 | 0.063 |

Paired at k = 1: 13 both, 3 dense only, 3 TF-IDF only, 6 neither. At k = 3: 24 both, 1 TF-IDF only. At k = 5: all 25 both.

Read these numbers with care:
- With n = 25, a difference of 1 to 3 questions is noise. The two retrievers are not separated by this data.
- Recall@5 is 1.000 for both, and k = 5 returns only about 6% of the 83 chunks, so the comparison shows mostly at k = 1 and in MRR.

## Cache and API use

Cache entries are keyed by content (task type plus chunk id, or the hash of the question) and the cache is committed. A rerun makes 0 API calls and reproduces the three files byte for byte (`.gitattributes` keeps their line endings), so `GEMINI_API_KEY` is only needed to build or extend the cache. Missing texts are embedded in batches of 20, with a pause between requests (`--pause`, default 15 seconds) to stay under the free-tier token limit. The cache is saved after every batch.

`retrieve.py` only reads `data/` and writes `runs/`, so it runs in the devcontainer too ([devcontainer.md](devcontainer.md)).

## Validation

`tests/test_retrieve.py` covers the retrievers and metrics; the byte-for-byte rerun is the check on the committed outputs.
