import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

# Dynamically resolve project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Locate input JSONL files (checking data/ first, then data/raw/ as fallback)
CHUNKS_JSONL = ROOT_DIR / "data" / "chunked_filtered_ragbench_tech.jsonl"
if not CHUNKS_JSONL.exists():
    CHUNKS_JSONL = ROOT_DIR / "data" / "raw" / "chunked_filtered_ragbench_tech.jsonl"

QA_JSONL = ROOT_DIR / "data" / "qa_v1.jsonl"
OUTPUT_JSONL = ROOT_DIR / "data" / "eval_set_v1.jsonl"
OUTPUT_CSV = ROOT_DIR / "data" / "eval_set_v1.csv"


def words(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    return set(re.sub(r"\s+", " ", text).split())


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(records).fillna("")


# Load chunks and Q&A datasets
chunks = load_jsonl(CHUNKS_JSONL)
qa = load_jsonl(QA_JSONL)

gold = []
for _, q in qa.iterrows():
    ev_text = str(q.get("evidence_text", "")).strip()
    hits = []
    if ev_text and ev_text.upper() != "NONE":
        evidence = words(ev_text)
        doc_chunks = chunks[chunks["doc_id"] == q["doc_id"]]
        for _, c in doc_chunks.iterrows():
            chunk_text = str(c["text"])
            if len(evidence) > 0 and (len(evidence & words(chunk_text)) / len(evidence)) >= 0.8:
                hits.append(str(c["chunk_id"]))
    gold.append(";".join(hits))

qa["gold_chunk_ids"] = gold

# Save evaluation set in both JSONL and CSV formats
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for record in qa.to_dict(orient="records"):
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

qa.to_csv(OUTPUT_CSV, index=False)

missing = qa[qa["gold_chunk_ids"] == ""]
print(f"Saved {OUTPUT_JSONL} and {OUTPUT_CSV}. Questions with no matching chunk: {len(missing)}")
if not missing.empty:
    print(missing[["qa_id"]].to_string(index=False))