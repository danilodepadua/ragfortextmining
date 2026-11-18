import hashlib
import json
from pathlib import Path

import pandas as pd

SRC = Path("curated_documents/filtered_ragbench_tech.jsonl")
RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

records = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
print("Lines:", len(records))
print("Fields in each line:", list(records[0].keys()))

corpus_rows, qa_rows, seen = [], [], {}
for i, r in enumerate(records, start=1):
    doc_id = f"D{i:03d}"
    context = r.get("context", "")
    if isinstance(context, list):
        context = "\n\n".join(context)
    context = context.strip()

    answer = ""
    for key in ("answer", "response", "gold_answer"):
        if key in r:
            answer = r[key]
            break

    h = hashlib.md5(" ".join(context.lower().split()).encode()).hexdigest()
    if h in seen:
        print(f"WARNING: {doc_id} has the same text as {seen[h]}")
    else:
        seen[h] = doc_id

    (RAW / f"{doc_id}.txt").write_text(context, encoding="utf-8")
    corpus_rows.append({"doc_id": doc_id, "source_line": i, "word_count": len(context.split())})
    qa_rows.append({"qa_id": f"Q{i:03d}", "doc_id": doc_id, "question": r.get("question", "").strip(),
                    "gold_answer": answer, "evidence_text": ""})

pd.DataFrame(corpus_rows).to_csv("data/corpus_v1.csv", index=False)
pd.DataFrame(qa_rows).to_csv("data/qa_v1.csv", index=False)
print("Saved data/corpus_v1.csv, data/qa_v1.csv and the texts in data/raw/")
