import hashlib
import json
from pathlib import Path

# Dynamically resolve project root (one level up from src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Correct paths anchored to the project root
SRC = ROOT_DIR / "data" / "raw" / "filtered_ragbench_tech.jsonl"
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Read records from the source JSONL file
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

    # Append structural metadata records
    corpus_rows.append({
        "doc_id": doc_id, 
        "source_line": i, 
        "word_count": len(context.split())
    })
    
    # Append Q&A record
    qa_rows.append({
        "qa_id": f"Q{i:03d}", 
        "doc_id": doc_id, 
        "question": r.get("question", "").strip(),
        "gold_answer": answer, 
        "evidence_text": ""
    })

# Write corpus and QA data as JSONL files
corpus_output_path = DATA_DIR / "corpus_v1.jsonl"
qa_output_path = DATA_DIR / "qa_v1.jsonl"

with open(corpus_output_path, "w", encoding="utf-8") as f:
    for row in corpus_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with open(qa_output_path, "w", encoding="utf-8") as f:
    for row in qa_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved {corpus_output_path} and {qa_output_path} successfully!")