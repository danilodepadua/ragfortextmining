import json
from pathlib import Path

# Dynamically resolve project root
ROOT_DIR = Path(__file__).resolve().parent.parent

SRC_JSONL = ROOT_DIR / "data" / "raw" / "filtered_ragbench_tech.jsonl"
QA_JSONL = ROOT_DIR / "data" / "qa_v1.jsonl"

# 1. Load raw tech dataset records from JSONL
raw_records = []
if SRC_JSONL.exists():
    raw_records = [json.loads(line) for line in SRC_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]

# Build a lookup mapping questions (and positional index fallback) to responses/evidence
evidence_map = {}
for idx, r in enumerate(raw_records):
    q = r.get("question", "").strip()
    resp = r.get("response", "").strip()
    if q:
        evidence_map[q] = resp
    # Positional fallback (e.g., Q001 corresponds to index 0)
    evidence_map[f"idx_{idx}"] = resp

# 2. Load existing qa_v1.jsonl records
qa_records = []
if QA_JSONL.exists():
    qa_records = [json.loads(line) for line in QA_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]

# 3. Populate evidence_text
filled_count = 0
for idx, qa in enumerate(qa_records):
    q = qa.get("question", "").strip()
    # Match by question text, with positional index fallback
    evidence = evidence_map.get(q, "")
    if not evidence and f"idx_{idx}" in evidence_map:
        evidence = evidence_map[f"idx_{idx}"]
    
    if evidence:
        qa["evidence_text"] = evidence
        filled_count += 1
    else:
        qa["evidence_text"] = qa.get("gold_answer", "")

# 4. Save updated records back to qa_v1.jsonl
with open(QA_JSONL, "w", encoding="utf-8") as f:
    for qa in qa_records:
        f.write(json.dumps(qa, ensure_ascii=False) + "\n")

print(f"Filled {filled_count} of {len(qa_records)} records in {QA_JSONL}")