import json
import re
from pathlib import Path

# Dynamically resolve project root
ROOT_DIR = Path(__file__).resolve().parent.parent

QA_JSONL = ROOT_DIR / "data" / "qa_v1.jsonl"
SRC_JSONL = ROOT_DIR / "data" / "raw" / "filtered_ragbench_tech.jsonl"
OUTPUT_TXT = ROOT_DIR / "data" / "evidence_candidates.txt"

STOP = set("the a an of to in and or is are for on with that this it be as at by from you your can how what why".split())


def toks(s):
    return {w for w in re.findall(r"[a-z0-9_.\-]+", str(s).lower()) if w not in STOP and len(w) > 2}


def pieces(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.split()) >= 6]


# 1. Load QA records from JSONL
if not QA_JSONL.exists():
    raise FileNotFoundError(f"Required file not found: {QA_JSONL}")
qa_records = [json.loads(line) for line in QA_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]

# 2. Load source documents and map doc_id (D001, D002...) to context text from filtered_ragbench_tech.jsonl
if not SRC_JSONL.exists():
    raise FileNotFoundError(f"Required file not found: {SRC_JSONL}")
raw_records = [json.loads(line) for line in SRC_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]

doc_map = {}
for i, r in enumerate(raw_records, start=1):
    doc_id = f"D{i:03d}"
    context = r.get("context", "")
    if isinstance(context, list):
        context = "\n\n".join(context)
    doc_map[doc_id] = context.strip()

# 3. Process candidate evidence extraction
lines = []
for q in qa_records:
    doc_id = q.get("doc_id", "")
    qa_id = q.get("qa_id", "")
    question = q.get("question", "")
    gold_answer = q.get("gold_answer", q.get("evidence_text", ""))

    doc = doc_map.get(doc_id, "")
    if not doc:
        continue

    target = toks(question) | toks(gold_answer)
    scored = []
    for p in pieces(doc):
        pt = toks(p)
        if pt:
            scored.append((len(pt & target) / len(pt) ** 0.5, p))
    scored.sort(reverse=True)
    
    lines.append(f"=== {qa_id} ({doc_id}) ===")
    lines.append("QUESTION: " + question[:300].replace("\n", " "))
    for n, (score, p) in enumerate(scored[:3], start=1):
        lines.append(f"  [{n}] ({score:.1f}) {p}")
    lines.append("")

# 4. Save output
OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved {OUTPUT_TXT}")