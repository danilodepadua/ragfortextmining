import re

import pandas as pd

STOP = set("the a an of to in and or is are for on with that this it be as at by from you your can how what why".split())


def toks(s):
    return {w for w in re.findall(r"[a-z0-9_.\-]+", str(s).lower()) if w not in STOP and len(w) > 2}


def pieces(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.split()) >= 6]


qa = pd.read_csv("data/qa_v1.csv").fillna("")
lines = []
for _, q in qa.iterrows():
    doc = open(f"data/raw/{q['doc_id']}.txt", encoding="utf-8").read()
    target = toks(q["question"]) | toks(q["gold_answer"])
    scored = []
    for p in pieces(doc):
        pt = toks(p)
        if pt:
            scored.append((len(pt & target) / len(pt) ** 0.5, p))
    scored.sort(reverse=True)
    lines.append(f"=== {q['qa_id']} ({q['doc_id']}) ===")
    lines.append("QUESTION: " + q["question"][:300].replace("\n", " "))
    for n, (score, p) in enumerate(scored[:3], start=1):
        lines.append(f"  [{n}] ({score:.1f}) {p}")
    lines.append("")

open("data/evidence_candidates.txt", "w", encoding="utf-8").write("\n".join(lines))
print("Saved data/evidence_candidates.txt")
