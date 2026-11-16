import re
import unicodedata

import pandas as pd


def words(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    return set(re.sub(r"\s+", " ", text).split())


chunks = pd.read_csv("data/chunks_v1.csv", dtype=str).fillna("")
qa = pd.read_csv("data/qa_v1.csv", dtype=str).fillna("")

gold = []
for _, q in qa.iterrows():
    ev_text = q["evidence_text"].strip()
    hits = []
    if ev_text and ev_text.upper() != "NONE":
        evidence = words(ev_text)
        for _, c in chunks[chunks.doc_id == q["doc_id"]].iterrows():
            if len(evidence & words(c["text"])) / len(evidence) >= 0.8:
                hits.append(c["chunk_id"])
    gold.append(";".join(hits))

qa["gold_chunk_ids"] = gold
qa.to_csv("data/eval_set_v1.csv", index=False)

missing = qa[qa["gold_chunk_ids"] == ""]
print(f"Saved data/eval_set_v1.csv. Questions with no matching chunk: {len(missing)}")
print(missing[["qa_id"]].to_string(index=False))
