import re
import unicodedata
from pathlib import Path

import pandas as pd

CHUNK_WORDS = 200
OVERLAP = 50
RAW = Path("data/raw")


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


rows = []
step = CHUNK_WORDS - OVERLAP
for path in sorted(RAW.glob("D*.txt")):
    doc_id = path.stem
    words = normalize(path.read_text(encoding="utf-8")).split()
    for n, start in enumerate(range(0, max(len(words) - OVERLAP, 1), step), start=1):
        rows.append({"chunk_id": f"{doc_id}_C{n:02d}", "doc_id": doc_id, "start_word": start,
                     "text": " ".join(words[start:start + CHUNK_WORDS])})

pd.DataFrame(rows).to_csv("data/chunks_v1.csv", index=False)
print(f"Saved {len(rows)} chunks to data/chunks_v1.csv")
