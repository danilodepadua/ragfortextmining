"""Shared helpers: paths, JSONL I/O, text normalization and chunking."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"

CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50

_WORD = re.compile(r"\S+")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # LF everywhere: byte-stable output
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    """NFC only. NFKC would rewrite Portuguese text ("nº" -> "no") and shift offsets."""
    return unicodedata.normalize("NFC", text or "")


def word_spans(text: str) -> list[tuple[int, int]]:
    """Character span (start, end) of every whitespace-delimited word."""
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def words_touched(text: str, start: int, end: int) -> int:
    """Number of words that overlap the character range [start, end)."""
    return sum(1 for s, e in word_spans(text) if s < end and e > start)


def count_occurrences(text: str, needle: str) -> int:
    """Occurrences of `needle` in `text`, overlapping ones included."""
    return len(re.findall("(?=" + re.escape(needle) + ")", text))


def chunk_spans(
    text: str,
    size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[dict]:
    """Overlapping word windows of `text`, each as a character range of `text`."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = word_spans(text)
    chunks = []
    for index, first in enumerate(range(0, len(words), size - overlap)):
        last = min(first + size, len(words))
        chunks.append(
            {
                "chunk_index": index,
                "char_start": words[first][0],
                "char_end": words[last - 1][1],
                "word_count": last - first,
            }
        )
        if last == len(words):
            break
    return chunks


def stable_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
    digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:8]
    return f"chk_{doc_id}_{chunk_index:03d}_{digest}"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(data_dir: Path, *names: str) -> None:
    """Raise unless each named data file still has the sha256 recorded in data/MANIFEST.json."""
    manifest = json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    for name in names:
        if manifest["files"].get(name) != sha256_file(data_dir / name):
            raise RuntimeError(f"{name} does not match data/MANIFEST.json")
