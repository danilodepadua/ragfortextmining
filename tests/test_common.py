import unicodedata

import pytest

from common import (
    chunk_spans,
    normalize_text,
    read_jsonl,
    stable_chunk_id,
    word_spans,
    words_touched,
    write_jsonl,
)


def make_text(n_words: int) -> str:
    return " ".join(f"w{i}" for i in range(n_words))


def test_normalize_is_nfc_and_keeps_portuguese_characters():
    decomposed = unicodedata.normalize("NFD", "ação nº 5… ok")
    out = normalize_text(decomposed)
    assert out == "ação nº 5… ok"  # NFKC would give "no 5..."
    assert normalize_text("já") == "já"
    assert len("ação nº 5… ok") == len(out)


def test_normalize_handles_empty():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_jsonl_round_trip(tmp_path):
    rows = [{"a": "ç"}, {"b": [1, 2]}]
    write_jsonl(tmp_path / "x" / "f.jsonl", rows)
    assert read_jsonl(tmp_path / "x" / "f.jsonl") == rows


def test_chunk_text_is_the_slice_and_windows_overlap():
    text = make_text(700)
    chunks = chunk_spans(text, 300, 50)
    assert [c["word_count"] for c in chunks] == [300, 300, 200]
    for c in chunks:
        piece = text[c["char_start"] : c["char_end"]]
        assert len(piece.split()) == c["word_count"]
    first, second = chunks[0], chunks[1]
    assert text[second["char_start"] : first["char_end"]].split()[0] == "w250"


def test_chunk_spans_single_chunk_and_invalid_overlap():
    assert len(chunk_spans(make_text(100))) == 1
    assert chunk_spans("") == []
    with pytest.raises(ValueError):
        chunk_spans("a b c", size=5, overlap=5)


def test_any_span_of_50_words_or_fewer_sits_inside_one_chunk():
    text = make_text(1100)
    chunks = chunk_spans(text, 300, 50)
    words = word_spans(text)
    for first in range(0, len(words) - 50):
        start, end = words[first][0], words[first + 49][1]
        assert any(c["char_start"] <= start and end <= c["char_end"] for c in chunks), first


def test_words_touched_counts_partial_words():
    text = "alpha beta gamma"
    assert words_touched(text, 0, 5) == 1
    assert words_touched(text, 3, 8) == 2  # "pha" + space + "b"
    assert words_touched(text, 0, len(text)) == 3


def test_stable_chunk_id_format_and_determinism():
    cid = stable_chunk_id("D001", 3, "texto")
    assert cid.startswith("chk_D001_003_") and len(cid.split("_")[-1]) == 8
    assert cid == stable_chunk_id("D001", 3, "texto")
    assert cid != stable_chunk_id("D001", 3, "outro")
