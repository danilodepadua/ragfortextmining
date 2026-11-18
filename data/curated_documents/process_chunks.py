import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    """Normalizes text by fixing unicode characters, standardizing whitespace,

    and preserving paragraph breaks.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_stable_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
    """Generates a deterministic stable ID for a chunk using its doc_id,

    index, and content hash.
    """
    hash_obj = hashlib.sha256(chunk_text.encode("utf-8"))
    return f"chk_{doc_id}_{chunk_index:03d}_{hash_obj.hexdigest()[:8]}"


def chunk_text_with_overlap(
    text: str, chunk_size: int = 300, overlap: int = 50
) -> list[dict]:
    """Splits normalized text into overlapping word-based chunks safely."""
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly smaller than chunk_size.")

    words = text.split(" ")
    total_words = len(words)
    chunks = []

    if total_words == 0:
        return chunks

    step = chunk_size - overlap
    for chunk_idx, start in enumerate(range(0, total_words, step)):
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        chunks.append({
            "chunk_index": chunk_idx,
            "text": chunk_str,
            "start_word": start,
            "end_word": end,
            "word_count": len(chunk_words),
        })

        if end == total_words:
            break

    return chunks


def process_jsonl_corpus(
    input_jsonl_path: str,
    output_jsonl_path: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> None:
    """Reads filtered_ragbench_tech.jsonl line by line, assigns D001-D025 IDs,

    normalizes text, and writes overlapping chunks out to a new JSONL.
    """
    input_file = Path(input_jsonl_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")

    total_chunks = 0
    total_docs = 0

    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_jsonl_path, "w", encoding="utf-8") as outfile,
    ):
        for idx, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue

            # Assign sequential protocol IDs: D001, D002, ..., up to D025
            doc_id = f"D{idx + 1:03d}"

            record = json.loads(line)

            # 1. Normalize fields
            norm_context = normalize_text(record.get("context", ""))
            norm_question = normalize_text(record.get("question", ""))
            norm_response = normalize_text(record.get("response", ""))
            dataset_name = record.get("dataset_name", "techqa_train")

            # 2. Create overlapping chunks
            chunks = chunk_text_with_overlap(
                norm_context, chunk_size=chunk_size, overlap=overlap
            )

            # 3. Assign stable IDs and write records
            for chunk in chunks:
                chunk_id = generate_stable_chunk_id(
                    doc_id, chunk["chunk_index"], chunk["text"]
                )

                output_record = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "metadata": {
                        "dataset_name": dataset_name,
                        "question": norm_question,
                        "ground_truth_response": norm_response,
                        "original_word_count": record.get("word_count"),
                        "chunk_word_count": chunk["word_count"],
                        "word_span": [chunk["start_word"], chunk["end_word"]],
                    },
                }
                outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                total_chunks += 1

            total_docs += 1

    print(
        f"Successfully processed {total_docs} documents (D001-D{total_docs:03d}) into {total_chunks} chunks."
    )
    print(f"Saved output to: {output_jsonl_path}")


if __name__ == "__main__":
    INPUT_JSONL = "filtered_ragbench_tech.jsonl"
    OUTPUT_JSONL = "chunked_filtered_ragbench_tech.jsonl"

    # Chunking configurations (300 words with 50-word overlap)
    CHUNK_SIZE_WORDS = 300
    OVERLAP_WORDS = 50

    process_jsonl_corpus(
        input_jsonl_path=INPUT_JSONL,
        output_jsonl_path=OUTPUT_JSONL,
        chunk_size=CHUNK_SIZE_WORDS,
        overlap=OVERLAP_WORDS,
    )