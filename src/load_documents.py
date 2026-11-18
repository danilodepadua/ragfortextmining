import hashlib
import json
import os
import re
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize OpenAI client (automatically picks up OPENAI_API_KEY from the environment)
client = OpenAI()

def is_mostly_code_or_tables(text: str) -> bool:
    """Heuristic check to quickly drop documents dominated by code blocks or markdown/ASCII tables."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    table_lines = sum(1 for line in lines if line.startswith("|") or line.count("|") >= 3 or set(line) <= set("-+|="))
    code_indicators = len(re.findall(r"```|def |class |import |function\s*\(|const |var |let |;\s*$|{\s*$|^\s*}", text, re.MULTILINE))
    
    if (table_lines / len(lines)) > 0.30:
        return True
    if code_indicators > (len(lines) * 0.25):
        return True
    return False

def is_broken_or_empty(text: str) -> bool:
    """Checks for empty text, encoding corruption, or excessive non-alphanumeric noise."""
    if not text or not text.strip():
        return True
    if "\ufffd" in text:
        return True
    
    valid_chars = sum(1 for c in text if c.isalnum() or c.isspace())
    if (valid_chars / len(text)) < 0.60:
        return True
    return False

def passes_llm_criteria(question: str, context_text: str, response_text: str) -> dict | None:
    """Uses a fast LLM to verify technical style, language, exclude news/opinion, and verify Q&A match."""
    prompt = f"""
Analyze the following text sample from a RAG dataset and determine if it meets ALL of these strict criteria:
1. It is written in English.
2. It is technical in nature (e.g., documentation, tutorial, how-to guide, technical explanation).
3. It is NOT news, journalism, personal opinion, or marketing/promotional material.
4. The provided Question and Response/Answer accurately match, are supported by the Context text, and the Response provides a substantive, complete answer (not empty, a refusal, or missing an answer).

Question: {question}

Context Text:
\"\"\"{context_text[:5000]}\"\"\"

Response: {response_text}

Respond ONLY with a valid JSON object in this exact format:
{{
  "is_english": true/false,
  "is_technical": true/false,
  "is_news_opinion_or_marketing": true/false,
  "has_matching_qa": true/false
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(response.choices[0].message.content)
        if (
            result.get("is_english") is True
            and result.get("is_technical") is True
            and result.get("is_news_opinion_or_marketing") is False
            and result.get("has_matching_qa") is True
        ):
            return result
    except Exception as e:
        print(f"LLM evaluation error: {e}")
    return None

def stream_and_filter(target_matches: int = 26, output_file: str = "data/curated_documents/filtered_ragbench_tech.jsonl"):
    # Ensure the directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load the 'techqa' config of ragbench dataset in streaming mode and shuffle deterministically
    ds = load_dataset("galileo-ai/ragbench", "techqa", split="train", streaming=True)
    ds = ds.shuffle(seed=42)
    
    seen_hashes = set()
    seen_questions = set()  # Track unique questions to prevent duplicate entries
    processed_count = 0
    saved_count = 0

    print(f"Starting stream. Looking for {target_matches} matching documents...")

    with open(output_file, "w", encoding="utf-8") as f:
        dataset_iterator = iter(ds)
        
        while saved_count < target_matches:
            try:
                row = next(dataset_iterator)
            except StopIteration:
                print("\nReached the end of the dataset stream before hitting target matches.")
                break

            processed_count += 1
            print(f"\n--- Evaluated Total: {processed_count} | Saved: {saved_count}/{target_matches} ---")

            question = row.get("question", "")
            documents = row.get("documents", [])
            context_text = " ".join(documents) if isinstance(documents, list) else str(documents)
            response_text = row.get("response", "")

            # 1. Response Presence & Non-Empty Check
            if not response_text or not response_text.strip():
                print(" -> Skipped: Empty or missing response")
                continue

            # 2. Question Uniqueness Check
            norm_question = question.strip().lower()
            if not norm_question or norm_question in seen_questions:
                print(" -> Skipped: Empty or duplicate question")
                continue

            # 3. Word Count Filter (100 - 1500 words on context text)
            words = context_text.split()
            word_count = len(words)
            if not (100 <= word_count <= 1500):
                print(f" -> Skipped: Word count ({word_count}) out of bounds (100-1500)")
                continue

            # 4. Broken / Empty Filter
            if is_broken_or_empty(context_text):
                print(" -> Skipped: Broken or empty text")
                continue

            # 5. Document Duplicate Filter (Normalized MD5 Hash)
            norm_hash = hashlib.md5(context_text.strip().lower().encode("utf-8")).hexdigest()
            if norm_hash in seen_hashes:
                print(" -> Skipped: Duplicate document context")
                continue

            # 6. Code & Tables Heuristic Filter
            if is_mostly_code_or_tables(context_text):
                print(" -> Skipped: Mostly code or tables")
                continue

            # 7. Quick Pre-Filter for Question Marks
            if "?" not in question and "?" not in context_text:
                print(" -> Skipped: No question mark found")
                continue

            # 8. Semantic LLM Filter (English, Technical, No News/Marketing, Matching Q&A)
            llm_verdict = passes_llm_criteria(question, context_text, response_text)
            if llm_verdict:
                # Mark as seen only upon successful save
                seen_hashes.add(norm_hash)
                seen_questions.add(norm_question)

                record = {
                    "question": question,
                    "context": context_text,
                    "response": response_text,
                    "word_count": word_count,
                    "dataset_name": row.get("dataset_name"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved_count += 1
                print(f" -> MATCH SAVED! Total saved so far: {saved_count}/{target_matches}")
            else:
                print(" -> Skipped: LLM criteria not met")

    print(f"\n=== Finished! Inspected {processed_count} documents to successfully collect {saved_count} matches into {output_file} ===")

if __name__ == "__main__":
    stream_and_filter(target_matches=26)