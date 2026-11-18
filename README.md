# Configuring the Environment

## Step 1: Create a virtual environment
- Open your terminal inside your project directory and run the following command:
```python -m venv venv```

## Step 2: Activate the virtual environment`
- On macOS and Linux:
```source venv/bin/activate```

- On Windows (Command Prompt or PowerShell):
```venv\Scripts\activate```

## Step 3: Install the dependencies
- Run the following command to install the required libraries into your virtual environment:
```pip install -r requirements.txt```

## Step 4: Configure your environment variables
- Create a file named `.env` in your root directory and add your OpenAI API key so the scripts can authenticate properly:
```OPENAI_API_KEY=your_actual_openai_api_key_here```


# Protocol of document curating

## 1. Which documents we use

- Dataset: rungalileo/ragbench, subset "techqa" (English)
- We use 25 documents, picked randomly (seed 42)
- Saved in `data/corpus_v1.csv` (IDs D001-D025). It does not change after M1.

**Include a document if it:**
- is in English
- is technical (documentation, how-to)
- is short (100-1,500 words)
- has a question and answer that match the text

**Exclude a document if it:**
- is a duplicate
- is mostly code or tables
- is empty or broken
- is opinion, news or marketing
- is not in English
- does not have an answer for its question
- has the same question as a previous document

## 2. How we measure

**Retrieval:** compare TF-IDF and dense search (same chunks, same k).
Score = evidence recall@k, for k = 1, 3, 5.

**Answers:** grade 10 answers by hand.
- Correct / Partial / Wrong
- Does the cited chunk support the answer? Yes / No

**Paired questions:** 5 questions, answered with both retrievers.
Chosen randomly (seed 42). 3 of them are highlighted.

**Logs:** save question, retriever, chunk IDs, prompt, model settings and answer for every run.

**Failures:** analyze 2 wrong answers. Name 1 search risk and 1 fix.


# Loading new documents
- Open your terminal inside the project directory and run
```python load_documents.py```