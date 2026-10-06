# tiny-llm-pipeline

A minimal RAG pipeline: ingest → chunk → embed → retrieve → prompt → generate.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Build the index (embeds docs/)
python pipeline.py

# 2. Ask a question
python query.py "When was the FrostByte 3000 released?"

# Or use OpenAI instead of local Ollama
export OPENAI_API_KEY=sk-...
python query.py "Where is Acme Corp based?"
```

## Stages

| # | Stage     | File         | Notes                                |
|---|-----------|--------------|--------------------------------------|
| 1 | Ingest    | `pipeline.py`| Load `.txt` — swap for S3/DB/crawler |
| 2 | Chunk     | `pipeline.py`| 200 chars, 40 overlap                |
| 3 | Embed     | `pipeline.py`| `all-MiniLM-L6-v2` (local, free)     |
| 4 | Store     | `pipeline.py`| joblib file — swap for FAISS/pgvector|
| 5 | Retrieve  | `query.py`   | Cosine top-k                         |
| 6 | Prompt    | `query.py`   | Grounded prompt w/ "I don't know"    |
| 7 | Generate  | `query.py`   | Ollama or OpenAI                     |
| 8 | Evaluate  | (next step)  | Faithfulness / answer relevance      |

## Why this shape?

- **Grounding** — the LLM sees retrieved text, so answers come from *your* docs.
- **Backend-agnostic** — swap the embedder or the LLM without touching the pipeline.
- **Inspectable** — retrieve and generate are separate functions you can log/score.