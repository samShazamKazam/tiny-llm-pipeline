"""
LLM/GenAI pipeline, part 1: build a retrieval index.

Stages: ingest -> chunk -> embed -> store

Run:  python pipeline.py
"""

from pathlib import Path
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path("docs")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


# 1. INGEST --------------------------------------------------------------
def ingest(docs_dir: Path) -> list[dict]:
    """Load raw documents. Swap for S3, a DB, a web crawler, etc."""
    docs = []
    for path in sorted(docs_dir.glob("*.txt")):
        docs.append({"source": path.name, "text": path.read_text()})
    return docs


# 2. CHUNK ---------------------------------------------------------------
def chunk(doc: dict, size: int = 200, overlap: int = 40) -> list[dict]:
    """Split a doc into overlapping character chunks."""
    text, chunks, start = doc["text"], [], 0
    while start < len(text):
        end = start + size
        chunks.append({
            "source": doc["source"],
            "text": text[start:end],
            "start": start,
        })
        start += size - overlap
    return chunks


# 3. EMBED + 4. STORE ----------------------------------------------------
def build_index(chunks: list[dict], model_name: str = "all-MiniLM-L6-v2"):
    print(f"Loading embedder: {model_name}")
    embedder = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    vectors = embedder.encode(texts, normalize_embeddings=True)

    index = {
        "chunks": chunks,
        "vectors": np.asarray(vectors, dtype=np.float32),
        "model_name": model_name,
    }
    joblib.dump(index, ARTIFACT_DIR / "index.joblib")
    print(f"Saved index -> {ARTIFACT_DIR / 'index.joblib'}")


def main():
    print("1) Ingesting docs...")
    docs = ingest(DOCS_DIR)
    print(f"   {len(docs)} documents")

    print("2) Chunking...")
    chunks = [c for d in docs for c in chunk(d)]
    print(f"   {len(chunks)} chunks")

    print("3) Embedding + storing...")
    build_index(chunks)


if __name__ == "__main__":
    main()