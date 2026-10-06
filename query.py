"""
LLM/GenAI pipeline, part 2: retrieve -> prompt -> generate.

Run:  python query.py "When was the FrostByte 3000 released?"
"""
import sys
import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "artifacts/index.joblib"
TOP_K = 3


# 5. RETRIEVE ------------------------------------------------------------
def retrieve(query: str, index: dict, k: int = TOP_K) -> list[dict]:
    embedder = SentenceTransformer(index["model_name"])
    q_vec = embedder.encode([query], normalize_embeddings=True)[0]

    # cosine similarity == dot product on normalized vectors
    scores = index["vectors"] @ q_vec
    top = np.argsort(scores)[::-1][:k]
    return [{**index["chunks"][i], "score": float(scores[i])} for i in top]


# 6. PROMPT --------------------------------------------------------------
def build_prompt(query: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    return (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


# 7. GENERATE ------------------------------------------------------------
def generate(prompt: str) -> str:
    """Backend switch: OpenAI if OPENAI_API_KEY is set, else Ollama."""
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return r.choices[0].message.content.strip()

    # Fallback: local Ollama (https://ollama.com) with `ollama pull llama3.2`
    import requests
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


# 8. RUN -----------------------------------------------------------------
def main():
    query = " ".join(sys.argv[1:]) or "When was the FrostByte 3000 released?"
    index = joblib.load(INDEX_PATH)

    hits = retrieve(query, index)
    print("\n--- Retrieved chunks ---")
    for h in hits:
        print(f"[{h['score']:.3f}] {h['source']}: {h['text'][:80]}...")

    prompt = build_prompt(query, hits)
    answer = generate(prompt)
    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()