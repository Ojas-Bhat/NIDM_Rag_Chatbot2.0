# cb.py
import asyncio
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

# --- Providers ---
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "groq")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "phi3:mini")

try:
    from groq import AsyncGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from ollama import AsyncClient
    HAS_OLLAMA_PY = True
except ImportError:
    HAS_OLLAMA_PY = False
    import subprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Paths ---
SCRIPT_DIR   = Path(__file__).resolve().parent
INDEX_DIR    = SCRIPT_DIR / "indices"
FAISS_PATH   = INDEX_DIR / "faiss_index.bin"
BM25_PATH    = INDEX_DIR / "bm25.pkl"
CHUNKS_PATH  = INDEX_DIR / "chunks_meta.pkl"

# --- Models ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
# Cross-encoder reranker for higher accuracy
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# --- Hybrid Retrieval + Reranking ---
def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    use_reranker: bool = True,
) -> List[Dict[str, Any]]:
    """
    1. Retrieve candidates via FAISS (dense) + BM25 (sparse)
    2. Rerank top candidates using CrossEncoder
    """
    for p in (FAISS_PATH, BM25_PATH, CHUNKS_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"Index file missing: {p}\n"
                "Run: python datapop3.py --build-indices"
            )

    index = faiss.read_index(str(FAISS_PATH))
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(CHUNKS_PATH, "rb") as f:
        data = pickle.load(f)

    texts   = data["texts"]
    metas   = data["metas"]
    n_docs  = len(texts)

    # 1) Vector search via FAISS
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    candidate_k = min(top_k * 4, n_docs)  # Retrieve 20 candidates for reranking
    D, I = index.search(q_emb, candidate_k)

    vector_scores = {}
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < n_docs:
            vector_scores[idx] = float(score)

    # 2) BM25 keyword search
    tokenized_q = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_q)

    # 3) Hybrid Combination
    combined = {}
    for i in range(n_docs):
        v = vector_scores.get(i, 0.0)
        b = float(bm25_scores[i]) if i < len(bm25_scores) else 0.0
        b_norm = b / (1.0 + b)
        combined[i] = vector_weight * v + bm25_weight * b_norm

    # Top candidates for reranking
    candidate_indices = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:candidate_k]

    # 4) Cross-Encoder Reranking
    if use_reranker and reranker and candidate_indices:
        pairs = [(query, texts[idx]) for idx in candidate_indices]
        rerank_scores = reranker.predict(pairs)

        # Sort candidate indices by cross-encoder score
        reranked_order = np.argsort(rerank_scores)[::-1][:top_k]
        final_indices = [candidate_indices[i] for i in reranked_order]
        scores_to_use = [float(rerank_scores[i]) for i in reranked_order]
    else:
        final_indices = candidate_indices[:top_k]
        scores_to_use = [combined[i] for i in final_indices]

    results: List[Dict[str, Any]] = []
    for idx, score in zip(final_indices, scores_to_use):
        results.append({
            "text":       texts[idx],
            "metadata":   metas[idx],
            "similarity": float(score),
        })

    return results


# --- LLM Providers ---
async def query_groq_async(prompt: str, model: str = GROQ_MODEL) -> str:
    if not HAS_GROQ:
        raise RuntimeError("groq package not installed. Run: pip install groq")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")

    client = AsyncGroq(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content


async def query_ollama_async(prompt: str, model: str = OLLAMA_MODEL) -> str:
    if HAS_OLLAMA_PY:
        client = AsyncClient(host="http://localhost:11434")
        response = await client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response["message"]["content"]
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True,
    )
    return result.stdout.decode("utf-8")


async def query_llm_async(prompt: str) -> str:
    if LLM_PROVIDER == "groq":
        return await query_groq_async(prompt)
    if LLM_PROVIDER == "ollama":
        return await query_ollama_async(prompt)
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


# --- Generate Answer ---
async def generate_answer_async(
    user_query: str,
    retrieved: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not retrieved:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "context": "",
        }

    context_parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    for idx, item in enumerate(retrieved, start=1):
        m = item["metadata"]
        label = f"[{idx}]"

        source_info = {
            "label": label,
            "document": m.get("filename", m.get("source", "Unknown")),
            "disaster": m.get("disaster", "Unknown"),
            "year": m.get("year", "Unknown"),
            "similarity": item["similarity"],
            "snippet": item["text"][:300] + ("..." if len(item["text"]) > 300 else ""),
        }
        sources.append(source_info)

        context_parts.append(
            f"{label} [{source_info['disaster']} {source_info['year']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are a disaster knowledge assistant.
Answer the question based ONLY on the context below.
If the answer is not in the context, say so clearly.
Cite your sources using bracket numbers like [1], [2] that appear before each source text.

Context:
{context}

Question: {user_query}

Answer:"""

    try:
        answer = await query_llm_async(prompt)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        answer = f"Error generating answer: {exc}"

    return {
        "answer": answer.strip(),
        "sources": sources,
        "context": context,
    }


def generate_answer(
    user_query: str,
    retrieved: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return asyncio.run(generate_answer_async(user_query, retrieved))


if __name__ == "__main__":
    user_query = "what caused the Chennai flood"

    try:
        retrieved = retrieve_relevant_chunks(user_query, top_k=5)
    except FileNotFoundError as exc:
        print(exc)
        retrieved = []

    if not retrieved:
        print("No relevant documents found.")
    else:
        result = generate_answer(user_query, retrieved)
        print("\n=== ANSWER ===\n")
        print(result["answer"])
        print("\n=== SOURCES ===\n")
        for s in result["sources"]:
            print(f"{s['label']} {s['document']} | {s['disaster']} {s['year']} | sim={s['similarity']:.3f}")
            print(f"   {s['snippet']}\n")