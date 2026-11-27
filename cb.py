import asyncio
import logging
import re
from typing import Any, Dict, List

import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# --- Optional async Ollama client ---
try:
    from ollama import AsyncClient
    HAS_OLLAMA_PY = True
except ImportError:
    HAS_OLLAMA_PY = False
    import subprocess

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- MongoDB Setup ---
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["disaster_hub"]
collection = db["documents"]

# --- Embedding Model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Query Filter Extraction ---
def extract_filters_from_query(query: str) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}

    # Year detection (e.g., "2018 floods in Kerala")
    year_match = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", query)
    if year_match:
        filters["metadata.year"] = int(year_match.group(0))

    # Disaster type detection (simple keyword matching)
    disaster_types = [
        "Flood", "Earthquake", "Cyclone", "Landslide",
        "Drought", "Tsunami", "Pandemic", "Storm", "Fire"
    ]
    for dtype in disaster_types:
        if dtype.lower() in query.lower():
            filters["metadata.disaster"] = dtype
            break

    return filters

# --- Retrieve Relevant Chunks ---
def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query_embedding = embedder.encode([query])[0]

    filters = extract_filters_from_query(query)
    mongo_query = filters if filters else {}

    docs = list(
        collection.find(
            mongo_query,
            {"text": 1, "embedding": 1, "metadata": 1, "_id": 0}
        )
    )

    if not docs:
        return []

    embeddings = np.array([doc["embedding"] for doc in docs])
    doc_texts = [doc["text"] for doc in docs]
    doc_metadata = [doc.get("metadata", {}) for doc in docs]

    # Cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = np.dot(doc_norms, query_norm)

    top_indices = np.argsort(sims)[::-1][:top_k]
    results: List[Dict[str, Any]] = []
    for i in top_indices:
        results.append({
            "text": doc_texts[i],
            "metadata": doc_metadata[i],
            "similarity": float(sims[i]),
        })
    return results

# --- Async Ollama Generation ---
async def query_ollama_async(model: str, prompt: str) -> str:
    if HAS_OLLAMA_PY:
        ollama_client = AsyncClient(host="http://localhost:11434")
        response = await ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response["message"]["content"]
    else:
        # Fallback to subprocess (blocking, but works without the package)
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
        )
        return result.stdout.decode("utf-8")

async def generate_answer_async(
    user_query: str,
    retrieved: List[Dict[str, Any]],
    model: str = "llama3.2:1b",
) -> Dict[str, Any]:
    if not retrieved:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "context": "",
        }

    # Build context with numbered citations
    context_parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    for idx, item in enumerate(retrieved, start=1):
        m = item["metadata"]
        citation_label = f"[{idx}]"

        source_info = {
            "label": citation_label,
            "document": m.get("filename", m.get("source", "Unknown")),
            "disaster": m.get("disaster", "Unknown"),
            "year": m.get("year", "Unknown"),
            "similarity": item["similarity"],
            "snippet": item["text"][:300] + ("..." if len(item["text"]) > 300 else ""),
        }
        sources.append(source_info)

        context_parts.append(
            f"{citation_label} [{source_info['disaster']} {source_info['year']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are a disaster knowledge assistant.
Answer the question based ONLY on the context below.
If the answer is not in the context, say so clearly.
Cite your sources using the bracket numbers like [1], [2] that appear before each source text.

Context:
{context}

Question: {user_query}

Answer:"""

    try:
        answer = await query_ollama_async(model, prompt)
    except Exception as exc:
        logger.error("Ollama generation failed: %s", exc)
        answer = f"Error generating answer: {exc}"

    return {
        "answer": answer.strip(),
        "sources": sources,
        "context": context,
    }

# Synchronous wrapper (useful for quick scripts)
def generate_answer(
    user_query: str,
    retrieved: List[Dict[str, Any]],
    model: str = "llama3",
) -> Dict[str, Any]:
    return asyncio.run(generate_answer_async(user_query, retrieved, model))

# --- Example Usage ---
if __name__ == "__main__":
    user_query = "Tell me about the Chennai Flood"

    retrieved = retrieve_relevant_chunks(user_query, top_k=5)
    if not retrieved:
        print("No relevant documents found.")
    else:
        result = generate_answer(user_query, retrieved, model="llama3")
        print("\n=== ANSWER ===\n")
        print(result["answer"])
        print("\n=== SOURCES ===\n")
        for s in result["sources"]:
            print(
                f"{s['label']} {s['document']} | {s['disaster']} {s['year']} | sim={s['similarity']:.3f}"
            )
            print(f"   {s['snippet']}")
            print()