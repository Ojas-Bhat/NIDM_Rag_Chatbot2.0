# datapop3.py
import os
import re
import json
import pickle
import hashlib
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pdfplumber
import nltk
import pytesseract
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient, ASCENDING, errors
from rank_bm25 import BM25Okapi

try:
    import faiss
except ImportError:
    faiss = None  # we'll warn later if needed

# ---------- CONFIG (adjust if needed) ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "disaster_hub"
COLL_NAME = "documents"

# Model: all-MiniLM-L6-v2 (384-d)
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

# Index location (relative to THIS script, not the current directory)
SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_DIR = SCRIPT_DIR / "indices"
FAISS_PATH = INDEX_DIR / "faiss_index.bin"
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHUNKS_PATH = INDEX_DIR / "chunks_meta.pkl"

warnings.filterwarnings("ignore", message="Cannot set .* color")

# ---------- DB SETUP ----------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLL_NAME]

try:
    collection.create_index(
        [("doc_id", ASCENDING), ("text_hash", ASCENDING)],
        unique=True,
        name="uniq_doc_chunk"
    )
except errors.PyMongoError:
    pass

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ---------- OCR CACHE HELPERS ----------
def _cache_root() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_cache")

def _cache_dir_for(pdf_path: str) -> str:
    h = hashlib.sha1(pdf_path.encode("utf-8")).hexdigest()
    d = os.path.join(_cache_root(), h)
    os.makedirs(d, exist_ok=True)
    return d

def _cache_key(pdf_path: str, page_index: int) -> str:
    try:
        mtime = os.path.getmtime(pdf_path)
    except Exception:
        mtime = 0
    return f"p{page_index}_m{int(mtime)}.json"

def _load_ocr_cache(pdf_path: str, page_index: int) -> str:
    d = _cache_dir_for(pdf_path)
    p = os.path.join(d, _cache_key(pdf_path, page_index))
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f).get("text", "")
        except Exception:
            return ""
    return ""

def _save_ocr_cache(pdf_path: str, page_index: int, text: str):
    d = _cache_dir_for(pdf_path)
    p = os.path.join(d, _cache_key(pdf_path, page_index))
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"text": text, "ts": datetime.utcnow().isoformat()}, f)
    except Exception:
        pass


# ---------- TEXT EXTRACTION (Selective OCR) ----------
def extract_text_from_pdf(
    pdf_path: str,
    *,
    enable_ocr: bool = True,
    ocr_dpi: int = 300,
    min_text_chars: int = 40,
    ocr_lang: str = "eng",
) -> str:
    full_text, ocr_count, pages = [], 0, 0

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            pages += 1
            page_text = page.extract_text() or ""
            if len(page_text.strip()) >= min_text_chars:
                full_text.append(page_text.strip())
                continue
            if not enable_ocr:
                if page_text:
                    full_text.append(page_text.strip())
                continue
            cached = _load_ocr_cache(pdf_path, i)
            if cached:
                full_text.append(cached.strip())
                ocr_count += 1
                continue
            try:
                pil_image = page.to_image(resolution=ocr_dpi).original
                config = r"--oem 1 --psm 3"
                ocr_text = pytesseract.image_to_string(pil_image, lang=ocr_lang, config=config) or ""
                ocr_text = ocr_text.strip()
                _save_ocr_cache(pdf_path, i, ocr_text)
                full_text.append(ocr_text)
                ocr_count += 1
            except Exception:
                full_text.append(page_text.strip())

    print(f"→ Selective OCR summary: {ocr_count}/{pages} page(s) OCR’d for {os.path.basename(pdf_path)}")
    return "\n".join(t for t in full_text if t)


# ---------- CHUNKING ----------
def clean_and_chunk_text(text: str, max_chunk_size: int = 500, min_chunk_len: int = 80):
    sentences = [s.strip() for s in sent_tokenize(text) if s and s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + 1 + len(sentence) <= max_chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if len(current) >= min_chunk_len:
                chunks.append(current)
            current = sentence
    if current and len(current) >= min_chunk_len:
        chunks.append(current)
    return chunks


# ---------- METADATA (Scoring-based) ----------
def _disaster_patterns():
    return {
        "Cyclone": [
            r"\bcyclone(s)?\b", r"\bcyclonic\b",
            r"\bvery severe cyclonic storm\b", r"\bvs?cs\b",
            r"\bhurricane(s)?\b", r"\btyphoon(s)?\b",
            r"\btropical cyclone(s)?\b", r"\btropical storm(s)?\b"
        ],
        "Flood": [
            r"\bflood(s|ing)?\b", r"\binundation(s)?\b",
            r"\briverine flood(s)?\b", r"\bflash flood(s)?\b"
        ],
        "Earthquake": [
            r"\bearthquake(s)?\b", r"\bseismic\b",
            r"\bmagnitude\s*\d+(\.\d+)?\b", r"\bepicenter\b"
        ],
        "Landslide": [
            r"\blandslide(s)?\b", r"\bdebris flow(s)?\b", r"\bflow failure(s)?\b"
        ],
        "Drought": [
            r"\bdrought(s)?\b", r"\bwater scarcity\b", r"\bmeteorological drought\b"
        ],
        "Tsunami": [r"\btsunami(s)?\b"],
        "Fire": [
            r"\b(wild)?fire(s)?\b", r"\bforest fire(s)?\b", r"\bwildland fire(s)?\b"
        ],
        "Heat Wave": [
            r"\bheat ?wave(s)?\b", r"\bheatwave(s)?\b",
            r"\bmaximum temperature(s)?\b", r"\bwet bulb\b"
        ],
        "Storm": [r"\bstorm(s)?\b", r"\bgale(s)?\b", r"\bwindstorm(s)?\b"],
        "Pandemic": [r"\bpandemic(s)?\b", r"\bcovid-?19\b", r"\bsars-cov-2\b"],
    }

def _count_matches(text: str, patterns: list) -> int:
    t = text.lower()
    return sum(len(re.findall(p, t, flags=re.IGNORECASE)) for p in patterns)

def _score_disaster(text: str, file_name: str) -> dict:
    pats = _disaster_patterns()
    scores = {k: 0 for k in pats.keys()}
    for k, plist in pats.items():
        scores[k] += _count_matches(text, plist)
    for k, plist in pats.items():
        boost = _count_matches(file_name, plist)
        if boost:
            scores[k] += boost * 5
    return scores

def _pick_disaster(scores: dict) -> str:
    if all(v == 0 for v in scores.values()):
        return "Unknown"
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0

    if top == "Storm":
        for specific in ["Cyclone", "Flood", "Heat Wave", "Fire", "Landslide",
                         "Earthquake", "Tsunami", "Pandemic"]:
            if scores.get(specific, 0) >= (0.8 * top_score) and scores.get(specific, 0) > 0:
                return specific

    specific_order = ["Cyclone", "Flood", "Heat Wave", "Fire", "Landslide",
                      "Earthquake", "Tsunami", "Pandemic", "Storm"]
    if second_score > 0 and top_score < 1.2 * second_score:
        t1 = specific_order.index(top) if top in specific_order else len(specific_order)
        second = next((k for k, v in scores.items() if v == second_score and k != top), None)
        if second and second in specific_order:
            t2 = specific_order.index(second)
            if t2 < t1:
                return second
    return top

def extract_metadata(text: str, file_name: str) -> dict:
    metadata = {"source": file_name}
    year_match = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", text)
    metadata["year"] = int(year_match.group(0)) if year_match else None
    scores = _score_disaster(text, file_name)
    metadata["disaster"] = _pick_disaster(scores)
    return metadata


# ---------- STORAGE ----------
def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def store_document(filename, chunks, metadata):
    """Store document chunks into MongoDB with embeddings."""
    metadata["uploaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc_id = os.path.splitext(filename)[0]

    print(f"→ Generating embeddings for {len(chunks)} chunks ...")
    embeddings = EMBEDDER.encode(chunks, show_progress_bar=False).tolist()

    batch = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        batch.append({
            "doc_id": doc_id,
            "text_hash": _hash_text(chunk),
            "filename": filename,
            "chunk_index": i,
            "text": chunk,
            "embedding": emb,
            "metadata": metadata
        })

    if batch:
        try:
            collection.insert_many(batch, ordered=False)
            print(f"✅ Stored {len(batch)} chunks for {filename}")
        except Exception as e:
            print(f"⚠️ MongoDB insert warning: {e}")
    else:
        print(f"⚠️ No chunks to store for {filename}")


# ============================================================
# NEW: FAISS + BM25 INDEX BUILDING
# ============================================================
def build_indices(verbose: bool = True) -> bool:
    """
    Read every chunk from MongoDB, build a FAISS (semantic) index and a
    BM25 (keyword) index, and save them to disk under indices/.

    Run this after ingesting new PDFs, and whenever you want to refresh
    the search index.
    """
    if faiss is None:
        print("✗ faiss is not installed. Run: pip install faiss-cpu")
        return False

    docs = list(collection.find(
        {},
        {"_id": 0, "text": 1, "embedding": 1, "metadata": 1,
         "filename": 1, "doc_id": 1, "chunk_index": 1}
    ))

    # Drop any rows missing data (e.g. a failed insert)
    clean, skipped = [], 0
    for d in docs:
        if d.get("text") and d.get("embedding"):
            clean.append(d)
        else:
            skipped += 1

    if not clean:
        print("⚠️ No usable chunks found in MongoDB. Ingest documents first.")
        return False

    texts = [d["text"] for d in clean]
    metas = [{
        "filename": d.get("filename", d.get("metadata", {}).get("source", "Unknown")),
        "doc_id": d.get("doc_id", ""),
        "chunk_index": d.get("chunk_index", -1),
        "disaster": d.get("metadata", {}).get("disaster", "Unknown"),
        "year": d.get("metadata", {}).get("year"),
        "uploaded_at": d.get("metadata", {}).get("uploaded_at"),
    } for d in clean]

    # ---- FAISS ----
    embeddings = np.array([d["embedding"] for d in clean], dtype="float32")
    if embeddings.ndim != 2 or embeddings.shape[1] != 384:
        print(f"⚠️ Unexpected embedding shape {embeddings.shape}; expected (n, 384).")
        return False

    faiss.normalize_L2(embeddings)   # so inner product == cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))

    # ---- BM25 ----
    bm25 = BM25Okapi([t.lower().split() for t in texts])
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    # ---- Texts + metadata ----
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump({"texts": texts, "metas": metas}, f)

    if verbose:
        print(f"✅ Built indices for {len(texts)} chunks "
              f"({skipped} skipped) → {INDEX_DIR}")
    return True


def test_index(query: str, top_k: int = 3) -> None:
    """Quick sanity check: search the FAISS index and print top results."""
    if faiss is None:
        print("✗ faiss is not installed. Run: pip install faiss-cpu")
        return
    if not FAISS_PATH.exists():
        print("✗ No index found. Run: python datapop3.py --build-indices")
        return

    index = faiss.read_index(str(FAISS_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        data = pickle.load(f)
    texts, metas = data["texts"], data["metas"]

    q = EMBEDDER.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, top_k)

    print(f"\n🔎 Test query: {query!r}")
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        if idx < 0:
            continue
        m = metas[idx]
        print(f"  [{rank}] score={score:.3f} | {m['filename']} "
              f"| {m['disaster']} {m['year']} | chunk {m['chunk_index']}")
        print(f"       {texts[idx][:160]}...")
    print()


# ---------- PIPELINE ----------
def process_pdf(pdf_path, *, enable_ocr=True, ocr_dpi=300,
                min_text_chars=40, ocr_lang="eng"):
    if not os.path.exists(pdf_path):
        print(f"✗ Skipped (file not found): {pdf_path}")
        return

    file_name = os.path.basename(pdf_path)
    print(f"→ Extracting: {pdf_path}")
    raw_text = extract_text_from_pdf(
        pdf_path, enable_ocr=enable_ocr, ocr_dpi=ocr_dpi,
        min_text_chars=min_text_chars, ocr_lang=ocr_lang,
    )
    if not raw_text or not raw_text.strip():
        print("  ✗ No extractable text.")
        return

    print("→ Chunking text ...")
    chunks = clean_and_chunk_text(raw_text, max_chunk_size=500, min_chunk_len=80)
    if not chunks:
        print("  ✗ No valid chunks.")
        return

    print("→ Building metadata ...")
    metadata = extract_metadata(raw_text, file_name)
    print(f"  • metadata: {metadata}")

    print("→ Storing in MongoDB ...")
    store_document(filename=file_name, chunks=chunks, metadata=metadata)
    print(f"  ✓ Stored {len(chunks)} chunks for {file_name}\n")


def process_folder(folder_path, *, enable_ocr=True, ocr_dpi=300,
                   min_text_chars=40, ocr_lang="eng"):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    pdfs = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {folder_path}")
    print(f"Found {len(pdfs)} PDF(s) in {folder_path}")
    for fname in pdfs:
        process_pdf(os.path.join(folder_path, fname), enable_ocr=enable_ocr,
                    ocr_dpi=ocr_dpi, min_text_chars=min_text_chars, ocr_lang=ocr_lang)


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Ingest disaster PDFs into MongoDB and build FAISS/BM25 search indices."
    )
    parser.add_argument("--file", "-f", help="Path to a single PDF to ingest.")
    parser.add_argument("--folder", "-d", help="Folder containing PDFs to ingest.")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR.")
    parser.add_argument("--ocr-lang", default="eng", help='OCR language(s), e.g. "eng+hin".')
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--min-text-chars", type=int, default=40)

    parser.add_argument("--build-indices", action="store_true",
                        help="Build FAISS + BM25 indices from MongoDB and exit.")
    parser.add_argument("--test-index", metavar="QUERY",
                        help="Run a test search against the saved index and exit.")

    args = parser.parse_args()

    # --- index-only modes ---
    if args.build_indices:
        build_indices(verbose=True)
        return
    if args.test_index:
        test_index(args.test_index)
        return

    # --- ingestion modes ---
    if args.file:
        process_pdf(args.file, enable_ocr=not args.no_ocr, ocr_dpi=args.ocr_dpi,
                    min_text_chars=args.min_text_chars, ocr_lang=args.ocr_lang)
    elif args.folder:
        process_folder(args.folder, enable_ocr=not args.no_ocr, ocr_dpi=args.ocr_dpi,
                       min_text_chars=args.min_text_chars, ocr_lang=args.ocr_lang)
    else:
        parser.print_help()
        print("\nTip: after ingesting PDFs, run:")
        print("  python datapop3.py --build-indices")
        print('  python datapop3.py --test-index "Chennai flood causes"')


if __name__ == "__main__":
    main()