import os, sys, json, argparse, re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

def chunk_text(text, chunk_chars=800, overlap=80):
    chunks = []
    start = 0
    cid = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunk = text[start:end]
        chunks.append((cid, chunk))
        if end == len(text): break
        start = end - overlap
        cid += 1
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", default="./docs")
    ap.add_argument("--index_dir", default="./rag_index")
    ap.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunk_chars", type=int, default=800)
    ap.add_argument("--chunk_overlap", type=int, default=80)
    args = ap.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)
    model = SentenceTransformer(args.embedding_model)

    texts, meta = [], []
    for fname in os.listdir(args.docs_dir):
        if not fname.endswith((".md",".txt",".csv",".json",".yaml",".yml")):
            continue
        fpath = os.path.join(args.docs_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        chunks = chunk_text(content, args.chunk_chars, args.chunk_overlap)
        for cid, chunk in chunks:
            texts.append(chunk)
            meta.append({"file": fname, "chunk_id": cid})

    print(f"Embedding {len(texts)} chunks...")
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, os.path.join(args.index_dir, "index.faiss"))
    with open(os.path.join(args.index_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Index built:", os.path.join(args.index_dir, "index.faiss"))

if __name__ == "__main__":
    main()
