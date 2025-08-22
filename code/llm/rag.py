import csv
import json
import faiss
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pypdf import PdfReader
from collections import Counter
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast 384-dim

# use FAISS
try:
    import faiss  # type: ignore
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False

# sklearn as a fallback
try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None  # type ignore


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row‑wise L2 normalization for cosine similarity."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

# FAISS inner‑product index with cosine normalization
class _FaissIPIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vecs: np.ndarray) -> None:
        self.index.add(vecs.astype(np.float32))

    def search(self, qvecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # returns (scores, idx)
        scores, idx = self.index.search(qvecs.astype(np.float32), k)
        return scores, idx

# fallback for scikit-learn NearestNeighbors with cosine metric
class _SklearnCosineIndex:
    def __init__(self, dim: int):
        if NearestNeighbors is None:
            raise RuntimeError("scikit-learn not available for FAISS fallback")
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._fitted = False

    def add(self, vecs: np.ndarray) -> None:
        self.nn.fit(vecs)
        self._fitted = True

    def search(self, qvecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Index not built. Call add(...) first.")
        dists, idx = self.nn.kneighbors(qvecs, n_neighbors=k, return_distance=True)
        # cosine distance ∈ [0, 2] // converts to cosine similarity = 1 - dist
        sims = 1.0 - dists
        return sims, idx

# wrapper that normalizes embeddings and uses FAISS if available, else sklearn cosine similarity // provides a consistent interface for searching embeddings.
class VectorIndex:
    def __init__(self, embeddings: np.ndarray):
        embs = embeddings.astype(np.float32)
        self.embs = _l2_normalize(embs)
        dim = self.embs.shape[1]

        if _HAVE_FAISS:
            try:
                self.backend = _FaissIPIndex(dim)
                self.backend.add(self.embs)
            except Exception:
                # last‑ditch fallback even if faiss imported but fails at runtime
                self.backend = _SklearnCosineIndex(dim)
                self.backend.add(self.embs)
        else:
            self.backend = _SklearnCosineIndex(dim)
            self.backend.add(self.embs)

    def search(self, query_embs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = _l2_normalize(query_embs.astype(np.float32))
        scores, idx = self.backend.search(q, k)
        return scores, idx

def _pdf_to_text_chunks(pdf_path: Path, max_chars=1200, overlap=200):
    text = ""
    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += "\n" + page_text

    chunks = []
    step = max_chars - overlap
    for i in range(0, max(1, len(text)), step):
        chunk = text[i:i+max_chars]
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def _load_corpus(docs_dir="docs/policies", qa_jsonl="data/qa_pairs.jsonl"):
    docs_dir = Path(docs_dir)
    items = []
    for pdf in docs_dir.glob("*.pdf"):
        for j, chunk in enumerate(_pdf_to_text_chunks(pdf)):
            items.append({"id": f"{pdf.stem}:{j}", "source": pdf.name, "text": chunk, "type":"policy"})
    qap = Path(qa_jsonl)
    if qap.exists():
        for k, line in enumerate(qap.open()):
            try:
                obj = json.loads(line)
                text = f"Q: {obj.get('question')}\nA: {obj.get('answer')}"
                items.append({"id": f"qa:{k}", "source": "qa_pairs", "text": text, "type":"qa"})
            except Exception:
                continue
    return items

def build_or_load_faiss(index_dir="indices", docs_dir="docs/policies", qa_jsonl="data/qa_pairs.jsonl"):
    index_dir = Path(index_dir); index_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = index_dir / "rag.faiss"
    meta_path  = index_dir / "rag_meta.json"

    if faiss_path.exists() and meta_path.exists():
        index = faiss.read_index(str(faiss_path))
        meta  = json.loads(meta_path.read_text())
        model = SentenceTransformer(meta["embed_model"])
        return index, meta, model

    corpus = _load_corpus(docs_dir, qa_jsonl)
    if not corpus:
        raise RuntimeError(f"No documents found in {docs_dir}. Add PDFs of your vetted policies.")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    embs = model.encode([it["text"] for it in tqdm(corpus, desc="Embedding")],
                        batch_size=64, show_progress_bar=False, normalize_embeddings=True)
   
    # Pure FAISS (cosine via L2-normalized inner product)
    doc_embs = np.asarray(doc_embs, dtype=np.float32)
    query_emb = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)

    doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12)
    query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)

    top_k = 5  # default top-k for retrieval
    dim = doc_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embs)
    scores, idx = index.search(query_emb, top_k)

    meta = {
        "embed_model": EMBED_MODEL_NAME,
        "dim": dim,
        "items": corpus
    }
    faiss.write_index(index, str(faiss_path))
    meta_path.write_text(json.dumps(meta))

    print(f"[rag] Indexed {len(corpus)} chunks from {docs_dir}")
    return index, meta, model

def retrieve(query: str, top_k=5, index_dir="indices"):
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "rag.faiss"))
    meta  = json.loads((index_dir / "rag_meta.json").read_text())
    model = SentenceTransformer(meta["embed_model"])
    q = model.encode([query], normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q, top_k)
    hits = []
    for rank, idx in enumerate(I[0].tolist()):
        item = meta["items"][idx]
        hits.append({
            "rank": rank+1, "score": float(D[0][rank]),
            "id": item["id"], "source": item["source"], "type": item["type"], "text": item["text"]
        })
    return hits

# chunk visualization and retrieval preview
def visualize_chunks(index_dir="indices", out_dir="figures"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    meta  = json.loads((Path(index_dir) / "rag_meta.json").read_text())
    items = meta["items"]

    lengths = [len(x["text"]) for x in items]
    sources = [x["source"] for x in items]
    src_counts = Counter(sources).most_common(10)

    # histogram of chunk lengths
    plt.figure()
    plt.hist(lengths, bins=20)
    plt.title("Chunk Length Distribution (chars)")
    plt.xlabel("Chars"); plt.ylabel("Count")
    path1 = out / "chunk_length_hist.png"
    plt.savefig(path1, bbox_inches="tight"); plt.close()

    # top-10 sources by chunk count
    labels = [s for s,_ in src_counts]
    vals   = [c for _,c in src_counts]
    plt.figure()
    plt.barh(labels, vals)
    plt.title("Top Sources by Chunk Count")
    plt.xlabel("Chunks"); plt.gca().invert_yaxis()
    path2 = out / "top_sources.png"
    plt.savefig(path2, bbox_inches="tight"); plt.close()

    print(f"[visualizer] Saved:\n - {path1}\n - {path2}")

def preview_retrieval(query: str, k=3, index_dir="indices"):
    hits = retrieve(query, top_k=k, index_dir=index_dir)
    print(f"\n[preview] Query: {query}")
    for h in hits:
        print(f"\n#{h['rank']}  score={h['score']:.3f}  src={h['source']}  id={h['id']}\n{h['text'][:700]}...")
    return hits

def retrieval_heatmap(
    test_queries: dict,
    top_k: int = 5,
    index_dir: str = "indices",
    out_dir: str = "figures"
):
    """
    test_queries: dict like
      {
        "pricing": ["smoker surcharge rules", "discounts for safe drivers"],
        "eligibility": ["minimum age requirements", "pre-existing condition policy"]
      }
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(index_dir) / "rag_meta.json").read_text())
    all_sources = sorted({it["source"] for it in meta["items"]})

    # counts[source][rank] -> int
    counts = {s: [0]*top_k for s in all_sources}

    for _, qlist in test_queries.items():
        for q in qlist:
            hits = retrieve(q, top_k=top_k, index_dir=index_dir)
            for h in hits:
                src = h["source"]; r = h["rank"]-1
                if src in counts and 0 <= r < top_k:
                    counts[src][r] += 1

    # rows=sources, cols=ranks
    M = np.array([counts[s] for s in all_sources], dtype=int)

    # Save CSV
    csv_path = Path(out_dir) / "retrieval_heatmap_counts.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source"] + [f"rank_{i+1}" for i in range(top_k)])
        for s, row in zip(all_sources, M):
            writer.writerow([s] + list(map(int, row)))

    # plot heatmap
    plt.figure(figsize=(8, max(3, len(all_sources)*0.3)))
    plt.imshow(M, aspect="auto")
    plt.colorbar(label="frequency")
    plt.yticks(range(len(all_sources)), all_sources)
    plt.xticks(range(top_k), [f"#{i+1}" for i in range(top_k)])
    plt.title("Retrieval heatmap (source × rank)")
    plt.xlabel("Rank"); plt.ylabel("Source")
    out_png = Path(out_dir) / "retrieval_heatmap.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
    print(f"[heatmap] Saved:\n - {csv_path}\n - {out_png}")
    return {"csv": str(csv_path), "png": str(out_png)}