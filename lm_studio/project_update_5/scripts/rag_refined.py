
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Features
--------
1) Corpus & Manifest:
   - Directory-based corpus with a YAML manifest containing per-document metadata.
   - Safety whitelist (source_type, jurisdiction, effective_date window).

2) Chunking:
   - Sentence-aware chunking with token-ish size limit and overlap.
   - Stores per-chunk metadata and SHA-1 for change detection.

3) Embeddings:
   - Backends:
       a) LM Studio embeddings API (OpenAI-compatible): set env LMSTUDIO_EMBED_URL and LMSTUDIO_EMBED_MODEL.
       b) SentenceTransformer (if installed), e.g., "all-MiniLM-L6-v2".
       c) Deterministic hashing fallback (for smoke tests only) when no backend present.
   - Caches per-chunk embeddings on disk to avoid recomputation.

4) Indexing:
   - FAISS index (if installed), cosine similarity; falls back to sklearn NearestNeighbors.
   - Persisted to disk under index_dir.

5) Retrieval:
   - Semantic search over embeddings with optional metadata filters & whitelist.
   - Lightweight hybrid boost with keyword scoring.
   - Deduplication & safety filtering.

6) Prompt Assembly:
   - Builds a system prompt block with top-k passages and inline citations.
   - Returns both the prompt string and a machine-readable provenance object.

CLI:
----
Build index:
    python rag_section_2b.py build --config rag_config.yaml

Query:
    python rag_section_2b.py query --config rag_config.yaml --q "What are the cancellation rules?" --k 5

Author: Michelle L. Wu
License: MIT
"""

from __future__ import annotations

import os
import re
import io
import gc
import sys
import json
import math
import time
import glob
import uuid
import yaml
import hashlib
import logging
import pickle
import random
import string
import base64
import numpy as np
import typing as T
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Optional deps
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    HAVE_SK = True
except Exception:
    HAVE_SK = False

try:
    import requests  # for LM Studio embeddings
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    HAVE_ST = True
except Exception:
    HAVE_ST = False


# Configuration dataclasses
@dataclass
class CorpusDoc:
    doc_id: str
    title: str
    path: str
    source_type: str  # e.g., "compliance", "regulation", "fairness_audit"
    jurisdiction: str # e.g., "US", "CA", "EU"
    effective_date: str # YYYY-MM-DD
    url: T.Optional[str] = None


@dataclass
class Config:
    # Paths
    corpus_dir: str
    manifest_path: str
    index_dir: str

    # Chunking
    chunk_chars: int = 900
    chunk_overlap: int = 150
    min_chunk_chars: int = 300

    # Embeddings
    embedding_backend: str = "lmstudio"  # "lmstudio" | "sentencetransformers" | "fallback"
    sentencetransformers_model: str = "all-MiniLM-L6-v2"
    lmstudio_embed_url: T.Optional[str] = None
    lmstudio_embed_model: T.Optional[str] = None
    embedding_dim: int = 384  # used for ST & fallback

    # Indexing
    use_faiss: bool = True
    normalize: bool = True

    # Retrieval
    top_k: int = 6
    whitelist_source_types: T.List[str] = None
    whitelist_jurisdictions: T.List[str] = None
    earliest_effective_date: T.Optional[str] = None  # "YYYY-MM-DD"
    keyword_boost: float = 0.15

    # Prompt
    max_prompt_tokens: int = 2800  # soft budget for retrieved snippets
    citation_style: str = "([{doc_id}:{chunk_id}] {title}, {effective_date})"

    # Safety
    blocked_patterns: T.List[str] = None

    def post_init(self):
        if self.whitelist_source_types is None:
            self.whitelist_source_types = ["compliance", "regulation", "fairness_audit"]
        if self.whitelist_jurisdictions is None:
            self.whitelist_jurisdictions = ["US"]
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r"\bhow\s+to\s+build\s+a\s+weapon\b",
                r"\bchild\s+sexual\b",
                r"\bdo\s+harm\s+to\b",
            ]
        if self.lmstudio_embed_url is None:
            self.lmstudio_embed_url = os.environ.get("LMSTUDIO_EMBED_URL", None)
        if self.lmstudio_embed_model is None:
            self.lmstudio_embed_model = os.environ.get("LMSTUDIO_EMBED_MODEL", None)


# Utilities
def read_yaml(p: T.Union[str, Path]) -> T.Any:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(p: T.Union[str, Path], data: T.Any) -> None:
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def list_text_files(root: Path) -> T.List[Path]:
    return [Path(p) for p in glob.glob(str(root / "**" / "*"), recursive=True) if Path(p).is_file() and p.lower().endswith((".txt", ".md"))]


def sentence_split(text: str) -> T.List[str]:
    # Very simple sentence splitter to avoid external deps
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9])", text.strip())
    # Fall back if splitting produced nonsense
    if not parts or sum(len(p) for p in parts) < 0.6 * len(text):
        return [text.strip()]
    return parts


def chunk_text(text: str, chunk_chars: int, overlap: int, min_chunk_chars: int) -> T.List[str]:
    sents = sentence_split(text)
    chunks, cur = [], ""
    for sent in sents:
        if len(cur) + len(sent) + 1 <= chunk_chars:
            cur = (cur + " " + sent).strip()
        else:
            if len(cur) >= min_chunk_chars:
                chunks.append(cur)
            else:
                # if too small, try to append one more sentence
                cur = (cur + " " + sent).strip()
                if len(cur) >= min_chunk_chars:
                    chunks.append(cur)
                    cur = ""
                    continue
            # new chunk starts with overlap tail of previous
            tail = cur[-overlap:]
            cur = tail + " " + sent
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

# Embedding backends
class Embedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backend = cfg.embedding_backend.lower()
        self.dim = cfg.embedding_dim
        self._st_model = None
        if self.backend == "sentencetransformers" and HAVE_ST:
            self._st_model = SentenceTransformer(cfg.sentencetransformers_model)
            try:
                self.dim = self._st_model.get_sentence_embedding_dimension()
            except Exception:
                pass

    def embed(self, texts: T.List[str]) -> np.ndarray:
        if self.backend == "lmstudio":
            return self._embed_lmstudio(texts)
        elif self.backend == "sentencetransformers" and HAVE_ST:
            vecs = self._st_model.encode(texts, normalize_embeddings=False, convert_to_numpy=True)
            return vecs.astype(np.float32)
        else:
            logging.warning("Using deterministic fallback embeddings; NOT suitable for production.")
            return self._embed_fallback(texts)

    def _embed_lmstudio(self, texts: T.List[str]) -> np.ndarray:
        if not HAVE_REQUESTS:
            raise RuntimeError("requests not installed; cannot call LM Studio embeddings API.")
        url = self.cfg.lmstudio_embed_url
        model = self.cfg.lmstudio_embed_model
        if not url or not model:
            raise RuntimeError("LMSTUDIO_EMBED_URL and LMSTUDIO_EMBED_MODEL must be set for lmstudio backend.")
        headers = {"Content-Type": "application/json"}
        payload = {"model": model, "input": texts}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # OpenAI-compatible: data["data"][i]["embedding"]
        embs = [np.array(row["embedding"], dtype=np.float32) for row in data.get("data", [])]
        if not embs:
            raise RuntimeError(f"LM Studio returned no embeddings. Response keys: {list(data.keys())}")
        # set dim from first embedding
        self.dim = int(embs[0].shape[0])
        return np.vstack(embs)

    def _embed_fallback(self, texts: T.List[str]) -> np.ndarray:
        # Deterministic 384-dim pseudo-embeddings via hashing (for smoke tests only)
        rng = np.random.default_rng(42)
        base = rng.normal(size=(384,), loc=0.0, scale=1.0).astype(np.float32)
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            bits = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            v = np.interp(np.arange(384), np.linspace(0, bits.size-1, 384), bits)
            out.append((v + base).astype(np.float32))
        return np.vstack(out)

# Index
class VectorIndex:
    def __init__(self, dim: int, use_faiss: bool = True, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize
        self.use_faiss = use_faiss and HAVE_FAISS
        self.index = None
        self._vectors = None  # fallback storage
        self._ids = []

        if self.use_faiss:
            self.index = faiss.IndexFlatIP(dim)
        else:
            if HAVE_SK:
                self.index = NearestNeighbors(metric="cosine")
            else:
                self.index = None

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def add(self, vecs: np.ndarray, ids: T.List[str]):
        assert vecs.shape[0] == len(ids)
        if self.normalize:
            vecs = self._l2_normalize(vecs)
        if self.use_faiss:
            self.index.add(vecs)
            self._ids.extend(ids)
        else:
            if HAVE_SK:
                # store for SK fit later
                if self._vectors is None:
                    self._vectors = vecs
                else:
                    self._vectors = np.vstack([self._vectors, vecs])
                self._ids.extend(ids)
            else:
                # naive storage
                if self._vectors is None:
                    self._vectors = vecs
                else:
                    self._vectors = np.vstack([self._vectors, vecs])
                self._ids.extend(ids)

    def finalize(self):
        if not self.use_faiss:
            if HAVE_SK and self._vectors is not None:
                self.index.fit(self._vectors)
        return self

    def save(self, path: Path):
        meta = {"dim": self.dim, "normalize": self.normalize, "use_faiss": self.use_faiss, "ids": self._ids}
        with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if self.use_faiss:
            faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        else:
            with open(path.with_suffix(".pkl"), "wb") as f:
                pickle.dump({"vectors": self._vectors, "ids": self._ids, "use_sk": HAVE_SK}, f)

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        with open(path.with_suffix(".meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        vi = cls(meta["dim"], meta["use_faiss"], meta["normalize"])
        vi._ids = meta["ids"]
        if meta["use_faiss"] and HAVE_FAISS:
            vi.index = faiss.read_index(str(path.with_suffix(".faiss")))
        else:
            with open(path.with_suffix(".pkl"), "rb") as f:
                data = pickle.load(f)
            vi._vectors = data["vectors"]
            if data.get("use_sk", False) and HAVE_SK:
                vi.index = NearestNeighbors(metric="cosine")
                vi.index.fit(vi._vectors)
            else:
                vi.index = None
        return vi

    def search(self, q: np.ndarray, top_k: int = 5) -> T.Tuple[np.ndarray, T.List[str]]:
        if self.normalize:
            q = self._l2_normalize(q)
        if self.use_faiss:
            D, I = self.index.search(q, top_k)
            ids = [[self._ids[idx] for idx in row] for row in I]
            return D, ids
        else:
            if HAVE_SK and self.index is not None:
                distances, indices = self.index.kneighbors(q, n_neighbors=top_k, return_distance=True)
                sims = 1.0 - distances
                ids = [[self._ids[idx] for idx in row] for row in indices]
                return sims, ids
            else:
                # naive cosine with numpy
                sims = []
                ids = []
                for qi in q:
                    scores = (self._vectors @ qi) / (np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(qi) + 1e-12)
                    top_idx = np.argsort(-scores)[:top_k]
                    sims.append(scores[top_idx])
                    ids.append([self._ids[i] for i in top_idx])
                return np.array(sims), ids


# RAG store
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_type: str
    jurisdiction: str
    effective_date: str
    path: str
    url: T.Optional[str] = None


class RAGStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.post_init()
        self.index_dir = Path(cfg.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = Embedder(cfg)
        self.index_path = self.index_dir / "vector.index"
        self.meta_path = self.index_dir / "chunks.jsonl"
        self._index: T.Optional[VectorIndex] = None
        self._chunks: T.Dict[str, Chunk] = {}

    def build(self):
        logging.info("Loading manifest and corpus...")
        manifest = read_yaml(self.cfg.manifest_path)
        corpus_docs = [CorpusDoc(**row) for row in manifest.get("documents", [])]

        all_chunks: T.List[Chunk] = []
        for doc in corpus_docs:
            full_path = Path(self.cfg.corpus_dir) / doc.path
            if not full_path.exists():
                logging.warning(f"Missing file for doc_id={doc.doc_id}: {full_path}")
                continue
            text = Path(full_path).read_text(encoding="utf-8")
            for i, ch in enumerate(chunk_text(text, self.cfg.chunk_chars, self.cfg.chunk_overlap, self.cfg.min_chunk_chars)):
                chunk_id = f"{doc.doc_id}:{i:04d}"
                all_chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=ch,
                    source_type=doc.source_type,
                    jurisdiction=doc.jurisdiction,
                    effective_date=doc.effective_date,
                    path=str(full_path),
                    url=doc.url
                ))

        # Save metadata
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for ch in all_chunks:
                f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

        # Embeddings (cached by hash)
        logging.info("Computing embeddings...")
        texts = [c.text for c in all_chunks]
        hashes = [sha1_text(t) for t in texts]

        cache_path = self.index_dir / "emb_cache.pkl"
        if cache_path.exists():
            cache = pickle.load(open(cache_path, "rb"))
        else:
            cache = {}

        vecs = np.zeros((len(texts), self.embedder.dim), dtype=np.float32)
        to_compute, idxs = [], []
        for i, h in enumerate(hashes):
            if h in cache:
                vecs[i] = cache[h]
            else:
                to_compute.append(texts[i])
                idxs.append(i)

        if to_compute:
            new_vecs = self.embedder.embed(to_compute)
            if new_vecs.shape[1] != vecs.shape[1]:
                vecs = np.zeros((len(texts), new_vecs.shape[1]), dtype=np.float32)
            for j, i in enumerate(idxs):
                vecs[i] = new_vecs[j]
                cache[hashes[i]] = new_vecs[j]
            pickle.dump(cache, open(cache_path, "wb"))

        # Build index
        logging.info("Building index...")
        index = VectorIndex(dim=vecs.shape[1], use_faiss=self.cfg.use_faiss, normalize=self.cfg.normalize)
        ids = [c.chunk_id for c in all_chunks]
        index.add(vecs, ids)
        index.finalize()
        index.save(self.index_path)

        # Local cache
        self._index = index
        self._chunks = {c.chunk_id: c for c in all_chunks}
        logging.info(f"Built index with {len(all_chunks)} chunks.")

    def _ensure_loaded(self):
        if self._index is None:
            self._index = VectorIndex.load(self.index_path)
        if not self._chunks:
            self._chunks = {}
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    self._chunks[d["chunk_id"]] = Chunk(**d)

    # Returns (system_prompt_block, provenance_dict)
    def query(self, q: str, top_k: int = None, filters: T.Optional[dict] = None) -> T.Tuple[str, dict]:
        self._ensure_loaded()
        top_k = top_k or self.cfg.top_k
        filters = filters or {}

        # Safety: blocked patterns
        for pat in self.cfg.blocked_patterns:
            if re.search(pat, q, flags=re.IGNORECASE):
                raise ValueError(f"Query contains blocked pattern: {pat}")

        # Embed query
        q_vec = self.embedder.embed([q])

        # Search
        sims, ids = self._index.search(q_vec, top_k=top_k*3)  # oversample for filtering
        cand_ids = ids[0]
        cand_scores = sims[0]

        # Keyword boost
        kws = set(w.lower() for w in re.findall(r"[A-Za-z]{3,}", q))
        boosted = []
        for cid, s in zip(cand_ids, cand_scores):
            ch = self._chunks[cid]
            text_lower = ch.text.lower()
            kw_hits = sum(1 for kw in kws if kw in text_lower)
            score = float(s) + self.cfg.keyword_boost * kw_hits
            boosted.append((score, cid))

        boosted.sort(reverse=True, key=lambda x: x[0])

        # Metadata filtering
        results = []
        for score, cid in boosted:
            ch = self._chunks[cid]
            if ch.source_type not in self.cfg.whitelist_source_types:
                continue
            if ch.jurisdiction not in self.cfg.whitelist_jurisdictions:
                continue
            if self.cfg.earliest_effective_date:
                if ch.effective_date < self.cfg.earliest_effective_date:
                    continue
            # Additional user filters
            ok = True
            for k, v in filters.items():
                if getattr(ch, k) != v:
                    ok = False
                    break
            if ok:
                results.append((score, ch))
            if len(results) >= top_k:
                break

        # Assemble system prompt
        prompt_lines = ["# Retrieved Policy Context (verbatim excerpts; cite by [doc:chunk])\n"]
        prov = {"query": q, "results": []}
        token_budget = self.cfg.max_prompt_tokens
        used = 0
        for rank, (score, ch) in enumerate(results, start=1):
            citation = self.cfg.citation_style.format(
                doc_id=ch.doc_id, chunk_id=ch.chunk_id.split(":")[-1], title=ch.title, effective_date=ch.effective_date
            )
            snippet = ch.text.strip()
            approx_tokens = int(len(snippet) / 4)  # rough heuristic
            if used + approx_tokens > token_budget and rank > 1:
                break
            prompt_lines.append(f"[{ch.doc_id}:{ch.chunk_id.split(':')[-1]}] {ch.title} ({ch.effective_date})")
            prompt_lines.append(snippet)
            prompt_lines.append("")  # blank line
            used += approx_tokens
            prov["results"].append({
                "rank": rank,
                "score": score,
                "chunk_id": ch.chunk_id,
                "doc_id": ch.doc_id,
                "title": ch.title,
                "effective_date": ch.effective_date,
                "path": ch.path,
                "url": ch.url,
            })

        system_block = "\n".join(prompt_lines).strip()
        return system_block, prov

# Prompt helper
SYSTEM_TEMPLATE = """Hi LLAMA! How are you today? Can you please become an expert in insurance policy? Please use only the retrieved context to answer.
If a user asks for individualized medical/legal/financial advice, return a safe general explanation, include a disclaimer that you are not a substitute for a professional, and suggest safe next steps.
For any content that could be hate or harassment, answer only in a neutral, analytical context and do not endorse claims.

<<CONTEXT>>
{context}

<<INSTRUCTIONS>>
- Cite using [doc:chunk] markers.
- If context is insufficient, say so and ask for permission to fetch more.
- Keep responses concise and professional.
"""

def assemble_system_prompt(context_block: str) -> str:
    return SYSTEM_TEMPLATE.format(context=context_block)


# CLI
def load_config(path: T.Union[str, Path]) -> Config:
    cfg_d = read_yaml(path)
    cfg = Config(**cfg_d)
    cfg.post_init()
    return cfg


def _cmd_build(cfg: Config):
    store = RAGStore(cfg)
    store.build()
    print(f"Built index at: {store.index_path}")


def _cmd_query(cfg: Config, q: str, k: int):
    store = RAGStore(cfg)
    context, prov = store.query(q, top_k=k)
    prompt = assemble_system_prompt(context)
    print("\n===== SYSTEM PROMPT (for your LLM) =====\n")
    print(prompt)
    print("\n===== PROVENANCE (JSON) =====\n")
    print(json.dumps(prov, ensure_ascii=False, indent=2))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="RAG Section 2B implementation")
    p.add_argument("cmd", choices=["build", "query"])
    p.add_argument("--config", required=True, help="Path to rag_config.yaml")
    p.add_argument("--q", default=None, help="Query (for 'query' cmd)")
    p.add_argument("--k", type=int, default=5, help="Top-k passages")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)

    if args.cmd == "build":
        _cmd_build(cfg)
    elif args.cmd == "query":
        if not args.q:
            p.error("--q is required for 'query'")
        _cmd_query(cfg, args.q, args.k)


if __name__ == "__main__":
    main()
