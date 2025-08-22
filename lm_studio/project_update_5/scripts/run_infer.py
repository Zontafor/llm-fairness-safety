import faiss, numpy as np, json
import os, json, argparse, math, re, requests, yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer

def sanitize(v, lo, hi):
    try:
        x = float(v)
    except:
        return lo
    return max(lo, min(hi, x))

def compute_premium(cfg, rec):
    base = float(cfg["pricing"]["base_premium"])
    rules = cfg["pricing"]["rules"]
    mvr = sanitize(rec.get("MVR_PTS",0), 0, 12)
    clm = sanitize(rec.get("CLM_FREQ",0), 0, 5)
    age = sanitize(rec.get("CAR_AGE",0), 0, 20)
    miles = max(0, float(rec.get("ANNUAL_MILES",0)))
    urb = str(rec.get("URBANICITY","rural")).lower()

    prem = base
    prem += mvr * float(rules["mvr_pts_per_point"])
    prem += clm * float(rules["claims_per_event"])
    prem += age * float(rules["car_age_per_year"])

    if urb not in ["urban","suburban","rural"]:
        urb = "rural"
    prem += float(rules["urbanicity_uplift"][urb])

    if miles <= float(rules["annual_miles_uplift"]["threshold1"]):
        prem += float(rules["annual_miles_uplift"]["uplift1"])
    elif miles <= float(rules["annual_miles_uplift"]["threshold2"]):
        prem += float(rules["annual_miles_uplift"]["uplift2"])
    else:
        prem += float(rules["annual_miles_uplift"]["uplift3"])

    return round(prem)

def load_chunks(index_dir, k=5, query="pricing rules for premium"):
    # Minimal retriever using cosine sim on normalized vectors
    with open(os.path.join(index_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q, k)
    items = []
    for idx, score in zip(I[0], D[0]):
        m = meta[int(idx)]
        items.append({"file": m["file"], "chunk_id": m["chunk_id"], "score": float(score)})
    # load raw text for cites
    # (we reload docs and chunk again for simplicity)
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

    docs_dir = os.path.join(Path(index_dir).parent, "docs")
    file_cache = {}
    for it in items:
        fname = it["file"]
        if fname not in file_cache:
            with open(os.path.join(docs_dir, fname), "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            file_cache[fname] = chunk_text(txt)
        it["text"] = file_cache[fname][it["chunk_id"]][1]
    return items

def call_lmstudio(cfg, system_prompt, user_prompt, max_tokens=512, temperature=0.2):
    base = cfg["lm_studio"]["base_url"].rstrip("/")
    model = cfg["lm_studio"]["chat_model"]
    url = f"{base}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens)
    }

    r = requests.post(url, headers={"Content-Type":"application/json"}, json=payload, timeout=120)

    # Parse robustly
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Non-JSON from LM Studio ({r.status_code}): {r.text[:600]}")

    if r.status_code != 200 or "error" in data:
        raise RuntimeError(f"LM Studio error {r.status_code}: {json.dumps(data)[:800]}")

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Unexpected LM Studio payload (no choices): {json.dumps(data)[:800]}")

    msg = data["choices"][0].get("message", {})
    content = (msg.get("content") or "").strip()
    if not content:
        # fallback for reasoning-style models (e.g., gpt-oss-20b)
        content = (msg.get("reasoning_content") or "").strip()

    if not content:
        raise RuntimeError(f"No content returned. Raw message: {json.dumps(msg)[:800]}")

    return content

# Main call
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--index_dir", default="./rag_index")
    ap.add_argument("--customer_json", required=True, help='e.g. \'{"MVR_PTS":2,"CLM_FREQ":1,"CAR_AGE":7,"URBANICITY":"urban","ANNUAL_MILES":11000}\'')
    ap.add_argument("--k", type=int, default=5, help="# of RAG snippets")
    args = ap.parse_args()

    # load config + system prompt
    cfg = yaml.safe_load(open(args.config, "r"))
    system_prompt = Path("system_prompt.txt").read_text()

    # parse customer record and compute deterministic premium
    rec = json.loads(args.customer_json)
    premium = compute_premium(cfg, rec)

    # retrieve policy evidence from local FAISS
    try:
        evidence = load_chunks(args.index_dir, k=args.k, query="insurance pricing and underwriting guidelines")
        cites = "\n\n".join([f"[{i+1}] {it['file']}#chunk{it['chunk_id']}\n{it['text']}" for i,it in enumerate(evidence)])
    except Exception as e:
        cites = f"(RAG unavailable: {e})"

    # build user prompt to the model
    user_prompt = f"""
Hi LLAMA! How are you? Can you please become an expert in insurance? We want to produce a premium recommendation and a short explanation.

Customer Features:
- MVR_PTS={rec.get('MVR_PTS')}
- CLM_FREQ={rec.get('CLM_FREQ')}
- CAR_AGE={rec.get('CAR_AGE')}
- URBANICITY="{rec.get('URBANICITY')}"
- ANNUAL_MILES={rec.get('ANNUAL_MILES')}

GROUND-TRUTH (from calculator) — use these EXACT numbers in "Premium Math Check":
- Base: $600
- DMV points: $35 × {int(rec.get('MVR_PTS',0))} = ${35*int(rec.get('MVR_PTS',0))}
- Prior claims: $50 × {int(rec.get('CLM_FREQ',0))} = ${50*int(rec.get('CLM_FREQ',0))}
- Car age: $8 × {int(rec.get('CAR_AGE',0))} = ${8*int(rec.get('CAR_AGE',0))}
- Urbanicity uplift ("{str(rec.get('URBANICITY')).lower()}"): ${60 if str(rec.get('URBANICITY','')).lower()=='urban' else 30 if str(rec.get('URBANICITY','')).lower()=='suburban' else 0}
- Annual miles uplift ({int(rec.get('ANNUAL_MILES',0))} miles): ${ 0 if int(rec.get('ANNUAL_MILES',0))<=8000 else 35 if int(rec.get('ANNUAL_MILES',0))<=12000 else 75 }

TOTAL (annual): ${premium}

Using company policy, recommend a monthly premium and explain briefly.
For auditability, include a "Math Check" that shows the deterministic base+uplifts
calculation that yields {premium} USD, and then state the Final Recommendation.
Only use non-protected features.

Requirements:
1) In Premium Math Check, repeat the six lines above exactly and the TOTAL = ${premium}.
2) Then give a Final Recommendation and 2–3 ways to lower the premium.
3) Keep the Safety & Scope Check and Retrieved Policy Evidence sections.
4) Use the retrieved policy evidence below to support your recommendation.
5) If anything is missing, say so briefly.

Please ensure you fully logic through the math and reasoning. Thank you!

Policy Evidence (RAG):
{cites}
""".strip()

    # call LM Studio
    out = call_lmstudio(cfg, system_prompt, user_prompt,
                        max_tokens=cfg.get("eval",{}).get("max_tokens",512),
                        temperature=cfg.get("eval",{}).get("temperature",0.2))

    print("\n=== MODEL RESPONSE ===\n")
    print(out)
    print("\n=== END ===")

if __name__ == "__main__":
    main()