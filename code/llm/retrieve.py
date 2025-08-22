# # # import os
# # # import openai
# # # import pandas as pd
# # # from rag import retrieve
# # # from typing import Dict, List, Optional, Tuple
# # # from premium_model import train_or_load_premium_model

# # # LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")

# # # openai.api_base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
# # # openai.api_key  = os.getenv("OPENAI_API_KEY", "lm-studio")  # LM Studio ignores value but requires a header

# # # SYSTEM = (
# # # "You are an insurance assistant. Given a customer profile, please estimate a premium "
# # # "using the provided numeric recommendation. Then, please justify it with retrieved policy excerpts, "
# # # "and answer follow-up questions clearly. Be honest about uncertainty. Please do not fabricate policy language."
# # # )

# # # def _chat(messages, model):
# # #     resp = openai.ChatCompletion.create(model=model, messages=messages)
# # #     return resp.choices[0].message["content"]

# # # def recommend_premium_and_answer(customer: Dict, followup_question: str = None):
# # #     model = train_or_load_premium_model()  # auto-detects schema
# # #     X = pd.DataFrame([customer])
# # #     est = float(model.predict(X)[0])

# # #     hits = retrieve(f"pricing, eligibility, underwriting guidance for: {customer}", top_k=5)
# # #     context = "\n\n".join([f"[{h['source']}] {h['text']}" for h in hits])

# # #     user_prompt = (
# # #         f"Customer profile:\n{customer}\n\n"
# # #         f"Model-recommended premium: ${est:,.2f} per period.\n\n"
# # #         "Using the context below, explain why this premium is reasonable, "
# # #         "list 2–3 key factors that drove the estimate, and note any caveats.\n\n"
# # #         f"Context:\n{context}\n"
# # #     )
# # #     if followup_question:
# # #         user_prompt += f"\nFollow-up question: {followup_question}\n"

# # #     messages = [
# # #         {"role": "system", "content": SYSTEM},
# # #         {"role": "user", "content": user_prompt}
# # #     ]
# # #     resp = _chat(messages, model=LLM_MODEL_NAME)
# # #     return {
# # #         "premium": est,
# # #         "retrieval": hits,
# # #         "answer": resp.choices[0].message["content"]
# # #     }

# # # def recommend_premium_and_answer(
# # #     customer: Dict,
# # #     followup_question: Optional[str] = None,
# # #     model_name: str = "YOUR-LM-STUDIO-MODEL",
# # # ) -> Dict:
# # #     """
# # #     Returns a dict like:
# # #       {
# # #         "premium_recommendation": <str>,
# # #         "answer": <str>,
# # #         "citations": <list[str]>  # optional, if you include RAG snippets
# # #       }
# # #     """
# # #     # builds the messages you already use (system / user / context / etc.) // keeps current prompt construction
# # #     messages = [
# # #         {"role": "system", "content": "You are a helpful insurance assistant."},
# # #         {"role": "user", "content": f"Customer profile: {customer}"},
# # #     ]
# # #     if followup_question:
# # #         messages.append({"role": "user", "content": followup_question})

# # #     # pass the explicit model_name
# # #     resp = _chat(messages, model=model_name)

# # #     # parsing
# # #     premium_reco = parse_recommendation(resp)
# # #     final_answer  = parse_answer(resp)

# # #     return {
# # #         "premium_recommendation": premium_reco,
# # #         "answer": final_answer,
# # #     }

# # ###
# # # export OPENAI_BASE_URL="http://localhost:1234/v1"
# # # export OPENAI_API_KEY="lm-studio"   # any string; LM Studio just checks header
# # # export LLM_MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
# # ###

# # import os
# # import pandas as pd
# # from typing import Dict, Optional, List
# # from premium_model import train_or_load_premium_model
# # from rag import retrieve

# # # Supports both SDK 1.x (preferred) and old 0.28 if present.
# # def _mk_client():
# #     try:
# #         from openai import OpenAI  # SDK >= 1.0
# #         base_url = (
# #             os.getenv("OPENAI_BASE_URL")
# #             or os.getenv("OPENAI_API_BASE")
# #             or "http://localhost:1234/v1"
# #         )
# #         api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
# #         return OpenAI(base_url=base_url, api_key=api_key), "new"
# #     except Exception:
# #         import openai  # fallback to 0.28
# #         if os.getenv("OPENAI_API_BASE"):
# #             openai.api_base = os.getenv("OPENAI_API_BASE")
# #         elif os.getenv("OPENAI_BASE_URL"):
# #             openai.api_base = os.getenv("OPENAI_BASE_URL")
# #         openai.api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
# #         return openai, "old"

# # _client, _mode = _mk_client()

# # def _chat(messages: List[Dict], model: str, temperature: float = 0.2) -> str:
# #     if _mode == "new":
# #         resp = _client.chat.completions.create(
# #             model=model,
# #             messages=messages,
# #             temperature=temperature,
# #         )
# #         return resp.choices[0].message.content
# #     else:
# #         resp = _client.ChatCompletion.create(
# #             model=model,
# #             messages=messages,
# #             temperature=temperature,
# #         )
# #         # old SDK returns dict-like message
# #         return resp.choices[0].message["content"]

# # # Config
# # LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Meta-Llama-3.1-8B-Instruct")

# # SYSTEM = (
# #     "You are an insurance assistant. Given a customer profile, please estimate a premium "
# #     "using the provided numeric recommendation. Then, please justify it with retrieved policy excerpts, "
# #     "and answer follow-up questions clearly. Be honest about uncertainty. Please do not fabricate policy language."
# # )

# # def _as_model_input(model, features_dict):
# #     # Turn the incoming dict into a single-row DataFrame
# #     x = pd.DataFrame([features_dict])

# #     # If you saved the schema as an attribute on the model:
# #     if hasattr(model, "trained_columns_"):
# #         # Add any missing cols with neutral defaults
# #         for col in model.trained_columns_:
# #             if col not in x.columns:
# #                 # numeric default 0; you can specialize per column if needed
# #                 x[col] = 0
# #         # Drop any unexpected extras and enforce order
# #         x = x[model.trained_columns_]
# #     return x

# # def recommend_premium_and_answer(
# #     customer: Dict,
# #     followup_question: Optional[str] = None,
# #     model_name: str = None,
# # ) -> Dict:
# #     """
# #     Returns:
# #       {
# #         "premium": <float numeric estimate>,
# #         "retrieval": <list of {source, text, ...}>,
# #         "answer": <model response string>
# #       }
# #     (Matches what pipeline_with_bench.run_generation expects.)
# #     """
# #     model_name = model_name or LLM_MODEL_NAME
# #     X_df = _as_model_input(model, dict)
# #     est = float(model.predict(X_df)[0])

# #     # 1) Numeric premium estimate
# #     model = train_or_load_premium_model()  # auto-detects schema
# #     X = pd.DataFrame([customer])
# #     est = float(model.predict(X)[0])

# #     # 2) Retrieve policy context
# #     hits = retrieve(f"pricing, eligibility, underwriting guidance for: {customer}", top_k=5)
# #     context = "\n\n".join([f"[{h.get('source','?')}] {h.get('text','')}" for h in hits])

# #     # 3) Compose prompt
# #     user_prompt = (
# #         f"Customer profile:\n{customer}\n\n"
# #         f"Model-recommended premium: ${est:,.2f} per period.\n\n"
# #         "Using the context below, explain why this premium is reasonable, "
# #         "list 2–3 key factors that drove the estimate, and note any caveats.\n\n"
# #         f"Context:\n{context}\n"
# #     )
# #     if followup_question:
# #         user_prompt += f"\nFollow-up question: {followup_question}\n"

# #     messages = [
# #         {"role": "system", "content": SYSTEM},
# #         {"role": "user",   "content": user_prompt},
# #     ]

# #     # 4) Call LLM
# #     answer_text = _chat(messages, model=model_name)

# #     # 5) Return in the schema expected by pipeline_with_bench.py
# #     return {
# #         "premium": est,
# #         "retrieval": hits,
# #         "answer": answer_text,
# #     }

# # retrieve.py
# import os
# import openai
# import pandas as pd
# from rag import retrieve
# from sklearn import set_config
# from typing import Dict, List, Optional
# from premium_model import train_or_load_premium_model

# LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Meta-Llama-3.1-8B-Instruct")

# SYSTEM = (
#     "You are an insurance assistant. Given a customer profile, please estimate a premium "
#     "using the provided numeric recommendation. Then, please justify it with retrieved policy excerpts, "
#     "and answer follow-up questions clearly. Be honest about uncertainty. Please do not fabricate policy language."
# )

# # -----------------------------
# # Helpers
# # -----------------------------
# def _ensure_trained_columns(model) -> List[str]:
#     """
#     Ensure the fitted Pipeline exposes a list of column names the preprocessor expects.
#     We store this on model.trained_columns_ so we can align inputs at predict time.
#     """
#     if hasattr(model, "trained_columns_"):
#         return model.trained_columns_

#     # Expect a pipeline with a ColumnTransformer under step name 'pre'
#     pre = getattr(model, "named_steps", {}).get("pre", None)
#     cols: List[str] = []
#     if pre is not None and hasattr(pre, "transformers_"):
#         # transformers_: List of tuples like (name, transformer, column_selection)
#         for name, _, colsel in pre.transformers_:
#             if colsel is None:
#                 # If a sub-transformer was given None (rare), skip safely
#                 continue
#             if isinstance(colsel, (list, tuple, pd.Index)):
#                 cols.extend(list(colsel))
#             else:
#                 # Could be slice/array-like; fall back to safe cast
#                 try:
#                     cols.extend(list(colsel))
#                 except Exception:
#                     pass

#     # Fallbacks: try scikit-learn's feature_names_in_ (set when fitting on DataFrame)
#     if not cols and hasattr(model, "feature_names_in_"):
#         cols = list(getattr(model, "feature_names_in_"))

#     # Last resort: do nothing and let downstream raise a clearer error
#     if not cols:
#         raise RuntimeError(
#             "Could not determine trained input columns. "
#             "Make sure your training used a DataFrame and a ColumnTransformer with explicit column lists."
#         )

#     model.trained_columns_ = cols
#     return cols


# def _as_model_input(model, features_dict: Dict) -> pd.DataFrame:
#     """
#     Build a single-row DataFrame aligned to the columns the model expects.
#     Missing columns are added with neutral defaults (0), extras are dropped,
#     and final column order matches training.
#     """
#     # Make a one-row DataFrame from the incoming dict
#     x = pd.DataFrame([features_dict])

#     trained_cols = _ensure_trained_columns(model)

#     # Add missing columns with neutral defaults (0). You can specialize per column if desired.
#     for col in trained_cols:
#         if col not in x.columns:
#             x[col] = 0

#     # Drop any unexpected columns
#     x = x[trained_cols]
#     return x


# # -----------------------------
# # Retrieval (minimal stub)
# # -----------------------------
# def retrieve(query: str, top_k: int = 5) -> List[Dict]:
#     """
#     Minimal placeholder retrieval. Looks for plain-text files under ./policies or ./data/policies,
#     returns top_k short 'hits'. Replace with your real RAG retriever.
#     """
#     candidates = []
#     for root in ["./policies", "./data/policies"]:
#         p = Path(root)
#         if p.exists():
#             for fp in p.glob("**/*.txt"):
#                 try:
#                     text = fp.read_text(encoding="utf-8", errors="ignore")
#                 except Exception:
#                     continue
#                 # Super naive filter
#                 if any(k in text.lower() for k in ["price", "premium", "eligib", "underwrit", "risk"]):
#                     snippet = text[:500].replace("\n", " ")
#                     candidates.append({"source": str(fp), "text": snippet})
#     return candidates[: top_k or 5]


# # -----------------------------
# # Chat (minimal stub)
# # -----------------------------
# def _chat(messages: List[Dict], model: str) -> str:
#     """
#     Deterministic, offline-friendly fallback.
#     It just composes a coherent answer from the provided context and numeric estimate.
#     Replace with your LM Studio / OpenAI / vLLM client as needed.
#     """
#     # Extract user content
#     user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
#     # Try to parse the recommended premium and any context for a plausible justification.
#     try:
#         # Very rough parse
#         lines = user_msg.splitlines()
#         prem_line = [ln for ln in lines if "Model-recommended premium:" in ln]
#         premium_str = prem_line[0].split(":")[1].strip() if prem_line else "$N/A"
#     except Exception:
#         premium_str = "$N/A"

#     response = (
#         f"Recommended premium: {premium_str}\n\n"
#         "Rationale (from retrieved policy context):\n"
#         "• Pricing reflects risk-related factors explicitly referenced in policy language (e.g., age, BMI, smoking status, dependents).\n"
#         "• Underwriting guidance indicates adjustments where elevated risks appear; discounts may apply for mitigating factors.\n"
#         "• Caveats: context may be incomplete; defer to the policy’s binding terms and any state/federal regulations.\n"
#     )
#     if "Follow-up question:" in user_msg:
#         fq = user_msg.split("Follow-up question:", 1)[1].strip()
#         response += f"\nFollow-up answer: {fq} — Based on the same guidance, the premium would be re-evaluated if those details change."
#     return response


# # -----------------------------
# # Main API used by pipeline_with_bench.py
# # -----------------------------
# def recommend_premium_and_answer(
#     customer: Dict,
#     followup_question: Optional[str] = None,
#     model_name: Optional[str] = None,
# ) -> Dict:
#     """
#     Returns a dict matching pipeline_with_bench.run_generation's expectation:
#       {
#         "premium": <float numeric estimate>,
#         "retrieval": <list of {source, text, ...}>,
#         "answer": <model response string>
#       }
#     """
#     # 1) Train or load the estimator and align schema
#     model = train_or_load_premium_model()  # auto-detects schema & fits if missing
#     X_df = _as_model_input(model, customer)
#     est = float(model.predict(X_df)[0])

#     # 2) Retrieve policy context
#     hits = retrieve(f"pricing, eligibility, underwriting guidance for: {json.dumps(customer)}", top_k=5)
#     context = "\n\n".join([f"[{h.get('source','?')}] {h.get('text','')}" for h in hits]) or "(no relevant documents found)"

#     # 3) Compose prompt for LLM
#     model_name = model_name or LLM_MODEL_NAME
#     user_prompt = (
#         f"Customer profile:\n{json.dumps(customer, indent=2)}\n\n"
#         f"Model-recommended premium: ${est:,.2f} per period.\n\n"
#         "Using the context below, explain why this premium is reasonable, "
#         "list 2–3 key factors that drove the estimate, and note any caveats.\n\n"
#         f"Context:\n{context}\n"
#     )
#     if followup_question:
#         user_prompt += f"\nFollow-up question: {followup_question}\n"

#     messages = [
#         {"role": "system", "content": SYSTEM},
#         {"role": "user", "content": user_prompt},
#     ]

#     # 4) Call LLM (or deterministic fallback)
#     answer_text = _chat(messages, model=model_name)

#     # 5) Return
#     return {
#         "premium": est,
#         "retrieval": hits,
#         "answer": answer_text,
#     }

import os
from typing import Dict, Optional, List
import pandas as pd

# --- your local modules ---
# make sure this import path matches your project layout
from premium_model import train_or_load_premium_model
try:
    # if you already have a retriever function, use it
    from rag import retrieve as _retrieve
except Exception:
    _retrieve = None

# LLM config
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Meta-Llama-3.1-8B-Instruct")

SYSTEM = (
    "You are an insurance assistant. Given a customer profile, please estimate a premium "
    "using the provided numeric recommendation. Then, please justify it with retrieved policy excerpts, "
    "and answer follow-up questions clearly. Be honest about uncertainty. Please do not fabricate policy language."
)

# --- minimal chat wrapper; replace with your actual LLM call if you have one ---
def _chat(messages: List[Dict], model: str) -> str:
    # Placeholder: return a compact string if no client is wired up
    # Swap this for your LM Studio/OpenAI/etc. call as needed.
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return (
        "Proposed premium justified by age, BMI, region, and smoking status based on retrieved policy snippets. "
        "Caveats: model uncertainty and missing income/credit_score if not provided. "
        "Follow-up guidance: provide additional underwriting variables for a tighter estimate."
    )

def _safe_retrieve(query: str, top_k: int = 5) -> List[Dict]:
    if _retrieve is None:
        return []
    try:
        return _retrieve(query, top_k=top_k) or []
    except Exception:
        return []

def _as_model_input(model, features_dict: Dict) -> pd.DataFrame:
    """
    Coerce the incoming dict to a single-row DataFrame whose columns exactly match
    the model’s training columns (names & order). Missing columns are filled with 0;
    unexpected extras are dropped.
    """
    x = pd.DataFrame([features_dict])

    if hasattr(model, "trained_columns_"):
        for col in model.trained_columns_:
            if col not in x.columns:
                # neutral default; ColumnTransformer has imputers, but we still need the column to exist
                x[col] = 0
        # Drop extras and enforce order
        x = x[model.trained_columns_]
    # If the attribute is missing (first run), we still return a DataFrame; the preprocessor
    # will select the fitted columns internally.
    return x

def recommend_premium_and_answer(
    customer: Dict,
    followup_question: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict:
    """
    Returns:
      {
        "premium": <float numeric estimate>,
        "retrieval": <list of {source, text, ...}>,
        "answer": <model response string>
      }
    (Matches what pipeline_with_bench.run_generation expects.)
    """
    # 1) Load / train the numeric model FIRST
    model = train_or_load_premium_model()

    # 2) Align input to the training schema
    X_df = _as_model_input(model, customer)

    # 3) Predict numeric premium
    est = float(model.predict(X_df)[0])

    # 4) Retrieve policy context (best‑effort; safe to no‑op)
    hits = _safe_retrieve(
        f"pricing, eligibility, underwriting guidance for: {customer}", top_k=5
    )
    context = "\n\n".join([f"[{h.get('source','?')}] {h.get('text','')}" for h in hits])

    # 5) Compose prompt for the LLM (explanation & follow‑up)
    user_prompt = (
        f"Customer profile:\n{customer}\n\n"
        f"Model-recommended premium: ${est:,.2f} per period.\n\n"
        "Using the context below, explain why this premium is reasonable, "
        "list 2–3 key factors that drove the estimate, and note any caveats.\n\n"
        f"Context:\n{context}\n"
    )
    if followup_question:
        user_prompt += f"\nFollow-up question: {followup_question}\n"

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_prompt},
    ]

    # 6) Call LLM
    answer_text = _chat(messages, model=model_name or LLM_MODEL_NAME)

    # 7) Return in the schema expected by pipeline_with_bench.py
    return {
        "premium": est,
        "retrieval": hits,
        "answer": answer_text,
    }