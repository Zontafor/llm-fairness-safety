import openai
from premium_model import schema_report
from retrieve import recommend_premium_and_answer
from eval import load_prompt_sets, run_safety_eval
from rag import build_or_load_faiss, visualize_chunks, preview_retrieval, retrieval_heatmap

# LM Studio settings (keep consistent across files)
openai.api_base = "http://localhost:1234/v1"
openai.api_key  = "lm-studio"
LLM_MODEL_NAME  = "YOUR-LM-STUDIO-MODEL"

def main():
    # ----- A) Data schema report -----
    schema_report(csv_path="data/insurance.csv", out_dir="figures")

    # ----- B) Build RAG + basic visuals -----
    build_or_load_faiss(docs_dir="docs/policies", qa_jsonl="data/qa_pairs.jsonl")
    visualize_chunks(index_dir="indices", out_dir="figures")

    # Heatmap using a tiny suite of queries (edit to your docs)
    test_queries = {
        "pricing": [
            "smoker surcharge rules",
            "discounts for safe drivers",
            "multi-policy discount",
            "BMI and health premium"
        ],
        "eligibility": [
            "minimum age requirements",
            "pre-existing condition policy",
            "territorial limitations",
            "payment grace period policy"
        ]
    }
    retrieval_heatmap(test_queries, top_k=5, index_dir="indices", out_dir="figures")

    # ----- C) Demo recommendation -----
    customer = {
        "age": 43, "sex": "female", "bmi": 27.9,
        "smoker": "no", "region": "southwest", "children": 2
    }
    result = recommend_premium_and_answer(customer, followup_question="How can I lower this premium?")
    print("\n--- Estimated premium:", f"${result['premium']:,.2f}")
    print("\n--- Top retrieval IDs:", [h["id"] for h in result["retrieval"]])
    print("\n--- Assistant answer:\n", result["answer"])

    # ----- D) Safety harness (ALERT / SALAD) -----
    # Prepare ~10 prompts per category in these JSONs:
    prompt_sets = load_prompt_sets([
        "data/safety/alert_small.json",
        "data/safety/salad_small.json"
    ])
    out_csv = run_safety_eval(prompt_sets, model=LLM_MODEL_NAME, out_dir="eval")
    print("\nSafety worksheet:", out_csv)

if __name__ == "__main__":
    main()