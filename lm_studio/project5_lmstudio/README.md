# Project Update 5: Toy LLM for Personalized Policy Recommendations

This project update builds a toy LLM (Meta Llama 3.1, 8B via LM Studio) for generating personalized insurance premium recommendations.  

The system instructions are in `system_prompt.txt`, the retrieval-augmented generation corpus and index are in `rag_config.yaml`, `index.faiss`, and `meta.json`, and the safety evaluations are documented in `safety_eval.csv` and `safety_eval_summary.txt`. The project uses `customers_sample.csv` for smoke testing and `insurance.csv` for continued evaluation. See `docs/manifest.yaml` for file mapping.