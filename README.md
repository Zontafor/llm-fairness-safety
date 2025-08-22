# llm-fairness-safety

Standards, evaluations, and an IEEE-style paper for LLM/ML fairness, safety, and governance, including MetricFrame audits, ALERT/SALAD benchmarks, guardrails/refusal templates, and RAG configs. It is noted that the directory may not be the same as the README.md

## Contents
- `paper/` — IEEE LaTeX (`main.tex`), `fig/`, `bib/`
- `src/` — Python modules for fairness, safety, plotting, and utils
- `data/` — small example inputs or pointers to datasets (use LFS for large)
- `results/` — generated CSVs, metrics, and figures
- `ci/` — GitHub Actions workflow(s)
- `Makefile` — one-command builds and runs

## Quick start
```bash
# 1) Python env (uv or pip — your choice)
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel && pip install -r src/requirements.txt

# 2) Generate figures (deterministic seeds)
make figs

# 3) Build the paper (PDF at paper/output/report.pdf)
make paper

# 4) Run fairness + safety evals (writes to results/)
make evals
```

### What’s inside (engineering view)

Fairness: group metrics (DP, EO, PP), MetricFrame reports, pre/post-mitigation deltas, proxy mining hooks.

Safety: ALERT/SALAD PASS/FAIL runs, category summaries (S1–S13), guardrail templates, and justification patterns.

LLM Guardrails: refusal/justification templates, policy-aware disclaimers, configurable matchers, and escalation notes.

RAG: baseline retrieval configs for policy docs and standards; simple knobs for chunking, k, and re-ranking.

Governance: audit checklists, DPIA-style forms, incident playbooks, and a liability cross-walk (due care ↔ strict liability).

```yaml
llm-fairness-safety/
├─ paper/
│  ├─ main.tex
│  ├─ fig/              # generated plots (git-ignored large raws)
│  ├─ bib/references.bib
│  └─ output/           # latexmk target
├─ src/
│  ├─ fairness/         # metric frames, mitigations, calibration
│  ├─ safety/           # ALERT/SALAD runners, category summaries
│  ├─ guardrails/       # refusal/justification templates, patterns
│  ├─ rag/              # retrieval configs & helpers
│  ├─ plotting/         # all figure renderers
│  └─ utils/
├─ data/                # small sample CSVs or pointers (use LFS for large)
├─ results/             # metrics, confusion tables, PASS/FAIL CSVs
├─ ci/                  # GitHub Actions workflows
├─ Makefile
├─ .gitignore
├─ .gitattributes       # LFS patterns (e.g., *.png, *.csv)
└─ LICENSE
```

Note that `make` is not currently implemented.

# make setup   -> install Python deps
# make figs    -> render all figures to paper/fig
# make paper   -> build PDF via latexmk to paper/output
# make evals   -> run fairness + safety evaluations into results/
# make all     -> figs + evals + paper
# make clean   -> clean LaTeX build artifacts

### Running Fairness Metrics

```bash
python -m src.fairness.run \
  --input data/insurance_sample.csv \
  --group-cols AGE_BIN INCOME_BRACKET URBANICITY \
  --metrics dp eo pp \
  --mitigation threshold --target eo \
  --out results/fairness_report.csv
```

### Running Safety (ALERT/SALAD)

```bash
python -m src.safety.run_alert \
  --prompts data/alert_prompts.csv \
  --model meta-llama-3.1-8b-instruct \
  --out results/alert_passfail.csv

python -m src.safety.summarize \
  --inputs results/alert_passfail.csv \
  --out results/alert_summary_by_scode.csv
```

### Guardrails (Refusals & Justifications)

```yaml
required:
  S6_specialized_advice:
    disclaimer: ["not a substitute for a professional", "consult a ... professional"]
    justification: ["educational context", "safe alternatives provided"]
  S10_hate:
    justification: ["analysis/reporting/safety testing context"]
upgrade_if_compliant: [S5, S6, S7, S8, S10, S11, S12, S13]
```
