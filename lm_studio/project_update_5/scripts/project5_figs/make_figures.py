
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report_figures.py
Regenerates all appendix figures (A1–A7) for the safety report using the TRUE
datapaths produced by the project's code. It automatically discovers the data
file location based on the defaults defined in evaluate_safety.py and a set of
sensible fallbacks.
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

# -----------------------------
# Path discovery helpers
# -----------------------------

def first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return str(Path(p).resolve())
    return None

# Primary results CSV (produced by evaluate_safety.py --out_csv, default ./results/safety_eval.csv)
RESULT_CANDIDATES = [
    "./results/safety_eval.csv",
    "./safety_eval.csv",
    "/mnt/data/results/safety_eval.csv",
    "/mnt/data/safety_eval.csv",
]

# Optional baseline CSV for A2 (if you kept a pre-mitigation run)
BASELINE_CANDIDATES = [
    "./results/safety_eval_baseline.csv",
    "./safety_eval_baseline.csv",
    "/mnt/data/results/safety_eval_baseline.csv",
    "/mnt/data/safety_eval_baseline.csv",
]

# Optional summary text (evaluate_safety.py writes *_summary.txt next to out_csv)
SUMMARY_CANDIDATES = []
for base in RESULT_CANDIDATES:
    if base.endswith(".csv"):
        SUMMARY_CANDIDATES.append(base.replace(".csv","_summary.txt"))
SUMMARY_CANDIDATES.extend([
    "./results/safety_eval_summary.txt",
    "./safety_eval_summary.txt",
    "/mnt/data/results/safety_eval_summary.txt",
    "/mnt/data/safety_eval_summary.txt",
])

OUTDIR = Path("/mnt/data/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

results_path = first_existing(RESULT_CANDIDATES)
baseline_path = first_existing(BASELINE_CANDIDATES)
summary_path = first_existing(SUMMARY_CANDIDATES)

if not results_path:
    raise FileNotFoundError(
        "Could not locate safety evaluation CSV. Tried:\n  - " +
        "\n  - ".join(RESULT_CANDIDATES)
    )

df = pd.read_csv(results_path)

# Defensive column normalization
expected_cols = {"category","idx","prompt","reply","judgment"}
missing = expected_cols - set(df.columns)
if missing:
    # Try to map common alt names
    rename_map = {}
    if "response" in df.columns and "reply" not in df.columns:
        rename_map["response"] = "reply"
    if "label" in df.columns and "judgment" not in df.columns:
        rename_map["label"] = "judgment"
    if "cat" in df.columns and "category" not in df.columns:
        rename_map["cat"] = "category"
    if rename_map:
        df = df.rename(columns=rename_map)
        missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns for plotting: {missing}.\nColumns present: {list(df.columns)}")

# -----------------------------
# A1. PASS/FAIL by Harm Category
# -----------------------------
def fig_A1_pass_fail_by_category(df):
    # Parse short category label if needed
    short_cat = df["category"].astype(str).str.replace("_"," ").str.replace(":"," - ")
    cats = short_cat.unique().tolist()
    cats_sorted = sorted(cats)  # stable order
    # Build counts
    counts = {c: {"PASS":0,"FAIL":0} for c in cats_sorted}
    for _, row in df.iterrows():
        c = str(row["category"]).replace("_"," ").replace(":"," - ")
        j = str(row["judgment"]).upper()
        if j not in ("PASS","FAIL"):
            continue
        counts[c][j] += 1
    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    x = range(len(cats_sorted))
    pass_vals = [counts[c]["PASS"] for c in cats_sorted]
    fail_vals = [counts[c]["FAIL"] for c in cats_sorted]
    width = 0.35
    ax.bar([i - width/2 for i in x], pass_vals, width=width, label="PASS")
    ax.bar([i + width/2 for i in x], fail_vals, width=width, label="FAIL")
    ax.set_title("A1. PASS/FAIL by Harm Category")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats_sorted, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    out = OUTDIR / "fig_A1_pass_fail_by_category.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

# -----------------------------
# A2. Baseline vs Current Pass Rates
# -----------------------------
def compute_pass_rate(d):
    denom = len(d)
    if denom == 0:
        return 0.0
    num = (d["judgment"].astype(str).str.upper()=="PASS").sum()
    return 100.0 * num / denom

def fig_A2_baseline_vs_current(df, baseline_path=None):
    current_rate = compute_pass_rate(df)
    baseline_rate = None
    if baseline_path and Path(baseline_path).exists():
        try:
            bdf = pd.read_csv(baseline_path)
            baseline_rate = compute_pass_rate(bdf)
        except Exception:
            baseline_rate = None

    # Plot single or dual bars depending on data
    fig, ax = plt.subplots(figsize=(12,6))
    labels = []
    values = []
    if baseline_rate is not None:
        labels.extend(["Baseline","Current"])
        values.extend([baseline_rate, current_rate])
    else:
        labels.append("Current")
        values.append(current_rate)

    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("A2. Baseline vs Current Pass Rates")
    for i,v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    out = OUTDIR / "fig_A2_baseline_vs_current.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

# -----------------------------
# A3. Annotated Adversarial Prompt Responses
# -----------------------------
def truncate(s, n=110):
    s = str(s).replace("\n"," ").strip()
    return s if len(s) <= n else s[:n-1] + "…"

def fig_A3_annotated_examples(df):
    # Aim to include one fail from S6 and S10, and one pass from S1 and S11 where possible
    want = [
        ("S6", "FAIL"),
        ("S10","FAIL"),
        ("S1", "PASS"),
        ("S11","PASS"),
    ]
    rows = []
    for prefix, lab in want:
        subset = df[df["category"].astype(str).str.startswith(prefix) & (df["judgment"].astype(str).str.upper()==lab)]
        if len(subset) == 0:
            continue
        rows.append(subset.iloc[0])

    # Fallback: fill with anything available to make a 3–4 row table
    if len(rows) < 3:
        extra = df.head(4-len(rows)).to_dict("records")
        rows.extend(extra)

    # Build table data
    columns = ["Category","Prompt (excerpt)","Response (excerpt)","Label"]
    table_data = [columns]
    for r in rows:
        table_data.append([
            str(r["category"]).replace("_"," "),
            truncate(r.get("prompt","")),
            truncate(r.get("reply","")),
            str(r["judgment"]).upper(),
        ])

    # Plot as table figure
    fig, ax = plt.subplots(figsize=(20,8))
    ax.axis("off")
    table = ax.table(cellText=table_data, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    ax.set_title("A3. Annotated Adversarial Prompt Responses", pad=20)
    fig.tight_layout()
    out = OUTDIR / "fig_A3_annotated_examples.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

# -----------------------------
# A4. Refusal Template Evolution (static illustrative text)
# -----------------------------
def fig_A4_refusal_template_evolution():
    fig, ax = plt.subplots(figsize=(18,4))
    ax.axis("off")
    before = "Before (terse refusal):\nSorry, I can’t help with that."
    after = ("After (policy-grounded refusal):\n"
             "I can’t provide individualized medical or financial advice. "
             "Here’s general information that may help, and I recommend speaking with "
             "a licensed professional for guidance tailored to your situation.")
    ax.text(0.02, 0.65, before, va="top", ha="left")
    ax.text(0.02, 0.30, after, va="top", ha="left")
    ax.set_title("A4. Refusal Template Evolution", pad=20)
    fig.tight_layout()
    out = OUTDIR / "fig_A4_refusal_template_evolution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

# -----------------------------
# A5. RAG Workflow Architecture (static schematic)
# -----------------------------
def fig_A5_rag_workflow():
    fig, ax = plt.subplots(figsize=(14,7))
    ax.axis("off")

    boxes = [
        (0.05, 0.55, 0.28, 0.25, "CatBoost Predictor\n(fairness-adjusted)"),
        (0.38, 0.55, 0.28, 0.25, "RAG Retrieval\n(vetted policy corpus)"),
        (0.71, 0.55, 0.24, 0.25, "LLM Generation\n(guardrails)"),
        (0.38, 0.15, 0.28, 0.22, "Audit & Logging\n(compliance review)"),
    ]

    for (x,y,w,h,label) in boxes:
        rect = plt.Rectangle((x,y), w, h, fill=False)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, label, ha="center", va="center")

    # arrows
    ax.arrow(0.33, 0.68, 0.04, 0, width=0.003, head_width=0.02, head_length=0.02, length_includes_head=True)
    ax.arrow(0.66, 0.68, 0.03, 0, width=0.003, head_width=0.02, head_length=0.02, length_includes_head=True)
    ax.arrow(0.83, 0.55, -0.20, -0.20, width=0.002, head_width=0.02, head_length=0.02, length_includes_head=True)

    ax.set_title("A5. RAG Workflow Architecture", pad=20)
    fig.tight_layout()
    out = OUTDIR / "fig_A5_rag_workflow.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

# -----------------------------
# A6. Safety Evaluation Pipeline (static schematic)
# -----------------------------
def fig_A6_safety_pipeline():
    steps = [
        "1) Benchmark Prompt Ingestion\n(ALERT, SALAD-Bench)",
        "2) LLM Interaction\n(system prompt + guardrails)",
        "3) Retrieval-Augmented Context\n(vetted documents only)",
        "4) PASS/FAIL Scoring\n(binary rubric)",
        "5) Mitigation Loop\n(disclaimers, templates, adversarial defense)",
        "6) Human Oversight\n(compliance review & logging)",
    ]

    fig, ax = plt.subplots(figsize=(10,8))
    ax.axis("off")

    y = 0.86
    for s in steps:
        rect = plt.Rectangle((0.15, y-0.10), 0.70, 0.10, fill=False)
        ax.add_patch(rect)
        ax.text(0.50, y-0.05, s, ha="center", va="center")
        y -= 0.16
        if y > 0.20:
            ax.arrow(0.50, y+0.04, 0, -0.04, width=0.002, head_width=0.02, head_length=0.02, length_includes_head=True)

    ax.set_title("A6. Safety Evaluation Pipeline", pad=20)
    fig.tight_layout()
    out = OUTDIR / "fig_A6_safety_pipeline.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

# -----------------------------
# A7. Normative Framework Mapping (static grid)
# -----------------------------
def fig_A7_normative_mapping():
    rows = ["Prompt Hardening","Response Templates","Rubric Hook","Adversarial Defense","Human Oversight"]
    cols = ["Deontology","Consequentialism","Virtue Ethics","Rawlsian Justice"]

    # Binary presence matrix (1 means explicitly supported by the framework)
    M = [
        [1,1,1,1],
        [1,1,1,1],
        [1,1,0,1],
        [1,1,1,1],
        [1,1,1,1],
    ]

    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14,7))
    im = ax.imshow(M, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)

    # add simple checkmarks for 1s
    for i in range(len(rows)):
        for j in range(len(cols)):
            if M[i][j] == 1:
                ax.text(j, i, "✓", ha="center", va="center")

    ax.set_title("A7. Normative Framework Mapping", pad=20)
    fig.tight_layout()
    out = OUTDIR / "fig_A7_normative_mapping.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

# -----------------------------
# Run all figures
# -----------------------------
if __name__ == "__main__":
    print("Using results CSV:", results_path)
    if baseline_path:
        print("Using baseline CSV:", baseline_path)
    if summary_path:
        print("Using summary:", summary_path)

    a1 = fig_A1_pass_fail_by_category(df)
    a2 = fig_A2_baseline_vs_current(df, baseline_path)
    a3 = fig_A3_annotated_examples(df)
    a4 = fig_A4_refusal_template_evolution()
    a5 = fig_A5_rag_workflow()
    a6 = fig_A6_safety_pipeline()
    a7 = fig_A7_normative_mapping()

    print("Wrote figures to:", OUTDIR)
    for p in [a1,a2,a3,a4,a5,a6,a7]:
        print(" -", p)
