"""
failure_analysis.py

Phase 3.1 - Systematic Error Analysis ("Digital Autopsy")

Reads the parsed CSVs and produces breakdown tables for:

  1. Accuracy by artifact tag
       - Empty artifact_tag rows are labeled "Clean" (no visual challenge)
       - Other tags: Oblique Angle, Obstruction, Low Resolution, etc.

  2. Accuracy by category (gauge vs pipe)

  3. Hard case delta
       - hard_case_flag = 1 vs 0 -- does visual difficulty hurt accuracy?

  4. False positive rate per category (Visual Priming analysis)
       - On rows where expected = SAFE, how often does the model say UNSAFE?
       - High FP on pipe images --> visual priming / confirmation bias

  5. Logic Short-Circuit detection
       - On wrong rows: does the ground_truth_value appear in model_reasoning?
       - If yes, the model read the value correctly but reasoned incorrectly.

All outputs are saved as CSVs to results/failure_analysis/.

Usage:
    python src/evaluation/failure_analysis.py
"""

import re
import os
import csv as _csv
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Run failure analysis on VLM results.")
parser.add_argument("--mode", type=str, choices=["baseline", "cot", "decomp", "multiturn", "contrast", "contrast_cot"], default="baseline",
                    help="Which set of results to analyze")
args = parser.parse_args()

if args.mode == "cot":
    PARSED_DIR  = "results/innovation/cot/parsed"
    OUTPUT_DIR  = "results/innovation/cot/failure_analysis"
    MODEL_FILES = {
        "SmolVLM":   "smolvlm_cot_parsed.csv",
        "InternVL2": "internvl2_cot_parsed.csv",
        "Janus":     "janus_cot_parsed.csv",
        "Qwen2-VL":  "qwen2_vl_cot_parsed.csv",
        "MiniCPM":   "minicpm_cot_parsed.csv",
    }
elif args.mode == "decomp":
    PARSED_DIR  = "results/innovation/decomposition/parsed"
    OUTPUT_DIR  = "results/innovation/decomposition/failure_analysis"
    MODEL_FILES = {
        "SmolVLM":   "smolvlm_decomp_parsed.csv",
        "InternVL2": "internvl2_decomp_parsed.csv",
        "Janus":     "janus_decomp_parsed.csv",
        "Qwen2-VL":  "qwen2_vl_decomp_parsed.csv",
        "MiniCPM":   "minicpm_decomp_parsed.csv",
    }
elif args.mode == "multiturn":
    PARSED_DIR  = "results/innovation/multiturn/parsed"
    OUTPUT_DIR  = "results/innovation/multiturn/failure_analysis"
    MODEL_FILES = {
        "SmolVLM":   "smolvlm_multiturn_parsed.csv",
        "InternVL2": "internvl2_multiturn_parsed.csv",
        "Janus":     "janus_multiturn_parsed.csv",
        "Qwen2-VL":  "qwen2_vl_multiturn_parsed.csv",
        "MiniCPM":   "minicpm_multiturn_parsed.csv",
    }
elif args.mode == "contrast":
    PARSED_DIR  = "results/innovation/contrast/parsed"
    OUTPUT_DIR  = "results/innovation/contrast/failure_analysis"
    MODEL_FILES = {
        "SmolVLM":   "smolvlm_contrast_parsed.csv",
        "InternVL2": "internvl2_contrast_parsed.csv",
        "Janus":     "janus_contrast_parsed.csv",
        "Qwen2-VL":  "qwen2_vl_contrast_parsed.csv",
        "MiniCPM":   "minicpm_contrast_parsed.csv",
    }
elif args.mode == "contrast_cot":
    PARSED_DIR  = "results/innovation/contrast_cot/parsed"
    OUTPUT_DIR  = "results/innovation/contrast_cot/failure_analysis"
    MODEL_FILES = {
        "SmolVLM":   "smolvlm_contrast_cot_parsed.csv",
        "InternVL2": "internvl2_contrast_cot_parsed.csv",
        "Janus":     "janus_contrast_cot_parsed.csv",
        "Qwen2-VL":  "qwen2_vl_contrast_cot_parsed.csv",
        "MiniCPM":   "minicpm_contrast_cot_parsed.csv",
    }
else:
    PARSED_DIR  = "results/baseline/parsed"
    OUTPUT_DIR  = "results/baseline/failure_analysis"
    MODEL_FILES = {
        "SmolVLM":   "smolvlm_parsed.csv",
        "InternVL2": "internvl2_parsed.csv",
        "Janus":     "janus_parsed.csv",
        "Qwen2-VL":  "qwen2_vl_parsed.csv",
        "MiniCPM":   "minicpm_parsed.csv",
    }

MODELS = list(MODEL_FILES.keys())

# ── CSV reader (same approach as metrics.py / parse_results.py) ────────────────

def read_csv_robust(path: str) -> pd.DataFrame:
    """
    Read a parsed results CSV using index-based row boundary detection.
    Handles rows where model_response contains unquoted commas.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    lines = raw.splitlines()
    header_row = next(_csv.reader([lines[0]]))
    header = [h.strip() for h in header_row]
    n = len(header)

    row_start = re.compile(r"^\s*\d+\s*,")

    logical_rows = []
    current_parts = []
    for line in lines[1:]:
        if row_start.match(line) and current_parts:
            logical_rows.append(" ".join(current_parts))
            current_parts = [line]
        else:
            current_parts.append(line)
    if current_parts:
        logical_rows.append(" ".join(current_parts))

    rows = []
    for logical in logical_rows:
        try:
            parsed = next(_csv.reader([logical]))
        except StopIteration:
            continue
        if len(parsed) < n:
            parsed = parsed + [""] * (n - len(parsed))
        elif len(parsed) > n:
            parsed = parsed[:n - 1] + [",".join(parsed[n - 1:])]
        rows.append(parsed)

    return pd.DataFrame(rows, columns=header)


# ── data loading ───────────────────────────────────────────────────────────────

def load_all_models() -> dict[str, pd.DataFrame]:
    """Load and prepare all parsed CSVs."""
    dfs = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(PARSED_DIR, fname)
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue

        df = read_csv_robust(path)

        # strip whitespace from key columns
        for col in ["model_verdict", "expected_verdict", "artifact_tag",
                    "category", "rule_id", "hard_case_flag", "ground_truth_value"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # fill empty artifact_tag with "Clean"
        df["artifact_tag"] = df["artifact_tag"].replace({"": "Clean", "nan": "Clean"})

        # ensure hard_case_flag is boolean-like
        df["hard_case_flag"] = df["hard_case_flag"].replace({"nan": "0"})

        dfs[name] = df

    return dfs


# ── helper: per-model accuracy for a filtered subset ──────────────────────────

def accuracy_for(df: pd.DataFrame) -> dict:
    """
    Compute accuracy on scored rows (exclude NO_VERDICT).
    Returns: correct, wrong, no_verdict, total, accuracy.
    """
    total      = len(df)
    no_verdict = (df["model_verdict"] == "NO_VERDICT").sum()
    scored     = df[df["model_verdict"] != "NO_VERDICT"]
    correct    = (scored["model_verdict"] == scored["expected_verdict"]).sum()
    wrong      = len(scored) - correct
    acc        = correct / total if total > 0 else 0.0
    return {
        "total": total,
        "correct": int(correct),
        "wrong": int(wrong),
        "no_verdict": int(no_verdict),
        "accuracy": round(acc, 4),
    }


# ── Analysis 1: Accuracy by artifact tag ─────────────────────────────────────

def analysis_by_artifact_tag(dfs: dict) -> pd.DataFrame:
    """
    Build a table: rows = artifact tags, columns = one accuracy column per model.
    NO_VERDICT rows are counted in total but excluded from correct/wrong.
    """
    # collect all unique tags across all models
    all_tags = set()
    for df in dfs.values():
        all_tags.update(df["artifact_tag"].unique())
    all_tags = sorted(all_tags)

    records = []
    for tag in all_tags:
        row = {"artifact_tag": tag}
        for model, df in dfs.items():
            subset = df[df["artifact_tag"] == tag]
            stats  = accuracy_for(subset)
            row[f"{model}_acc"]       = stats["accuracy"]
            row[f"{model}_correct"]   = stats["correct"]
            row[f"{model}_wrong"]     = stats["wrong"]
            row[f"{model}_no_verdict"]= stats["no_verdict"]
            row[f"{model}_total"]     = stats["total"]
        records.append(row)

    return pd.DataFrame(records)


# ── Analysis 2: Accuracy by category (gauge vs pipe) ─────────────────────────

def analysis_by_category(dfs: dict) -> pd.DataFrame:
    all_cats = set()
    for df in dfs.values():
        all_cats.update(df["category"].unique())
    all_cats = sorted(all_cats)

    records = []
    for cat in all_cats:
        row = {"category": cat}
        for model, df in dfs.items():
            subset = df[df["category"] == cat]
            stats  = accuracy_for(subset)
            row[f"{model}_acc"]   = stats["accuracy"]
            row[f"{model}_total"] = stats["total"]
        records.append(row)

    return pd.DataFrame(records)


# ── Analysis 3: Hard case delta ───────────────────────────────────────────────

def analysis_hard_case_delta(dfs: dict) -> pd.DataFrame:
    """
    Compare accuracy on hard_case_flag=1 vs hard_case_flag=0.
    Delta = hard_acc - easy_acc (negative means harder cases are worse).
    """
    records = []
    for model, df in dfs.items():
        easy  = df[df["hard_case_flag"] == "0"]
        hard  = df[df["hard_case_flag"] == "1"]
        e_acc = accuracy_for(easy)["accuracy"]
        h_acc = accuracy_for(hard)["accuracy"]
        records.append({
            "model":    model,
            "easy_acc": e_acc,
            "hard_acc": h_acc,
            "delta":    round(h_acc - e_acc, 4),
            "easy_n":   len(easy),
            "hard_n":   len(hard),
        })
    return pd.DataFrame(records)


# ── Analysis 4: False positive rate (Visual Priming) ─────────────────────────

def analysis_false_positives(dfs: dict) -> pd.DataFrame:
    """
    On rows where expected_verdict = SAFE and model gave a verdict:
    - How often does the model say UNSAFE? (false positive = visual priming)
    Broken down by category.
    """
    records = []
    for model, df in dfs.items():
        # only rows that had a verdict and were expected to be safe
        safe_rows = df[
            (df["expected_verdict"] == "SAFE") &
            (df["model_verdict"] != "NO_VERDICT")
        ]
        for cat in sorted(df["category"].unique()):
            subset = safe_rows[safe_rows["category"] == cat]
            if len(subset) == 0:
                continue
            fp = (subset["model_verdict"] == "UNSAFE").sum()
            records.append({
                "model":    model,
                "category": cat,
                "safe_expected_n": len(subset),
                "false_positives": int(fp),
                "fp_rate": round(fp / len(subset), 4) if len(subset) > 0 else 0.0,
            })

    return pd.DataFrame(records)


# ── Analysis 5: Logic Short-Circuit detection ─────────────────────────────────

def analysis_short_circuit(dfs: dict) -> pd.DataFrame:
    """
    For rows where model verdict is WRONG and not NO_VERDICT:
    - Check if the ground_truth_value string appears in model_reasoning.
    - If yes: model saw the right value but still reasoned incorrectly.
      This is a Logic Short-Circuit (reasoning failure, not perception failure).
    - If no: model likely misread the value too (perception failure).
    """
    records = []
    for model, df in dfs.items():
        scored = df[df["model_verdict"] != "NO_VERDICT"]
        wrong  = scored[scored["model_verdict"] != scored["expected_verdict"]]

        short_circuit = 0
        perception    = 0

        for _, row in wrong.iterrows():
            gt    = str(row.get("ground_truth_value", "")).strip()
            rtext = str(row.get("model_reasoning", "")).lower()

            # check if the numeric ground truth value appears in the reasoning
            if gt and gt.lower() in rtext:
                short_circuit += 1
            else:
                perception += 1

        total_wrong = len(wrong)
        records.append({
            "model":              model,
            "total_wrong":        total_wrong,
            "short_circuit":      short_circuit,
            "perception_failure": perception,
            "short_circuit_pct":  round(short_circuit / total_wrong, 4) if total_wrong > 0 else 0.0,
        })

    return pd.DataFrame(records)


# ── printing helpers ───────────────────────────────────────────────────────────

def print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading parsed CSVs...")
    dfs = load_all_models()
    print(f"Loaded: {list(dfs.keys())}")

    # 1. Accuracy by artifact tag
    tag_df = analysis_by_artifact_tag(dfs)
    tag_df.to_csv(os.path.join(OUTPUT_DIR, "accuracy_by_artifact_tag.csv"), index=False)

    # quick readable summary: just the _acc columns
    acc_cols = ["artifact_tag"] + [f"{m}_acc" for m in MODELS if m in dfs]
    print_section("1. Accuracy by Artifact Tag", tag_df[acc_cols])

    # 2. Accuracy by category
    cat_df = analysis_by_category(dfs)
    cat_df.to_csv(os.path.join(OUTPUT_DIR, "accuracy_by_category.csv"), index=False)
    cat_cols = ["category"] + [f"{m}_acc" for m in MODELS if m in dfs]
    print_section("2. Accuracy by Category (gauge vs pipe)", cat_df[cat_cols])

    # 3. Hard case delta
    hard_df = analysis_hard_case_delta(dfs)
    hard_df.to_csv(os.path.join(OUTPUT_DIR, "hard_case_delta.csv"), index=False)
    print_section("3. Hard Case Delta (hard - easy accuracy)", hard_df)

    # 4. False positive rate (Visual Priming)
    fp_df = analysis_false_positives(dfs)
    fp_df.to_csv(os.path.join(OUTPUT_DIR, "false_positive_rate.csv"), index=False)
    print_section("4. False Positive Rate by Category (Visual Priming)", fp_df)

    # 5. Logic Short-Circuit
    sc_df = analysis_short_circuit(dfs)
    sc_df.to_csv(os.path.join(OUTPUT_DIR, "short_circuit_analysis.csv"), index=False)
    print_section("5. Logic Short-Circuit vs Perception Failure", sc_df)

    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
