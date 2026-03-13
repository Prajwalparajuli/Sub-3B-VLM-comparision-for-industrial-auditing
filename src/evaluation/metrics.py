"""
metrics.py

Computes three evaluation metrics from parsed model output CSVs:

  - ANLS  (Average Normalized Levenshtein Similarity)
      Measures how close the numeric value the model mentions in its reasoning
      is to the ground-truth reading.  A score of 1.0 means exact match.

  - LCR   (Logic Compliance Rate)
      For each (image, Rule A / Rule B) pair, checks whether the model's verdict
      flips correctly: UNSAFE on the tight rule, SAFE on the lenient rule.
      A model that truly reasons about the constraint should comply on both.

  - Binary Classification Metrics (Precision / Recall / F1)
      Standard binary classification evaluation treating UNSAFE as the positive
      class.  Computes precision, recall, F1, and accuracy.  NO_VERDICT rows
      are excluded from scoring since there is no prediction to evaluate.

Usage:
    python src/evaluation/metrics.py
    python src/evaluation/metrics.py --model internvl2   # single model
    python src/evaluation/metrics.py --save              # also write CSV to results/metrics/
"""

import re
import os
import csv
import argparse
import pandas as pd


PARSED_DIR  = "results/parsed"
METRICS_DIR = "results/metrics"

MODEL_FILES = {
    "SmolVLM":   "smolvlm_parsed.csv",
    "InternVL2": "internvl2_parsed.csv",
    "Janus":     "janus_parsed.csv",
    "Qwen2-VL":  "qwen2_vl_parsed.csv",
    "MiniCPM":   "minicpm_parsed.csv",
}

# ANLS threshold: if normalized edit distance > this, similarity is 0
ANLS_THRESHOLD = 0.5


#ANLS helpers

def levenshtein(a: str, b: str) -> int:
    """Standard dynamic-programming Levenshtein distance, no external libs."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = curr
    return prev[-1]


def anls_single(pred: str, gt: str, threshold: float = ANLS_THRESHOLD) -> float:
    """
    Normalized Levenshtein similarity for one (prediction, ground-truth) pair.
    Returns a value in [0, 1].
    """
    pred = pred.strip().lower()
    gt   = gt.strip().lower()
    if not pred or not gt:
        return 0.0
    nld = levenshtein(pred, gt) / max(len(pred), len(gt))
    return 1.0 - nld if nld <= threshold else 0.0


def extract_first_number(text: str) -> str:
    """
    Pulls the first numeric token (integer or decimal) out of a text blob.
    Specifically removes 'step 1', 'step 2', 'step 3' prefixes so CoT
    headers don't artificially break ANLS evaluation.
    Also strips 'Q1:', 'Q2:', 'Q3:' for Rule Decomposition.
    Returns an empty string if nothing is found.
    """
    if not isinstance(text, str):
        return ""
    
    # Strip out step labels and Q1/Q2/Q3 headers before searching
    clean_text = re.sub(r"(?i)\bstep\s*\d+\b", "", text)
    clean_text = re.sub(r"(?i)\bq[123]\s*:\s*", "", clean_text)
    
    # look for floats / ints, but not inside a word (e.g. skip 'h2o')
    match = re.search(r"(?<!\w)(\d+(?:\.\d+)?)(?!\w)", clean_text)
    return match.group(1) if match else ""


def compute_anls(df: pd.DataFrame) -> float:
    """
    ANLS: compare the first number found in model_reasoning against
    ground_truth_value for every row.  Rows with no number extracted get 0.
    """
    scores = []
    for _, row in df.iterrows():
        gt_str   = str(row["ground_truth_value"]).strip()
        pred_num = extract_first_number(str(row.get("model_reasoning", "")))
        scores.append(anls_single(pred_num, gt_str))
    return sum(scores) / len(scores) if scores else 0.0


# LCR helpers 

def compute_lcr(df: pd.DataFrame) -> dict:
    """
    Logic Compliance Rate.

    Each image has two rows: one for Rule A (expected UNSAFE) and one for
    Rule B (expected SAFE).  A pair is 'compliant' if the model gave the
    correct verdict on BOTH rules.  NO_VERDICT on either rule = non-compliant.

    Returns a dict with:
      - lcr         : fraction of image pairs that are fully compliant
      - pairs_total : number of (image_id, Rule A / Rule B) pairs evaluated
      - pairs_ok    : number that were compliant
    """
    # build a lookup: image_id -> {rule_id: model_verdict}
    pair_map: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        img  = str(row["image_id"]).strip()
        rule = str(row["rule_id"]).strip()   # e.g. "Rule A" or "Rule B"
        verdict = str(row["model_verdict"]).strip()
        expected = str(row["expected_verdict"]).strip()

        if img not in pair_map:
            pair_map[img] = {}
        pair_map[img][rule] = (verdict, expected)

    compliant = 0
    total     = 0

    for img, rules in pair_map.items():
        # we need both Rule A and Rule B to score the pair
        if "Rule A" not in rules or "Rule B" not in rules:
            continue
        total += 1

        verdict_a, expected_a = rules["Rule A"]
        verdict_b, expected_b = rules["Rule B"]

        # compliant = model got the right answer on both rules
        ok = (verdict_a == expected_a) and (verdict_b == expected_b)
        if ok:
            compliant += 1

    return {
        "lcr":         compliant / total if total else 0.0,
        "pairs_total": total,
        "pairs_ok":    compliant,
    }


# ── Binary classification helpers (Precision / Recall / F1) ──────────────────

def compute_binary_clf(df: pd.DataFrame) -> dict:
    """
    Standard binary classification metrics with UNSAFE as the positive class.

    Rows where model_verdict == NO_VERDICT are excluded -- the model produced
    no answer, so there is nothing to evaluate.

    Returns precision, recall, F1, accuracy, and the confusion matrix counts.
    """
    # work only on rows where the model gave a verdict
    scored = df[df["model_verdict"] != "NO_VERDICT"].copy()

    tp = ((scored["model_verdict"] == "UNSAFE") & (scored["expected_verdict"] == "UNSAFE")).sum()
    fp = ((scored["model_verdict"] == "UNSAFE") & (scored["expected_verdict"] == "SAFE")).sum()
    fn = ((scored["model_verdict"] == "SAFE")   & (scored["expected_verdict"] == "UNSAFE")).sum()
    tn = ((scored["model_verdict"] == "SAFE")   & (scored["expected_verdict"] == "SAFE")).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(scored) if len(scored) > 0 else 0.0

    return {
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "accuracy":   accuracy,
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "no_verdict": int((df["model_verdict"] == "NO_VERDICT").sum()),
        "scored_rows": len(scored),
        "total_rows":  len(df),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def read_csv_robust(path: str) -> pd.DataFrame:
    """
    Read a parsed results CSV reliably.

    The parsed CSVs were written with newlines collapsed to spaces, so each
    data row is a single physical line.  We still use the index-based boundary
    detection for safety, since commas inside model_response can fool pandas.
    """
    import csv as _csv

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    lines = raw.splitlines()
    header_row = next(_csv.reader([lines[0]]))
    header = [h.strip() for h in header_row]
    n = len(header)

    # each data row starts with a space-padded integer index + comma
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
            # extra fields come from unquoted commas in the last text column
            # merge them back into a single last field
            parsed = parsed[:n - 1] + [",".join(parsed[n - 1:])]
        rows.append(parsed)

    return pd.DataFrame(rows, columns=header)




def evaluate_model(name: str, csv_path: str) -> dict:
    """Load one parsed CSV and compute all three metrics."""
    df = read_csv_robust(csv_path)

    # strip whitespace from key columns so comparisons are reliable
    for col in ["model_verdict", "expected_verdict", "rule_id", "image_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    anls = compute_anls(df)
    lcr  = compute_lcr(df)
    clf  = compute_binary_clf(df)

    return {"model": name, "anls": anls, **lcr, **clf}


def print_summary(results: list[dict]) -> None:
    """Pretty-print a results table to stdout."""
    header = (
        f"{'Model':<12}  "
        f"{'ANLS':>6}  "
        f"{'LCR':>6}  "
        f"{'Prec':>6}  "
        f"{'Recall':>7}  "
        f"{'F1':>6}  "
        f"{'Acc':>6}  "
        f"{'NoVerd':>7}"
    )
    print("\n" + header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['model']:<12}  "
            f"{r['anls']:>6.3f}  "
            f"{r['lcr']:>6.3f}  "
            f"{r['precision']:>6.3f}  "
            f"{r['recall']:>7.3f}  "
            f"{r['f1']:>6.3f}  "
            f"{r['accuracy']:>6.3f}  "
            f"{r['no_verdict']:>7}"
        )

    print()
    print("ANLS   = Average Normalized Levenshtein Similarity  (numeric reading vs. ground truth)")
    print("LCR    = Logic Compliance Rate  (% images where model flipped verdict correctly)")
    print("Prec   = Precision  (UNSAFE as positive class, NO_VERDICT rows excluded)")
    print("Recall = Recall")
    print("F1     = F1 score")
    print("Acc    = Raw Accuracy on scored (non-NO_VERDICT) rows")
    print("NoVerd = rows where model produced no SAFE/UNSAFE verdict")


def save_results(results: list[dict], out_dir: str) -> None:
    """Write the metrics summary to a CSV file."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics_summary.csv")

    fieldnames = [
        "model", "anls", "lcr", "pairs_total", "pairs_ok",
        "precision", "recall", "f1", "accuracy",
        "tp", "fp", "fn", "tn",
        "no_verdict", "scored_rows", "total_rows",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nMetrics saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute ANLS, LCR, Binary classification metrics.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_FILES.keys()),
        default=None,
        help="Evaluate a single model (default: all models).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save summary to results/metrics/metrics_summary.csv",
    )
    parser.add_argument("--mode", choices=["baseline", "cot", "decomp", "multiturn", "contrast", "contrast_cot"], default="baseline")
    args = parser.parse_args()

    if args.mode == "cot":
        parsed_dir = "results/innovation/cot/parsed"
        metrics_dir = "results/innovation/cot/metrics"
        target_files = {
            "SmolVLM":   "smolvlm_cot_parsed.csv",
            "InternVL2": "internvl2_cot_parsed.csv",
            "Janus":     "janus_cot_parsed.csv",
            "Qwen2-VL":  "qwen2_vl_cot_parsed.csv",
            "MiniCPM":   "minicpm_cot_parsed.csv",
        }
    elif args.mode == "decomp":
        parsed_dir = "results/innovation/decomposition/parsed"
        metrics_dir = "results/innovation/decomposition/metrics"
        target_files = {
            "SmolVLM":   "smolvlm_decomp_parsed.csv",
            "InternVL2": "internvl2_decomp_parsed.csv",
            "Janus":     "janus_decomp_parsed.csv",
            "Qwen2-VL":  "qwen2_vl_decomp_parsed.csv",
            "MiniCPM":   "minicpm_decomp_parsed.csv",
        }
    elif args.mode == "multiturn":
        parsed_dir = "results/innovation/multiturn/parsed"
        metrics_dir = "results/innovation/multiturn/metrics"
        target_files = {
            "SmolVLM":   "smolvlm_multiturn_parsed.csv",
            "InternVL2": "internvl2_multiturn_parsed.csv",
            "Janus":     "janus_multiturn_parsed.csv",
            "Qwen2-VL":  "qwen2_vl_multiturn_parsed.csv",
            "MiniCPM":   "minicpm_multiturn_parsed.csv",
        }
    elif args.mode == "contrast":
        parsed_dir = "results/innovation/contrast/parsed"
        metrics_dir = "results/innovation/contrast/metrics"
        target_files = {
            "SmolVLM":   "smolvlm_contrast_parsed.csv",
            "InternVL2": "internvl2_contrast_parsed.csv",
            "Janus":     "janus_contrast_parsed.csv",
            "Qwen2-VL":  "qwen2_vl_contrast_parsed.csv",
            "MiniCPM":   "minicpm_contrast_parsed.csv",
        }
    elif args.mode == "contrast_cot":
        parsed_dir = "results/innovation/contrast_cot/parsed"
        metrics_dir = "results/innovation/contrast_cot/metrics"
        target_files = {
            "SmolVLM":   "smolvlm_contrast_cot_parsed.csv",
            "InternVL2": "internvl2_contrast_cot_parsed.csv",
            "Janus":     "janus_contrast_cot_parsed.csv",
            "Qwen2-VL":  "qwen2_vl_contrast_cot_parsed.csv",
            "MiniCPM":   "minicpm_contrast_cot_parsed.csv",
        }
    else:
        parsed_dir = "results/baseline/parsed"
        metrics_dir = "results/baseline/metrics"
        target_files = MODEL_FILES

    if args.model:
        targets = {args.model: target_files[args.model]}
    else:
        targets = target_files

    results = []
    for name, fname in targets.items():
        csv_path = os.path.join(parsed_dir, fname)
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
        r = evaluate_model(name, csv_path)
        results.append(r)

    if not results:
        print("No results to display.")
        return

    print_summary(results)

    if args.save:
        save_results(results, metrics_dir)


if __name__ == "__main__":
    main()
