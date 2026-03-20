import os
import sys
import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats

# Ensure the script can locate metrics.py when run from the root directory
sys.path.append(os.path.join(os.path.dirname(__file__)))
from metrics import evaluate_model # Reuse the logic from the existing metrics script

def compute_confidence_interval(data, confidence=0.95):
    """
    Computes the 95% confidence interval using the t-distribution
    for small sample sizes (N=3).
    """
    n = len(data)
    if n < 2:
        return 0.0
    
    m = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics across N runs.")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to aggregate")
    parser.add_argument("--mode", choices=["baseline", "cot", "decomp", "multiturn"], default="baseline")
    args = parser.parse_args()

    models = ["smolvlm", "internvl2", "janus", "qwen2_vl", "minicpm"]
    model_display_names = {
        "smolvlm": "SmolVLM",
        "internvl2": "InternVL2",
        "janus": "Janus",
        "qwen2_vl": "Qwen2-VL",
        "minicpm": "MiniCPM"
    }

    base_dir = f"results/{args.mode}/parsed"
    
    print(f"\n==================================================")
    print(f"Aggregating {args.runs} runs for {args.mode.upper()} mode")
    print(f"==================================================")

    aggregated_results = []

    for model_key in models:
        runs_data = {
            "anls": [],
            "lcr": [],
            "accuracy": [],
            "f1": [],
            "fpr": [],
            "fnr": []
        }
        
        valid_runs = 0

        for i in range(1, args.runs + 1):
            file_name = f"{model_key}_run_{i}_parsed.csv"
            file_path = os.path.join(base_dir, file_name)

            if not os.path.exists(file_path):
                continue
                
            # Compute point metrics for this specific run
            res = evaluate_model(model_display_names[model_key], file_path)
            
            runs_data["anls"].append(res["anls"])
            runs_data["lcr"].append(res["lcr"])
            runs_data["accuracy"].append(res["accuracy"])
            runs_data["f1"].append(res["f1"])
            
            # Calculate FPR and FNR
            # FPR = FP / (FP + TN)
            # FNR = FN / (FN + TP)
            fp = res.get("fp", 0)
            tn = res.get("tn", 0)
            fn = res.get("fn", 0)
            tp = res.get("tp", 0)
            
            fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = (fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            
            runs_data["fpr"].append(fpr)
            runs_data["fnr"].append(fnr)
            
            valid_runs += 1

        if valid_runs > 0:
            # Aggregate the stats
            agg_entry = {"Model": model_display_names[model_key], "Runs": valid_runs}
            
            for metric in ["accuracy", "lcr", "f1", "anls", "fpr", "fnr"]:
                data = runs_data[metric]
                mean_val = np.mean(data)
                std_val = np.std(data, ddof=1) if len(data) > 1 else 0.0
                ci_val = compute_confidence_interval(data)
                
                # Store formatted string for display
                agg_entry[f"{metric.upper()}"] = f"{mean_val:.3f} ± {ci_val:.3f}"
                # Store raw floats for internal use
                agg_entry[f"raw_{metric}_mean"] = mean_val
                agg_entry[f"raw_{metric}_std"] = std_val

            aggregated_results.append(agg_entry)
        else:
            print(f"[skip] No run files found for {model_key} in {base_dir}")

    if not aggregated_results:
        print("No valid data to aggregate.")
        return

    # Create the DataFrame and print the markdown table
    df = pd.DataFrame(aggregated_results)
    display_cols = ["Model", "Runs", "ACCURACY", "LCR", "F1", "ANLS", "FPR", "FNR"]
    df_display = df[display_cols]
    
    print("\n### Multi-Run Aggregated Results (95% CI)\n")
    print(df_display.to_markdown(index=False))
    
    # Save the aggregated summary
    out_dir = f"results/{args.mode}/metrics"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "aggregated_multi_run_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"\nAggregated metrics saved to: {out_path}")

if __name__ == "__main__":
    main()
