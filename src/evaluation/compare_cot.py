import sys, os, csv, pandas as pd
from parse_results import extract_verdict
import matplotlib.pyplot as plt

def get_metrics(filepath):
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()
    lines = raw.splitlines()
    header = [h.strip() for h in next(csv.reader([lines[0]]))]
    n = len(header)
    
    import re
    row_start = re.compile(r'^\s*\d+\s*,')
    logical_rows, current_parts = [], []
    for line in lines[1:]:
        if row_start.match(line) and current_parts:
            logical_rows.append(' '.join(current_parts))
            current_parts = [line]
        else:
            current_parts.append(line)
    if current_parts:
        logical_rows.append(' '.join(current_parts))
        
    rows = []
    for logical in logical_rows:
        try: parsed = next(csv.reader([logical]))
        except StopIteration: continue
        if len(parsed) < n: parsed += [''] * (n - len(parsed))
        elif len(parsed) > n: parsed = parsed[:n-1] + [','.join(parsed[n-1:])]
        rows.append(parsed)

    df = pd.DataFrame(rows, columns=header)
    df['model_verdict'] = df['model_response'].apply(extract_verdict)
    for col in ['model_verdict', 'expected_verdict']:
        df[col] = df[col].astype(str).str.strip()

    counts = df['model_verdict'].value_counts().to_dict()
    scored = df[df['model_verdict'] != 'NO_VERDICT']
    
    if len(df) == 0:
        return None
        
    correct = (scored['model_verdict'] == scored['expected_verdict']).sum()
    acc = correct / len(df)
    
    safe_ct = counts.get('SAFE', 0)
    unsafe_ct = counts.get('UNSAFE', 0)
    noverdict_ct = counts.get('NO_VERDICT', 0)
    
    return {
        'Accuracy': acc,
        'SAFE': safe_ct,
        'UNSAFE': unsafe_ct,
        'NO_VERDICT': noverdict_ct,
        'Total': len(df)
    }

models = {
    'SmolVLM (0.6B)': ('results/baseline/smolvlm_results.csv', 'results/innovation/cot/smolvlm_cot_results.csv'),
    'InternVL2 (0.9B)': ('results/baseline/internvl2_results.csv', 'results/innovation/cot/internvl2_cot_results.csv'),
    'Janus (1.3B)': ('results/baseline/janus_results.csv', 'results/innovation/cot/janus_cot_results.csv'),
    'Qwen2-VL (2.2B)': ('results/baseline/qwen2_vl_results.csv', 'results/innovation/cot/qwen2_vl_cot_results.csv'),
    'MiniCPM (2.8B)': ('results/baseline/minicpm_results.csv', 'results/innovation/cot/minicpm_cot_results.csv')
}

summary = []

for model_name, (base_path, cot_path) in models.items():
    base_m = get_metrics(base_path)
    cot_m = get_metrics(cot_path)
    
    if not base_m or not cot_m:
        print(f"Skipping {model_name} - missing files.")
        continue
        
    acc_diff = cot_m['Accuracy'] - base_m['Accuracy']
    
    summary.append({
        'Model': model_name,
        'Baseline Acc': f"{base_m['Accuracy']:.3f}",
        'CoT Acc': f"{cot_m['Accuracy']:.3f}",
        'Acc Δ': f"{acc_diff:+.3f}",
        'Base Verdicts (S/U/NV)': f"{base_m['SAFE']}/{base_m['UNSAFE']}/{base_m['NO_VERDICT']}",
        'CoT Verdicts (S/U/NV)': f"{cot_m['SAFE']}/{cot_m['UNSAFE']}/{cot_m['NO_VERDICT']}"
    })

print("\n--- Baseline vs. CoT Comparison ---\n")
df_summary = pd.DataFrame(summary)
print(df_summary.to_markdown(index=False))

os.makedirs('results/innovation/cot', exist_ok=True)
df_summary.to_csv('results/innovation/cot/cot_vs_baseline_summary.csv', index=False)
print("\nSaved summary to results/innovation/cot/cot_vs_baseline_summary.csv")
