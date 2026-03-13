"""
parse_results.py

Reads raw model output CSVs from results/baseline/ and produces parsed CSVs
in results/parsed/ with two new columns added:

  model_verdict  -- SAFE | UNSAFE | NO_VERDICT
  model_reasoning -- the reasoning text extracted before the verdict line

Each model has a slightly different output format, so extraction is done
in priority order: most-specific pattern first, generic fallback last.
"""

import re
import os
import pandas as pd


# Verdict Extraction

# Ordered list of labeled verdict patterns to try first.
# Each model tends to use one of these consistently:
#   SmolVLM  -> "Final Verdict: SAFE"
#   Qwen2-VL -> "Verdict: SAFE" or "**Verdict:**\n- SAFE"
#   InternVL2-> "verdict: UNSAFE" (lowercase, at end of bullet)
#   Janus    -> "[SAFE]" or "[UNSAFE]" on its own line
#   MiniCPM  -> "[SAFE]" or "[UNSAFE]" inline in brackets
#
# UNSAFE must be checked before SAFE in every pattern because
# "UNSAFE" contains the substring "SAFE". A naive search for
# "SAFE" would match inside "UNSAFE".

LABELED_PATTERNS = [
    # "Turn 2 Verdict: SAFE" (Multi-turn conversational format)
    re.compile(r"Turn\s*2\s*Verdict\s*:\s*(UNSAFE|SAFE)", re.IGNORECASE),

    # "Q3: Final Verdict: UNSAFE" or "Q3: UNSAFE" (Rule Decomposition format Q1/Q2/Q3)
    re.compile(r"Q3:\s*(?:Final\s*Verdict\s*:\s*)?(UNSAFE|SAFE)", re.IGNORECASE),

    # "Final Verdict: UNSAFE" / "Final Verdict: SAFE" (SmolVLM baseline + CoT)
    re.compile(r"final\s+verdict\s*:\s*(UNSAFE|SAFE)", re.IGNORECASE),

    # "**Verdict:** UNSAFE" or "**Verdict:**\n- UNSAFE" (Qwen2-VL markdown)
    re.compile(r"\*\*verdict\*\*\s*:?\s*[-\s]*(UNSAFE|SAFE)", re.IGNORECASE),

    # "Verdict: UNSAFE" plain label (Qwen2-VL, InternVL2)
    re.compile(r"verdict\s*:\s*(UNSAFE|SAFE)", re.IGNORECASE),

    # "[UNSAFE]" or "[SAFE]" bracket notation (MiniCPM, Janus)
    re.compile(r"\[(UNSAFE|SAFE)\]", re.IGNORECASE),

    # "Conclusion: UNSAFE" or "conclusion is SAFE" (InternVL2 variants)
    re.compile(r"conclusion\s*(?:is)?\s*:?\s*(UNSAFE|SAFE)", re.IGNORECASE),

    # "STEP 3 - VERDICT: SAFE" -- CoT output format when the model
    # doesn't use the "Final Verdict:" phrasing we asked for.
    # Must come after the more specific patterns above.
    re.compile(r"step\s*3\s*[-\u2014]?\s*verdict\s*:?\s*(UNSAFE|SAFE)", re.IGNORECASE),
]


# Standalone word fallback -- last resort when no labeled pattern matched.
# \b is a word boundary, so this won't match "UNSAFE" when scanning for "SAFE".
# We take the LAST match in the text because the verdict is usually at the end.
STANDALONE_PATTERN = re.compile(r"\b(UNSAFE|SAFE)\b", re.IGNORECASE)


def extract_verdict(text):
    """
    Try each labeled pattern in order. If none match, fall back to the
    last standalone SAFE/UNSAFE word in the text.

    Returns "SAFE", "UNSAFE", or "NO_VERDICT" (all uppercase).
    """
    if not isinstance(text, str) or not text.strip():
        return "NO_VERDICT"

    for pattern in LABELED_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()

    # Fallback: scan the whole response and take the last hit
    all_matches = STANDALONE_PATTERN.findall(text)
    if all_matches:
        return all_matches[-1].upper()

    return "NO_VERDICT"


# Reasoning Extraction

# Patterns that typically introduce the verdict line.
# Everything before the first of these is considered the reasoning block.
VERDICT_LINE_STARTERS = re.compile(
    r"(final\s+verdict|verdict|\*\*verdict\*\*|conclusion|\[SAFE\]|\[UNSAFE\]|turn\s*2\s*verdict)",
    re.IGNORECASE
)


def extract_reasoning(text):
    """
    Return the text that appears before the verdict line.

    If the model wrote a structured response like:
      "Reasoning: ... Verdict: SAFE"
    this will return the reasoning portion only.

    If no verdict-line marker is found, return the full text -- it may
    still contain useful context even without a labeled verdict.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    match = VERDICT_LINE_STARTERS.search(text)
    if match:
        # Take everything before the verdict label, strip whitespace
        reasoning = text[:match.start()].strip()
        # Remove common prefixes like "Reasoning:" or "Audit Reasoning:"
        reasoning = re.sub(r"^(audit\s+)?reasoning\s*:\s*", "", reasoning, flags=re.IGNORECASE)
        return reasoning.strip()

    # No labeled verdict found -- return the full response as reasoning
    return text.strip()


# File Processing

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["baseline", "cot", "decomp", "multiturn", "contrast", "contrast_cot"], default="baseline")
args = parser.parse_args()

if args.mode == "cot":
    INPUT_DIR = "results/innovation/cot"
    OUTPUT_DIR = "results/innovation/cot/parsed"
    MODEL_FILES = [
        "smolvlm_cot_results.csv",
        "internvl2_cot_results.csv",
        "janus_cot_results.csv",
        "qwen2_vl_cot_results.csv",
        "minicpm_cot_results.csv",
    ]
elif args.mode == "decomp":
    INPUT_DIR = "results/innovation/decomposition"
    OUTPUT_DIR = "results/innovation/decomposition/parsed"
    MODEL_FILES = [
        "smolvlm_decomp_results.csv",
        "internvl2_decomp_results.csv",
        "janus_decomp_results.csv",
        "qwen2_vl_decomp_results.csv",
        "minicpm_decomp_results.csv",
    ]
elif args.mode == "multiturn":
    INPUT_DIR = "results/innovation/multiturn"
    OUTPUT_DIR = "results/innovation/multiturn/parsed"
    MODEL_FILES = [
        "smolvlm_multiturn_results.csv",
        "internvl2_multiturn_results.csv",
        "janus_multiturn_results.csv",
        "qwen2_vl_multiturn_results.csv",
        "minicpm_multiturn_results.csv",
    ]
elif args.mode == "contrast":
    INPUT_DIR = "results/innovation/contrast"
    OUTPUT_DIR = "results/innovation/contrast/parsed"
    MODEL_FILES = [
        "smolvlm_contrast_results.csv",
        "internvl2_contrast_results.csv",
        "janus_contrast_results.csv",
        "qwen2_vl_contrast_results.csv",
        "minicpm_contrast_results.csv",
    ]
elif args.mode == "contrast_cot":
    INPUT_DIR = "results/innovation/contrast_cot"
    OUTPUT_DIR = "results/innovation/contrast_cot/parsed"
    MODEL_FILES = [
        "smolvlm_contrast_cot_results.csv",
        "internvl2_contrast_cot_results.csv",
        "janus_contrast_cot_results.csv",
        "qwen2_vl_contrast_cot_results.csv",
        "minicpm_contrast_cot_results.csv",
    ]
else:
    INPUT_DIR = "results/baseline"
    OUTPUT_DIR = "results/baseline/parsed"
    MODEL_FILES = [
        "smolvlm_results.csv",
        "internvl2_results.csv",
        "janus_results.csv",
        "qwen2_vl_results.csv",
        "minicpm_results.csv",
    ]


def parse_model_file(input_path, output_path):
    """
    Load one model's raw result CSV, add model_verdict and model_reasoning
    columns, and save to the parsed output directory.
    """
    # The baseline CSVs have model_response cells with raw (unquoted) newlines.
    # Standard CSV parsers and even Python's csv module cannot handle this.
    # Strategy: each data row starts with a space-padded integer index followed
    # by a comma, e.g. "    1,".  We use that pattern as a row boundary sentinel
    # to reconstruct full rows before parsing.
    import csv as _csv

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    lines = raw.splitlines()
    header_row = next(_csv.reader([lines[0]]))
    header = [h.strip() for h in header_row]

    # regex: physical line that starts a new data row (leading spaces + int + comma)
    row_start = re.compile(r"^\s*\d+\s*,")

    # group physical lines into logical rows
    logical_rows = []
    current_parts = []
    for line in lines[1:]:
        if row_start.match(line) and current_parts:
            # flush the previous logical row — join continuation lines with space
            logical_rows.append(" ".join(current_parts))
            current_parts = [line]
        else:
            current_parts.append(line)
    if current_parts:
        logical_rows.append(" ".join(current_parts))

    rows = []
    n = len(header)
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

    df = pd.DataFrame(rows, columns=header)



    if "model_response" not in df.columns:
        print(f"WARNING: 'model_response' column not found in {input_path}, skipping.")
        return

    df["model_verdict"] = df["model_response"].apply(extract_verdict)
    df["model_reasoning"] = df["model_response"].apply(extract_reasoning)

    # Replace embedded newlines with a space before saving.
    # Multiline cells in pandas CSVs are valid but are difficult to read back
    # correctly across different parsers and environments.  Collapsing newlines
    # to a space keeps every row on a single physical line.
    for col in ["model_response", "model_reasoning", "logic_constraint"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\n|\r\n|\r", " ", regex=True)

    df.to_csv(output_path, index=False)

    print(f"Parsed {len(df)} rows -> {output_path}")

    # Quick summary so you can spot obvious parsing problems immediately
    counts = df["model_verdict"].value_counts().to_dict()
    print(f"  Verdict breakdown: {counts}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in MODEL_FILES:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace("_results", "_parsed"))

        if not os.path.exists(input_path):
            print(f"WARNING: File not found, skipping: {input_path}")
            continue

        print(f"\nProcessing: {filename}")
        parse_model_file(input_path, output_path)

    print("\nAll files parsed.")


if __name__ == "__main__":
    main()
