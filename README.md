# Sub-3B VLM Comparison for Industrial Auditing

A modular research pipeline for evaluating Vision-Language Models (VLMs) under strict hardware constraints (8GB VRAM ceiling; peak observed usage 4.6GB). This project compares six sub-3B parameter models on industrial safety auditing tasks involving analog gauges and pipeline integrity assessment.

## Key Findings

- **All six models fail zero-shot.** Every architecture defaulted to conversational safety priors over quantitative SOP logic. Qwen2-VL produced an 81.0% False Negative Rate at baseline; Gemma-4-E2B-it reached 93.3%. In industrial auditing, a false negative is a missed defect.
- **Rule Decomposition is the most effective intervention.** Breaking safety rules into sequential binary checks more than doubled Qwen2-VL Logic Compliance Rate from 19.0% to 40.0% (McNemar exact test, p < 0.0001) and cut its False Negative Rate to 47.6%.
- **Chain-of-Thought breaks the smallest models.** Step-by-step formatting collapsed SmolVLM entirely (100% FNR) and degraded Janus from 18.0% to 15.0% LCR, while rescuing Qwen2-VL. We term this the Formatting Penalty.
- **Agentic Foveation is not a general fix.** Our spatial-cropping pipeline significantly improved the CoT pathway (23.0% to 33.0% LCR, p = 0.0154) but regressed under Rule Decomposition. Removing peripheral context induced Tunnel Vision, spiking the False Positive Rate from 17.9% to 31.6% and dragging LCR from 40.0% down to 35.0%.
- **LoRA fine-tuning made things worse.** Adapting the vision encoder and projector depressed baseline LCR from 19.0% to 16.0%, and crashed the Decomposition peak from 40.0% to 14.0% (p = 0.0163). We term this Optimization Saturation: the sub-3B reasoning bottleneck is linguistic, not visual.

**Takeaway:** reliable edge auditing comes from restructuring the Standard Operating Procedure into constrained sequential logic, not from scaling parameters, cropping images, or fine-tuning weights.

## Tech Stack

- **Frameworks**: PyTorch, Transformers
- **Quantization**: Bitsandbytes (4-bit NF4 for >2B models)
- **Models**:
  * SmolVLM-500M (bfloat16)
  * InternVL2-1B (bfloat16)
  * Janus-Pro-1B (4-bit NF4)
  * Qwen2-VL-2B (bfloat16)
  * MiniCPM-V-2 (2.8B, 4-bit NF4)
  * Gemma-4-E2B-it (2B, 4-bit NF4)

## Hardware Requirements

- **VRAM**: 8GB ceiling. Peak observed consumption 4.6GB (Qwen2-VL) and 5.0GB (InternVL2) under 4-bit quantization. Tested on T4/L4/RTX Mobile.
- **Storage**: ~20GB for model weights and environment.
- **CUDA**: 12.4+ recommended.

## Dataset: The Golden 100

The `Dataset/` directory contains a manually curated benchmark:

- **50 analog gauge images** with visual stressors (glare, oblique angles, low resolution, obstructions)
- **50 pipeline images** (25 corroded, 25 non-corroded) with texture overlap challenges
- **200 evaluation rows**: each image is paired with two mutually exclusive Standard Operating Procedures. Rule A treats any surface oxidation as an alert condition; Rule B treats surface rust as within tolerance. True logic compliance requires the model to reverse its verdict on the identical image tensor when only the text rule changes.
- Full metadata in `Dataset/Metadata.Rmd`

Human annotators selected every image and supplied ground-truth readings. No automated scraping was used. The dataset and preprocessed images are tracked in Git, so no additional ingestion step is required.

## Methodology

- **Precision**: bfloat16 for models under 1.5B, 4-bit NF4 for larger models
- **Resolution**: standardized to model-native input resolution (384px or 448px)
- **Decoding**: greedy search with repetition penalty 1.1
- **Metrics**: ANLS (reading accuracy), LCR (logic compliance), F1, Accuracy, FNR/FPR
- **Statistical testing**: multi-run (N=3) deterministic validation using McNemar exact paired test with multiple-hypothesis correction
- **Architectural analysis**: safety-critical tradeoffs (FPR vs FNR) mapped to architectural families (Attention Overshadowing, Modality Collapse)

## Project Structure

```
├── src/
│   ├── ingestion/               # Data loading and preprocessing
│   ├── generation_baseline/     # Baseline zero-shot inference scripts
│   ├── generation_cot/          # Chain-of-Thought prompting scripts
│   ├── generation_decomposition/# Rule Decomposition prompting scripts
│   ├── generation_contrast/     # CLAHE + Decomposition inference scripts
│   ├── generation_contrast_cot/ # CLAHE + CoT inference scripts
│   ├── generation_profiling/    # Hardware profiling scripts (VRAM, throughput)
│   ├── evaluation/              # Metrics, parsing, and failure analysis
│   └── analysis/                # Additional analysis utilities
├── Dataset/                     # 100 industrial images + metadata (Golden 100)
├── Data_Preprocessed/           # CLAHE-enhanced images
├── Dataset_FineTune/            # Augmented samples for the LoRA intervention
├── models/                      # Local model weights (Git Ignored)
├── Janus/                       # Cloned Janus source code (Git Ignored)
├── paper/                       # LaTeX source, figures, compiled PDF, results table
├── notebooks/                   # Exploratory and fine-tuning notebooks
├── results/
│   ├── baseline/                # Baseline outputs, metrics, failure analysis
│   ├── innovation/              # CoT, Decomposition, Contrast, Contrast+CoT results
│   └── profiling/               # Hardware profiling summaries
├── run_all.ps1                  # One-command full pipeline reproduction
└── run_all_profiling.ps1        # Hardware profiling only
```

## Quick Start

### 1. Environment Setup

This project requires separate virtual environments due to model-specific dependency constraints.

#### General VLM Environment (four_models)

Used for SmolVLM, InternVL2, Janus, and Qwen2-VL.

```
python -m venv four_models
four_models\Scripts\activate     # Windows
pip install -r requirements.txt
```

#### MiniCPM Environment (minicpm)

Used specifically for MiniCPM-V-2.

```
python -m venv minicpm
minicpm\Scripts\activate         # Windows
pip install -r requirements_minicpm.txt
```

#### Gemma Environment

Gemma-4-E2B-it has its own dependency set (see `requirements_gemma.txt`).

### 2. Model & Repository Preparation

Download all model weights and clone the Janus architecture:

```
four_models\Scripts\python.exe src\generation_baseline\download_models.py
```
> **Note:** This script uses `snapshot_download` for reliability and will automatically `git clone` the required Janus architecture if it is missing.

### 3. Full Pipeline Reproduction (One Command)

```
.\run_all.ps1
```

This runs all 7 phases sequentially:

1. **Baseline inference** (zero-shot)
2. **Chain-of-Thought inference**
3. **Rule Decomposition inference**
4. **CLAHE + Decomposition inference**
5. **CLAHE + CoT inference**
6. **Hardware profiling** (VRAM and throughput)
7. **Evaluation** (parsing, metrics, failure analysis)

Estimated runtime: ~4-6 hours on a single GPU (T4/L4/RTX class).

### 4. Running Individual Phases

```
# Baseline
four_models\Scripts\python.exe src\generation_baseline\run_smolvlm.py
four_models\Scripts\python.exe src\generation_baseline\run_internvl2.py
four_models\Scripts\python.exe src\generation_baseline\run_qwen2_vl.py
four_models\Scripts\python.exe src\generation_baseline\run_janus.py
minicpm\Scripts\python.exe src\generation_baseline\run_minicpm.py

# Chain-of-Thought
four_models\Scripts\python.exe src\generation_cot\run_smolvlm_cot.py
# ... (same pattern for all models)

# Evaluation only (after inference is complete)
four_models\Scripts\python.exe src\evaluation\parse_results.py
four_models\Scripts\python.exe src\evaluation\metrics.py
four_models\Scripts\python.exe src\evaluation\failure_analysis.py
four_models\Scripts\python.exe src\evaluation\multi_run_metrics.py
four_models\Scripts\python.exe src\evaluation\statistical_tests.py --intervention decomp
```

### 5. Results & Outputs

All outputs are saved to `results/`:

- **Baseline**: `results/baseline/` (raw outputs, parsed results, metrics, failure analysis)
- **Innovation phases**: `results/innovation/{cot,decomposition,contrast,contrast_cot}/`
- **Profiling**: `results/profiling/hardware_summary.csv`
- **Metrics**: each phase directory contains `metrics/metrics_summary.csv` and `aggregated_multi_run_metrics.csv`
- **Statistical significance**: McNemar p-values in `results/metrics/mcnemar_{intervention}_significance.csv`

## Paper

LaTeX source and the compiled PDF are in `paper/`. Full per-model metrics across all five interventions are in `paper/results_table.tex`.

## License

Refer to the individual model cards or official repositories for specific licensing information.
