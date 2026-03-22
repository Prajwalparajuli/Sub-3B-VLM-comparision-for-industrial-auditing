# Sub-3B VLM Comparison for Industrial Auditing

    QR Link to this GitHUb Repository:
![QR Code](https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=https://github.com/Prajwalparajuli/Sub-3B-VLM-comparision-for-industrial-auditing)

A modular research pipeline for evaluating Vision-Language Models (VLMs) under strict hardware constraints (6GB VRAM). This project compares five sub-3B parameter models on industrial safety auditing tasks involving analog gauges and pipeline integrity assessment.

## Tech Stack
- **Frameworks**: PyTorch, Transformers
- **Quantization**: Bitsandbytes (4-bit NF4 for >2B models)
- **Models**:
  - SmolVLM-500M (bfloat16)
  - InternVL2-1B (bfloat16)
  - Janus-Pro-1B (4-bit NF4)
  - Qwen2-VL-2B (bfloat16)
  - MiniCPM-V-2 (2.8B, 4-bit NF4)

## Hardware Requirements
- **VRAM**: Minimum 6GB (Tested on T4/L4/RTX Mobile).
- **Storage**: ~20GB for model weights and environment.
- **CUDA**: 12.4+ recommended.

## Project Structure
```text
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
├── models/                      # Local model weights (Git Ignored)
├── Janus/                       # Cloned Janus source code (Git Ignored)
├── results/
│   ├── baseline/                # Baseline inference outputs, metrics, failure analysis
│   ├── innovation/              # CoT, Decomposition, Contrast, Contrast+CoT results
│   └── profiling/               # Hardware profiling summaries
├── run_all.ps1                  # One-command full pipeline reproduction
└── run_all_profiling.ps1        # Hardware profiling only
```

## Quick Start

### 1. Environment Setup
This project requires two separate virtual environments due to model-specific dependency constraints.

#### General VLM Environment (four_models)
Used for SmolVLM, InternVL2, Janus, and Qwen2-VL.
```bash
python -m venv four_models
four_models\Scripts\activate     # Windows
pip install -r requirements.txt
```

#### MiniCPM Environment (minicpm)
Used specifically for MiniCPM-V-2.
```bash
python -m venv minicpm
minicpm\Scripts\activate         # Windows
pip install -r requirements_minicpm.txt
```

### 2. Model & Repository Preparation
Download all model weights and clone the Janus architecture:
```bash
four_models\Scripts\python.exe src\generation_baseline\download_models.py
```
> **Note:** This script uses `snapshot_download` for reliability and will automatically `git clone` the required Janus architecture if it is missing.

### 3. Full Pipeline Reproduction (One Command)
To reproduce all experimental phases end-to-end:
```powershell
.\run_all.ps1
```
This script runs all 7 phases sequentially:
1. **Baseline inference** (5 models, zero-shot)
2. **Chain-of-Thought inference** (5 models)
3. **Rule Decomposition inference** (5 models)
4. **CLAHE + Decomposition inference** (5 models)
5. **CLAHE + CoT inference** (5 models)
6. **Hardware profiling** (VRAM and throughput measurement)
7. **Evaluation** (parsing, metrics computation, failure analysis)

Estimated runtime: ~4-6 hours on a single GPU (T4/L4/RTX class).

### 4. Running Individual Phases
To run a single model or phase independently:
```bash
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
```

### 5. Results & Outputs
All outputs are saved to `results/`:
- **Baseline**: `results/baseline/` (raw outputs, parsed results, metrics, failure analysis)
- **Innovation phases**: `results/innovation/{cot,decomposition,contrast,contrast_cot}/`
- **Profiling**: `results/profiling/hardware_summary.csv`
- **Metrics**: Each phase directory contains a `metrics/metrics_summary.csv`

## Dataset Information
The `Dataset/` directory contains the manually curated "Golden 100" benchmark:
- **50 analog gauge images** with visual stressors (glare, oblique angles, low resolution, obstructions)
- **50 pipeline images** (25 corroded, 25 non-corroded) with texture overlap challenges
- **200 evaluation rows** (each image tested under two opposing SOP rules)
- Full metadata documentation in `Dataset/Metadata.Rmd`

No additional data ingestion is needed. The dataset and preprocessed images are tracked in Git.

## Methodology
- **Precision**: bfloat16 for models under 1.5B, 4-bit NF4 for larger models
- **Resolution**: Standardized to model-native input resolution (384px or 448px)
- **Decoding**: Greedy search with repetition penalty 1.1
- **Evaluation**: ANLS (reading accuracy), LCR (logic compliance), F1, Accuracy

                            
## License
Refer to the individual model cards or official repositories for specific licensing information.
