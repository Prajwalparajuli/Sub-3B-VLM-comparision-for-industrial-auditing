# Sub-3B VLM Comparison for Industrial Auditing

A modular research pipeline designed to evaluate Vision-Language Models (VLMs) under strict hardware constraints (specifically 4GB VRAM). This project compares several "Sub-3B" models for objective, researcher-led industrial component description.

## Tech Stack
- **Frameworks**: PyTorch, Transformers
- **Quantization**: Bitsandbytes (4-bit for >2B models)
- **Models**:
  - SmolVLM-500M
  - InternVL2-1B
  - Janus-Pro-1B
  - Qwen2-VL-2B
  - MiniCPM-V-2 (2.8B)

## Hardware Requirements
- **VRAM**: Minimum 4GB (Tested on T4/L4/RTX Mobile).
- **Storage**: ~20GB for model weights and environment.
- **CUDA**: 12.4+ recommended.

## Project Structure
```text
├── src/
│   ├── ingestion/         # Data loading and preprocessing
│   └── generation_baseline/ # Model download and inference scripts
├── Dataset/               # Input images and metadata (Golden 100)
├── models/                # Local model weights (Git Ignored)
├── Janus/                # Cloned Janus source code (Git Ignored)
└── results/               # Inference outputs (Git Ignored)
```

## Quick Start

### 1. Environment Setup
This project requires two separate virtual environments due to model-specific dependency constraints.

#### General VLM Environment (ragenv)
Used for SmolVLM, InternVL2, Janus, and Qwen2-VL.
```bash
python -m venv ragenv
source ragenv/bin/activate  # Or ragenv\Scripts\activate on Windows
pip install -r requirements.txt
```

#### MiniCPM Environment (minicpm_env)
Used specifically for MiniCPM-V-2.
```bash
python -m venv minicpm_env
source minicpm_env/bin/activate  # Or minicpm_env\Scripts\activate on Windows
pip install -r requirements_minicpm.txt
```

### 2. Model & Repository Preparation
To acquire all necessary model weights and architecture code (specifically for Janus), use the **General VLM Environment**:
```bash
source ragenv/bin/activate
python src/generation_baseline/download_models.py
```
> [!NOTE]
> This script uses `snapshot_download` for reliability and will automatically `git clone` the required Janus architecture if it is missing.

### 3. Data Ingestion
Verify the data loader and ingestion pipeline:
```bash
python src/ingestion/ingestion_utils.py
```

### 4. Running Benchmarks
Run inference for individual models:
```bash
python src/generation_baseline/run_smolvlm.py
python src/generation_baseline/run_janus.py
# ... etc
```

### 5. Results & Outputs
All inference results are saved to the `results/` directory as CSV files. 
- **Format**: `[model_name]_results.csv`
- **Aggregation**: (Planned) A final script will aggregate these into a master comparison report.

## Dataset Information
The `Dataset/` and `Data_Preprocessed/` folders are pre-populated and tracked in Git. 
- **Golden 100**: 100 industrial audit images with associated text constraints.
- **No Ingestion Needed**: You do **not** need to run `preprocessing.py` or ingest scripts unless adding new data.

## Methodology
- **ENSR**: Standardized resolution (384px or 448px).
- **BAP**: Balanced Precision (bfloat16 for <1.5B, 4-bit for >2B).
- **SGP**: Standardized Generation Protocol (Repetition Penalty 1.1, Greedy search).

## License
Refer to the individual model sheets or official repositories for specific licensing information.
