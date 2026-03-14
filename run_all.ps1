# =============================================================================
#  Full Reproducibility Script
#  Runs all experimental phases end-to-end: Baseline, CoT, Decomposition,
#  Contrast+Decomp, Contrast+CoT, Profiling, Evaluation, and Failure Analysis.
#
#  Prerequisites:
#    1. Two virtual environments created (see README):
#       - four_models\  (SmolVLM, InternVL2, Janus, Qwen2-VL)
#       - minicpm\      (MiniCPM-V-2)
#    2. Model weights downloaded: python src/generation_baseline/download_models.py
#    3. CUDA-capable GPU with at least 6GB VRAM
#
#  Usage:  .\run_all.ps1
#  Estimated runtime: ~4-6 hours on a single GPU (T4/L4/RTX class)
# =============================================================================

$fourModels = "four_models\Scripts\python.exe"
$miniCPM    = "minicpm\Scripts\python.exe"

# --- Phase 1: Baseline Inference ---
Write-Host "`n============================================"
Write-Host "  PHASE 1: Baseline Inference (5 models)"
Write-Host "============================================"

Write-Host "`n---> [1/5] SmolVLM baseline..."
& $fourModels src\generation_baseline\run_smolvlm.py

Write-Host "`n---> [2/5] InternVL2 baseline..."
& $fourModels src\generation_baseline\run_internvl2.py

Write-Host "`n---> [3/5] Qwen2-VL baseline..."
& $fourModels src\generation_baseline\run_qwen2_vl.py

Write-Host "`n---> [4/5] Janus baseline..."
& $fourModels src\generation_baseline\run_janus.py

Write-Host "`n---> [5/5] MiniCPM baseline..."
& $miniCPM src\generation_baseline\run_minicpm.py

# --- Phase 2: Chain-of-Thought Inference ---
Write-Host "`n============================================"
Write-Host "  PHASE 2: Chain-of-Thought Inference"
Write-Host "============================================"

Write-Host "`n---> [1/5] SmolVLM CoT..."
& $fourModels src\generation_cot\run_smolvlm_cot.py

Write-Host "`n---> [2/5] InternVL2 CoT..."
& $fourModels src\generation_cot\run_internvl2_cot.py

Write-Host "`n---> [3/5] Qwen2-VL CoT..."
& $fourModels src\generation_cot\run_qwen2_vl_cot.py

Write-Host "`n---> [4/5] Janus CoT..."
& $fourModels src\generation_cot\run_janus_cot.py

Write-Host "`n---> [5/5] MiniCPM CoT..."
& $miniCPM src\generation_cot\run_minicpm_cot.py

# --- Phase 3: Rule Decomposition Inference ---
Write-Host "`n============================================"
Write-Host "  PHASE 3: Rule Decomposition Inference"
Write-Host "============================================"

Write-Host "`n---> [1/5] SmolVLM Decomposition..."
& $fourModels src\generation_decomposition\run_smolvlm_decomp.py

Write-Host "`n---> [2/5] InternVL2 Decomposition..."
& $fourModels src\generation_decomposition\run_internvl2_decomp.py

Write-Host "`n---> [3/5] Qwen2-VL Decomposition..."
& $fourModels src\generation_decomposition\run_qwen2_vl_decomp.py

Write-Host "`n---> [4/5] Janus Decomposition..."
& $fourModels src\generation_decomposition\run_janus_decomp.py

Write-Host "`n---> [5/5] MiniCPM Decomposition..."
& $miniCPM src\generation_decomposition\run_minicpm_decomp.py

# --- Phase 4: CLAHE + Decomposition Inference ---
Write-Host "`n============================================"
Write-Host "  PHASE 4: CLAHE + Decomposition Inference"
Write-Host "============================================"

Write-Host "`n---> [1/5] SmolVLM Contrast..."
& $fourModels src\generation_contrast\run_smolvlm_contrast.py

Write-Host "`n---> [2/5] InternVL2 Contrast..."
& $fourModels src\generation_contrast\run_internvl2_contrast.py

Write-Host "`n---> [3/5] Qwen2-VL Contrast..."
& $fourModels src\generation_contrast\run_qwen2_vl_contrast.py

Write-Host "`n---> [4/5] Janus Contrast..."
& $fourModels src\generation_contrast\run_janus_contrast.py

Write-Host "`n---> [5/5] MiniCPM Contrast..."
& $miniCPM src\generation_contrast\run_minicpm_contrast.py

# --- Phase 5: CLAHE + CoT Inference ---
Write-Host "`n============================================"
Write-Host "  PHASE 5: CLAHE + CoT Inference"
Write-Host "============================================"

Write-Host "`n---> [1/5] SmolVLM Contrast+CoT..."
& $fourModels src\generation_contrast_cot\run_smolvlm_contrast_cot.py

Write-Host "`n---> [2/5] InternVL2 Contrast+CoT..."
& $fourModels src\generation_contrast_cot\run_internvl2_contrast_cot.py

Write-Host "`n---> [3/5] Qwen2-VL Contrast+CoT..."
& $fourModels src\generation_contrast_cot\run_qwen2_vl_contrast_cot.py

Write-Host "`n---> [4/5] Janus Contrast+CoT..."
& $fourModels src\generation_contrast_cot\run_janus_contrast_cot.py

Write-Host "`n---> [5/5] MiniCPM Contrast+CoT..."
& $miniCPM src\generation_contrast_cot\run_minicpm_contrast_cot.py

# --- Phase 6: Hardware Profiling ---
Write-Host "`n============================================"
Write-Host "  PHASE 6: Hardware Profiling"
Write-Host "============================================"

Write-Host "`n---> [1/5] Profiling SmolVLM..."
& $fourModels src\generation_profiling\run_smolvlm_profile.py

Write-Host "`n---> [2/5] Profiling InternVL2..."
& $fourModels src\generation_profiling\run_internvl2_profile.py

Write-Host "`n---> [3/5] Profiling Qwen2-VL..."
& $fourModels src\generation_profiling\run_qwen2_vl_profile.py

Write-Host "`n---> [4/5] Profiling Janus..."
& $fourModels src\generation_profiling\run_janus_profile.py

Write-Host "`n---> [5/5] Profiling MiniCPM..."
& $miniCPM src\generation_profiling\run_minicpm_profile.py

# --- Phase 7: Evaluation & Metrics ---
Write-Host "`n============================================"
Write-Host "  PHASE 7: Evaluation & Metrics"
Write-Host "============================================"

Write-Host "`n---> Parsing all results..."
& $fourModels src\evaluation\parse_results.py

Write-Host "`n---> Computing metrics (ANLS, LCR, F1, Accuracy)..."
& $fourModels src\evaluation\metrics.py

Write-Host "`n---> Running failure analysis..."
& $fourModels src\evaluation\failure_analysis.py

Write-Host "`n---> Comparing CoT results..."
& $fourModels src\evaluation\compare_cot.py

Write-Host "`n---> Aggregating hardware profiles..."
& $fourModels src\evaluation\aggregate_profiles.py

Write-Host "`n============================================"
Write-Host "  ALL PHASES COMPLETE"
Write-Host "  Results saved to: results/"
Write-Host "============================================"
