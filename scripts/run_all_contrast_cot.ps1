$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   Contrast+CoT Runs - All 5 Models (N=3)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# --- SmolVLM (four_models env) ---
Write-Host "--- [1/5] SmolVLM Contrast+CoT  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_contrast_cot\run_smolvlm_contrast_cot.py"
Write-Host "--- SmolVLM done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- InternVL2 (four_models env) ---
Write-Host "--- [2/5] InternVL2 Contrast+CoT  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_contrast_cot\run_internvl2_contrast_cot.py"
Write-Host "--- InternVL2 done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- Janus (four_models env) ---
Write-Host "--- [3/5] Janus Contrast+CoT  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_contrast_cot\run_janus_contrast_cot.py"
Write-Host "--- Janus done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- Qwen2-VL (four_models env) ---
Write-Host "--- [4/5] Qwen2-VL Contrast+CoT  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_contrast_cot\run_qwen2_vl_contrast_cot.py"
Write-Host "--- Qwen2-VL done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- MiniCPM (minicpm env) ---
Write-Host "--- [5/5] MiniCPM Contrast+CoT  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\minicpm\Scripts\python.exe" "$root\src\generation_contrast_cot\run_minicpm_contrast_cot.py"
Write-Host "--- MiniCPM done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

Write-Host "=============================================" -ForegroundColor Green
Write-Host "   ALL 5 CONTRAST+COT MODELS DONE!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Results in: results/innovation/contrast_cot/"
