$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   Contrast+Decomp Runs - All 5 Models (N=3)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# --- SmolVLM (four_models env) ---
Write-Host "--- [1/5] SmolVLM Contrast  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_contrast\run_smolvlm_contrast.py"
Write-Host "--- SmolVLM done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- InternVL2 (four_models env) ---
Write-Host "--- [2/5] InternVL2 Contrast  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_contrast\run_internvl2_contrast.py"
Write-Host "--- InternVL2 done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- Janus (four_models env) ---
Write-Host "--- [3/5] Janus Contrast  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_contrast\run_janus_contrast.py"
Write-Host "--- Janus done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- Qwen2-VL (four_models env) ---
Write-Host "--- [4/5] Qwen2-VL Contrast  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_contrast\run_qwen2_vl_contrast.py"
Write-Host "--- Qwen2-VL done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- MiniCPM (minicpm env) ---
Write-Host "--- [5/5] MiniCPM Contrast  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\minicpm\Scripts\python.exe" "$root\src\generation_contrast\run_minicpm_contrast.py"
Write-Host "--- MiniCPM done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

Write-Host "=============================================" -ForegroundColor Green
Write-Host "   ALL 5 CONTRAST+DECOMP MODELS DONE!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Results in: results/innovation/contrast/"
