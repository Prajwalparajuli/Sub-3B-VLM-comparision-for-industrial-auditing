$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   Decomposition Runs - All 5 Models (N=3)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# --- SmolVLM (four_models env) ---
Write-Host "--- [1/5] SmolVLM Decomp  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_decomposition\run_smolvlm_decomp.py"
Write-Host "--- SmolVLM Decomp done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- InternVL2 (four_models env) ---
Write-Host "--- [2/5] InternVL2 Decomp  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_decomposition\run_internvl2_decomp.py"
Write-Host "--- InternVL2 Decomp done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- Janus (four_models env) ---
Write-Host "--- [3/5] Janus Decomp  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_decomposition\run_janus_decomp.py"
Write-Host "--- Janus Decomp done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- Qwen2-VL (four_models env) ---
Write-Host "--- [4/5] Qwen2-VL Decomp  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_decomposition\run_qwen2_vl_decomp.py"
Write-Host "--- Qwen2-VL Decomp done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- MiniCPM (minicpm env) ---
Write-Host "--- [5/5] MiniCPM Decomp  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\minicpm\Scripts\python.exe" "$root\src\generation_decomposition\run_minicpm_decomp.py"
Write-Host "--- MiniCPM Decomp done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

Write-Host "=============================================" -ForegroundColor Green
Write-Host "   ALL 5 DECOMPOSITION MODELS DONE!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Results in: results/innovation/decomposition/"
