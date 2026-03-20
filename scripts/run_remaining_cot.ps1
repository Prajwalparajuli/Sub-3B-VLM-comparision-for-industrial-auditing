$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   CoT Runs - Remaining 4 Models (N=3 each)" -ForegroundColor Cyan
Write-Host "   SmolVLM already done, skipping." -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# --- InternVL2 (four_models env) ---
Write-Host "--- [1/4] InternVL2 CoT  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_cot\run_internvl2_cot.py"
Write-Host "--- InternVL2 CoT done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- Janus (four_models env) ---
Write-Host "--- [2/4] Janus CoT  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_cot\run_janus_cot.py"
Write-Host "--- Janus CoT done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- Qwen2-VL (four_models env) ---
Write-Host "--- [3/4] Qwen2-VL CoT  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\four_models\Scripts\python.exe" "$root\src\generation_cot\run_qwen2_vl_cot.py"
Write-Host "--- Qwen2-VL CoT done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

# --- MiniCPM (minicpm env) ---
Write-Host "--- [4/4] MiniCPM CoT  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
& "$root\minicpm\Scripts\python.exe" "$root\src\generation_cot\run_minicpm_cot.py"
Write-Host "--- MiniCPM CoT done: exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss') ---" -ForegroundColor Yellow
Write-Host ""

Write-Host "=============================================" -ForegroundColor Green
Write-Host "   ALL 4 CoT MODELS DONE!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Results in: results/innovation/cot/"
