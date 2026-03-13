Write-Host "=========================================="
Write-Host "  Starting Hardware Profiling Benchmark   "
Write-Host "=========================================="

Write-Host "`n---> [1/5] Profiling SmolVLM..."
& four_models\Scripts\python.exe src\generation_profiling\run_smolvlm_profile.py

Write-Host "`n---> [2/5] Profiling InternVL2..."
& four_models\Scripts\python.exe src\generation_profiling\run_internvl2_profile.py

Write-Host "`n---> [3/5] Profiling Qwen2-VL..."
& four_models\Scripts\python.exe src\generation_profiling\run_qwen2_vl_profile.py

Write-Host "`n---> [4/5] Profiling Janus..."
& four_models\Scripts\python.exe src\generation_profiling\run_janus_profile.py

Write-Host "`n---> [5/5] Profiling MiniCPM..."
& minicpm\Scripts\python.exe src\generation_profiling\run_minicpm_profile.py

Write-Host "`n=========================================="
Write-Host "  Hardware Profiling Complete!"
Write-Host "=========================================="
