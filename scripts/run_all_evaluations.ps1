$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
# Assume $root is '.../Sub-3B-VLM-comparision-for-industrial-auditing/scripts', so project root is one level up:
$projectRoot = Split-Path -Parent $root

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   Starting Full VLM Evaluation Pipeline  " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

$envs = @{
    "four_models" = "$projectRoot\four_models\Scripts\python.exe"
    "minicpm" = "$projectRoot\minicpm\Scripts\python.exe"
}

# Ensure envs exist
foreach ($env in $envs.Keys) {
    if (-Not (Test-Path $envs[$env])) {
        Write-Host "WARNING: Python executable for $env not found at $($envs[$env]). Scripts using this environment might fail if path is incorrect." -ForegroundColor Yellow
    }
}

# The scripts are categorized by methodology
$methodologies = @(
    "generation_baseline",
    "generation_cot",
    "generation_decomposition",
    "generation_contrast",
    "generation_contrast_cot"
)

# Models and their respective environments and script suffixes
$models = @(
    @{ name="Qwen2-VL"; script_suffix="qwen2_vl"; env="four_models" },
    @{ name="Janus"; script_suffix="janus"; env="four_models" },
    @{ name="InternVL2"; script_suffix="internvl2"; env="four_models" },
    @{ name="SmolVLM"; script_suffix="smolvlm"; env="four_models" },
    @{ name="MiniCPM"; script_suffix="minicpm"; env="minicpm" }
)

$totalScripts = $methodologies.Count * $models.Count
$currentScript = 1

foreach ($methodology in $methodologies) {
    Write-Host "---------------------------------------------" -ForegroundColor Green
    Write-Host "   Running Methodology: $methodology" -ForegroundColor Green
    Write-Host "---------------------------------------------" -ForegroundColor Green
    
    foreach ($model in $models) {
        $timestamp = Get-Date -Format 'HH:mm:ss'
        Write-Host "[$currentScript/$totalScripts] Starting $($model.name) -> $methodology @ $timestamp" -ForegroundColor Yellow
        
        $scriptName = "run_$($model.script_suffix)"
        
        # Determine actual script name depending on methodology suffix
        if ($methodology -eq "generation_baseline") {
            $scriptName = $scriptName + ".py"
        } elseif ($methodology -eq "generation_cot") {
            $scriptName = $scriptName + "_cot.py"
        } elseif ($methodology -eq "generation_decomposition") {
            $scriptName = $scriptName + "_decomp.py"
        } elseif ($methodology -eq "generation_contrast") {
            $scriptName = $scriptName + "_contrast.py"
        } elseif ($methodology -eq "generation_contrast_cot") {
            $scriptName = $scriptName + "_contrast_cot.py"
        }

        $scriptPath = "$projectRoot\src\$methodology\$scriptName"
        $pythonExe = $envs[$model.env]
        
        if (Test-Path $scriptPath) {
            Write-Host "Executing: $pythonExe $scriptPath" -ForegroundColor DarkGray
            # Run the script
            try {
                & $pythonExe $scriptPath
                $exitCode = $LASTEXITCODE
                if ($exitCode -eq 0) {
                    Write-Host "[$currentScript/$totalScripts] Completed $($model.name) -> $methodology successfully.`n" -ForegroundColor Green
                } else {
                    Write-Host "[$currentScript/$totalScripts] Completed $($model.name) -> $methodology with EXIT CODE $exitCode.`n" -ForegroundColor Red
                }
            } catch {
                Write-Host "[$currentScript/$totalScripts] ERROR executing $($model.name). Exception: $_`n" -ForegroundColor Red
            }
        } else {
            Write-Host "[$currentScript/$totalScripts] SKIPPED $($model.name) -> $methodology (Script not found: $scriptPath)`n" -ForegroundColor Yellow
        }
        
        $currentScript++
    }
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   Full Evaluation Pipeline Finished!" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
