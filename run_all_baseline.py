import subprocess
import os
import sys

# Define the models, their respective scripts, and the required Python environment
models_to_run = [
    ("SmolVLM", "run_smolvlm.py", r"four_models\Scripts\python.exe"),
    ("InternVL2", "run_internvl2.py", r"four_models\Scripts\python.exe"),
    ("Janus", "run_janus.py", r"four_models\Scripts\python.exe"),
    ("Qwen2-VL", "run_qwen2_vl.py", r"four_models\Scripts\python.exe"),
    ("MiniCPM", "run_minicpm.py", r"minicpm\Scripts\python.exe")
]

script_dir = "src/generation_baseline"

print("==================================================")
print("Starting Multi-Run Baseline Evaluation Suite (N=3)")
print("==================================================")

for model_name, script_name, python_exe in models_to_run:
    script_path = os.path.join(script_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"\n[ERROR] Script not found for {model_name}: {script_path}")
        continue
    
    if not os.path.exists(python_exe):
        print(f"\n[ERROR] Environment not found for {model_name}: {python_exe}")
        continue
        
    print(f"\n=== Running Baseline Evaluation for: {model_name} ===")
    
    # Run the script and stream output
    try:
        # We use the specific python executable for the model's virtual environment
        process = subprocess.Popen(
            [python_exe, "-u", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        
        if process.returncode == 0:
            print(f"=== Successfully completed {model_name} ===")
        else:
            print(f"\n[ERROR] Process for {model_name} failed with return code {process.returncode}")
            # If a model fails, we might want to stop the whole suite or just report it.
            # For benchmarking, it's usually better to continue and log the failure.
            
    except Exception as e:
        print(f"\n[EXCEPTION] Failed to run {model_name}: {str(e)}")

print("\n==================================================")
print("Multi-Run Baseline Evaluation Suite Complete.")
print("==================================================")
