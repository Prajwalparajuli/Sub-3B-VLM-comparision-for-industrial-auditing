import os
import torch
import subprocess
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq, 
    AutoTokenizer, 
    AutoModel, 
    BitsAndBytesConfig
)

# 0. SETUP UTILITIES
def ensure_janus_repo():
    """Ensure the Janus repository is present in the project root for reproducibility."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    janus_path = os.path.join(project_root, "Janus")
    
    if not os.path.exists(janus_path):
        print(f"Janus repository not found at {janus_path}. Cloning for reproducibility...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/deepseek-ai/Janus.git", janus_path],
                check=True
            )
            print("[OK] Janus repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to clone Janus repository: {e}")
            print("Please manually clone it to the project root: https://github.com/deepseek-ai/Janus.git")
    else:
        print("[OK] Janus repository already present.")

# 1. DOWNLOAD PROTOCOLS
def download_smolvlm(models_dir):
    """BAP: bfloat16 for < 1.5B models"""
    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    save_path = os.path.join(models_dir, "SmolVLM")
    if not os.path.exists(save_path):
        print(f"Downloading {model_id}...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        processor.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print(f"Saved to {save_path}")
    else:
        print("SmolVLM already exists.")

def download_internvl2(models_dir):
    """BAP: bfloat16 for < 1.5B models"""
    model_id = "OpenGVLab/InternVL2-1B"
    save_path = os.path.join(models_dir, "InternVL2")
    if not os.path.exists(save_path):
        print(f"Downloading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print(f"Saved to {save_path}")
    else:
        print("InternVL2 already exists.")

def download_minicpm(models_dir):
    """BAP: Download full-precision weights; 4-bit quantization happens at inference time.
    Uses snapshot_download so no GPU is required during download (runs under ragenv)."""
    from huggingface_hub import snapshot_download
    model_id = "openbmb/MiniCPM-V-2"
    save_path = os.path.join(models_dir, "MiniCPM")
    if not os.path.exists(save_path):
        print(f"Downloading {model_id} via snapshot_download (full precision)...")
        snapshot_download(repo_id=model_id, local_dir=save_path, local_dir_use_symlinks=False)
        print(f"Saved to {save_path}")
    else:
        print("MiniCPM already exists.")

def download_janus(models_dir):
    """BAP: bfloat16 for < 1.5B models - Using snapshot_download for custom architecture"""
    model_id = "deepseek-ai/Janus-Pro-1B"
    save_path = os.path.join(models_dir, "Janus")
    if not os.path.exists(save_path):
        print(f"Downloading {model_id} via snapshot_download...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=save_path, local_dir_use_symlinks=False)
        print(f"Saved to {save_path}")
    else:
        print("Janus already exists.")

def download_qwen2_vl(models_dir):
    """BAP: bfloat16 for < 1.5B models"""
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    save_path = os.path.join(models_dir, "Qwen2VL")
    if not os.path.exists(save_path):
        print(f"Downloading {model_id}...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        processor.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print(f"Saved to {save_path}")
    else:
        print("Qwen2-VL already exists.")

if __name__ == "__main__":
    # 1. Setup repository dependencies
    ensure_janus_repo()
    
    # 2. Create the models directory in the project root
    base_models_dir = "models"
    os.makedirs(base_models_dir, exist_ok=True)
    
    print("Starting bulk model download...")
    download_smolvlm(base_models_dir)
    download_internvl2(base_models_dir)
    download_minicpm(base_models_dir)
    download_janus(base_models_dir)
    download_qwen2_vl(base_models_dir) # Added call for Qwen2-VL
    print("All models prepared.")
