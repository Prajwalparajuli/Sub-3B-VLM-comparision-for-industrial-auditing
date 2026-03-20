import torch
from PIL import Image
import os
import sys
import csv
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

# inference_utils lives in generation_baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_baseline"))
from inference_utils import load_preprocessed_metadata, save_results

# Add the local directory for image_utils
sys.path.insert(0, os.path.dirname(__file__))
from image_utils import apply_clahe_and_concatenate

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Model & Tokenizer
local_model_path = "models/InternVL2"
print(f"Loading InternVL2 model from {local_model_path}...")

model = AutoModel.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto" if device == "cuda" else None
).eval()

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True, use_fast=False)

print(f"Model loaded on {device}.")

# 3. Image Processing (same protocol as baseline)
def get_pixel_values_safe(image, input_size=448):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    pixel_values = transform(image).unsqueeze(0)
    if device == "cuda":
        pixel_values = pixel_values.to(torch.bfloat16).to(device)
    return pixel_values

# 4. Load Data
dataset = load_preprocessed_metadata()
N_RUNS = 3

for run_i in range(1, N_RUNS + 1):
    results = []
    torch.manual_seed(42 + run_i)

    # 5. Inference Loop
    print(f"\n--- Starting Contrast Run {run_i}/{N_RUNS} on {len(dataset)} images ---")
    
    for i, item in enumerate(dataset):
        image_path = item.get("processed_path")
        constraint = item.get("logic_constraint") or item.get("constraint") or "Inspect this image for any safety concern."
    
        if not image_path or not os.path.exists(image_path):
            print(f"WARNING: Skipping missing image: {image_path}")
            continue
    
        # Load Image
        original_image = Image.open(image_path).convert("RGB")
        
        # APPLY DUAL-CHANNEL CLAHE INNOVATION
        image = apply_clahe_and_concatenate(original_image, max_dim=512)
        pixel_values = get_pixel_values_safe(image)
    
        decomp_prompt = (
            f"<image>\n"
            f"You are an AI industrial safety auditor.\n"
            f"Safety Rule: '{constraint}'\n\n"
            f"Identify the current reading or condition shown in the image, then evaluate it against the rule.\n"
            f"Answer the following three questions exactly in order:\n\n"
            f"Q1: Observation: What is the exact numeric reading or visible physical condition shown in the image?\n"
            f"Q2: Evaluation: Does the observation from Q1 violate the Safety Rule? (Answer Yes or No)\n"
            f"Q3: Final Verdict: If Q2 is Yes, output UNSAFE. If Q2 is No, output SAFE.\n\n"
            f"Format your response exactly as:\n"
            f"Q1: Observation: [Your observation here]\n"
            f"Q2: Evaluation: [Yes or No]\n"
            f"Q3: Final Verdict: [SAFE or UNSAFE]"
        )
    
        # Generate Response
        with torch.no_grad():
            response = model.chat(
                tokenizer,
                pixel_values,
                decomp_prompt,
                generation_config=dict(
                    max_new_tokens=384,
                    do_sample=False,
                    repetition_penalty=1.1
                ),
                num_patches_list=[1]  # No tiling, same as baseline
            )
    
        # Store result
        result_entry = item.copy()
        result_entry["model_response"] = response
        result_entry["run_iteration"] = run_i
        results.append(result_entry)
    
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images...")

    # 6. Save Results
    save_results(results, "internvl2_contrast", iteration=run_i, out_dir="results/innovation/contrast")

# Cleanup
del model, tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
