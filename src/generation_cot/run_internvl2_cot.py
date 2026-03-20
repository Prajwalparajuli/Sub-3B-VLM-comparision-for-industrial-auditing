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
    print(f"\n--- Starting CoT Run {run_i}/{N_RUNS} on {len(dataset)} images ---")
    
    for i, item in enumerate(dataset):
        image_path = item.get("processed_path")
        constraint = item.get("logic_constraint") or item.get("constraint") or "Inspect this image for any safety concern."
    
        if not image_path or not os.path.exists(image_path):
            print(f"WARNING: Skipping missing image: {image_path}")
            continue
    
        # Load Image
        image = Image.open(image_path).convert("RGB")
        pixel_values = get_pixel_values_safe(image)
    
        # Chain-of-Thought prompt -- identical to all other CoT scripts.
        # No per-model tuning so results are directly comparable.
        cot_prompt = (
            f"<image>\n"
            f"You are an industrial safety auditor. "
            f"Answer in exactly three steps:\n\n"
            f"STEP 1 - OBSERVATION: Describe only what you see in the image. "
            f"For a gauge: state the numeric reading and its unit. "
            f"For a pipe: describe the physical surface condition.\n\n"
            f"STEP 2 - RULE APPLICATION: The safety rule is: {constraint} "
            f"Show whether the observation from Step 1 satisfies this rule.\n\n"
            f"STEP 3 - VERDICT: Write ONLY one of these two lines, nothing else:\n"
            f"Final Verdict: SAFE\n"
            f"Final Verdict: UNSAFE"
        )
    
        # Generate Response -- max_new_tokens raised to 384 to fit the 3-step answer
        with torch.no_grad():
            response = model.chat(
                tokenizer,
                pixel_values,
                cot_prompt,
                generation_config=dict(
                    max_new_tokens=384,
                    do_sample=False,
                    repetition_penalty=1.1
                ),
                num_patches_list=[1]  # No tiling, same as baseline
            )
    
        # Store result -- same fields as baseline so parse_results.py works unchanged
        result_entry = item.copy()
        result_entry["model_response"] = response
        result_entry["run_iteration"] = run_i
        results.append(result_entry)
    
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images...")

    # 6. Save Results
    save_results(results, "internvl2_cot", iteration=run_i, out_dir="results/innovation/cot")

# Cleanup
del model, tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
