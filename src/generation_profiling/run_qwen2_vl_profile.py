import torch
from PIL import Image
import os
import sys
import csv
import time
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

# inference_utils lives in generation_baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_baseline"))
from inference_utils import load_preprocessed_metadata

# Add the local directory for image_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_contrast"))
from image_utils import apply_clahe_and_concatenate

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Model & Processor
local_model_path = "models/Qwen2VL"
print(f"Loading Qwen2-VL model from {local_model_path}...")

processor = AutoProcessor.from_pretrained(local_model_path)
model = AutoModelForVision2Seq.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True
).eval()

print(f"Model loaded on {device}.")

# 3. Load Data
dataset = load_preprocessed_metadata()[:50]  # Profile on 50-row subset
results = []

# 4. Inference Loop
print(f"Starting Dual-Channel CLAHE inference on {len(dataset)} images...")

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

    # Chain-of-Thought prompt -- identical to all other CoT scripts.
    decomp_prompt = (
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

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": decomp_prompt}
            ]
        }
    ]

    # Apply Template and process vision info
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Generate
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            repetition_penalty=1.1
        )

    end_time = time.perf_counter()
    inference_time_sec = end_time - start_time
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        peak_vram_gb = 0.0

    # Trim prompt tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Store result
    result_entry = item.copy()
    result_entry["model_response"] = output_text
    result_entry["inference_time_sec"] = inference_time_sec
    result_entry["peak_vram_gb"] = peak_vram_gb
    results.append(result_entry)

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images...")

# 5. Save Results
output_dir = "results/profiling"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "qwen2_vl_profile_results.csv")

if results:
    keys = results[0].keys()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
print(f"Contrast results saved to {output_path}")

# Cleanup
del model, processor
if torch.cuda.is_available():
    torch.cuda.empty_cache()
