import torch
from PIL import Image
import os
import sys
import csv
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

# inference_utils lives in generation_baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_baseline"))
from inference_utils import load_preprocessed_metadata, save_results

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
dataset = load_preprocessed_metadata()
N_RUNS = 3

for run_i in range(1, N_RUNS + 1):
    results = []
    torch.manual_seed(42 + run_i)

    # 4. Inference Loop
    print(f"\n--- Starting CoT Run {run_i}/{N_RUNS} on {len(dataset)} images ---")
    
    for i, item in enumerate(dataset):
        image_path = item.get("processed_path")
        constraint = item.get("logic_constraint") or item.get("constraint") or "Inspect this image for any safety concern."
    
        if not image_path or not os.path.exists(image_path):
            print(f"WARNING: Skipping missing image: {image_path}")
            continue
    
        # Load Image
        image = Image.open(image_path).convert("RGB")
    
        # Chain-of-Thought prompt -- identical to all other CoT scripts.
        # No per-model tuning so results are directly comparable.
        cot_prompt = (
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
    
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": cot_prompt}
                ]
            }
        ]
    
        # Apply Template and process vision info (same as baseline)
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
    
        # Generate -- max_new_tokens raised to 384 for the 3-step response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                repetition_penalty=1.1
            )
    
        # Trim prompt tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
        # Store result -- same fields as baseline so parse_results.py works unchanged
        result_entry = item.copy()
        result_entry["model_response"] = output_text
        result_entry["run_iteration"] = run_i
        results.append(result_entry)
    
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images...")

    # 5. Save Results
    save_results(results, "qwen2_vl_cot", iteration=run_i, out_dir="results/innovation/cot")

# Cleanup
del model, processor
if torch.cuda.is_available():
    torch.cuda.empty_cache()
