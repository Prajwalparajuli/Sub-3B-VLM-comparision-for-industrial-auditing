import torch
from PIL import Image
import os
import sys

# inference_utils lives in generation_baseline; add it to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_baseline"))
from inference_utils import load_preprocessed_metadata

# Add the local directory for image_utils
sys.path.insert(0, os.path.dirname(__file__))
from image_utils import apply_clahe_and_concatenate

# 1. Setup Device & Resolution
device = "cuda" if torch.cuda.is_available() else "cpu"
RESOLUTION = 384
TOKENS = (RESOLUTION // 64) ** 2  # 36 tokens at 384px (same as baseline)

# 2. Load Model & Processor
from transformers import AutoProcessor, AutoModelForVision2Seq

local_model_path = "models/SmolVLM"
print(f"Loading SmolVLM model from {local_model_path}...")

processor = AutoProcessor.from_pretrained(local_model_path)

# Keep the same fixed-resolution settings as the baseline run
processor.image_processor.size["longest_edge"] = RESOLUTION
processor.image_processor.max_image_size = {"longest_edge": RESOLUTION}
processor.image_seq_len = TOKENS

model = AutoModelForVision2Seq.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    trust_remote_code=True
).to(device)

print(f"Model loaded on {device}.")

# 3. Load Data
dataset = load_preprocessed_metadata()
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
    # 512 max_dim ensures it stays VRAM-safe when passed to processor
    image = apply_clahe_and_concatenate(original_image, max_dim=512)

    # Chain-of-Thought prompt: three explicit steps force the model to
    # commit to a perception reading before applying logic.
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
                {"type": "image"},
                {"type": "text", "text": cot_prompt}
            ]
        }
    ]

    # Apply the chat template
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    prompt_with_nudge = prompt + "STEP 1 - OBSERVATION: "
    inputs = processor(text=prompt_with_nudge, images=image, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1
        )

    # Trim the prompt tokens so we only decode the new output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Clean stray "Assistant:" prefix if the model adds it
    output_text = output_text.strip()
    if output_text.lower().startswith("assistant:"):
        output_text = output_text[len("assistant:"):].strip()

    output_text = "STEP 1 - OBSERVATION: " + output_text
    
    # Store result
    result_entry = item.copy()
    result_entry["model_response"] = output_text
    results.append(result_entry)

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images...")

# 5. Save Results
output_dir = "results/innovation/contrast_cot"
os.makedirs(output_dir, exist_ok=True)

import csv
output_path = os.path.join(output_dir, "smolvlm_contrast_cot_results.csv")
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
