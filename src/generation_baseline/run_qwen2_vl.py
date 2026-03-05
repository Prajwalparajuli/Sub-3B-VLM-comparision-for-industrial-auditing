import torch
from PIL import Image
import os
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from inference_utils import load_preprocessed_metadata, get_standard_prompt, save_results

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
results = []

# 4. Inference Loop
print(f"Starting inference on {len(dataset)} images...")
for i, item in enumerate(dataset):
    image_path = item.get("processed_path")
    constraint = item.get("logic_constraint") or item.get("constraint") or "Inspect this image for any safety concern."
    
    if not image_path or not os.path.exists(image_path):
        print(f"WARNING: Skipping missing image: {image_path}")
        continue
    
    # Load Image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare Prompt
    standard_prompt = get_standard_prompt(constraint)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": standard_prompt}
            ]
        }
    ]
    
    # Apply Template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process Vision Info
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Process Inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate Response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=False, # Greedy
            repetition_penalty=1.1 # SGP Standard
        )
    
    # Trim prompt from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Store result
    result_entry = item.copy()
    result_entry["model_response"] = output_text
    results.append(result_entry)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images...")

# 5. Save Results
save_results(results, "qwen2_vl")

# Cleanup
del model, processor
if torch.cuda.is_available():
    torch.cuda.empty_cache()
