import torch
from PIL import Image
import os
from transformers import AutoProcessor, AutoModelForVision2Seq
from inference_utils import load_preprocessed_metadata, get_standard_prompt, save_results

# 1. Setup Device & Resolution
device = "cuda" if torch.cuda.is_available() else "cpu"
RESOLUTION = 384
TOKENS = (RESOLUTION // 64) ** 2  # = 36 tokens for SmolVLM at 384px

# 2. Load Model & Processor
local_model_path = "models/SmolVLM"
print(f"Loading SmolVLM model from {local_model_path}...")

processor = AutoProcessor.from_pretrained(local_model_path)

# Configure processor for fixed resolution
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
print(f"Starting inference on {len(dataset)} images...")
for i, item in enumerate(dataset):
    image_path = item.get("processed_path")
    constraint = item.get("constraint")
    
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
                {"type": "image"},
                {"type": "text", "text": standard_prompt}
            ]
        }
    ]
    
    # Apply Template
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Process Inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    # Generate Response
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1  # SGP Standard
        )
    
    output_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    
    # Extract response (processor.apply_chat_template might include the prompt in decode, 
    # but SmolVLM batch_decode usually gives the full message including Assistant: ...)
    # If the prompt is included, we might want to trim it.
    
    # Store result
    result_entry = item.copy()
    result_entry["model_response"] = output_text
    results.append(result_entry)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images...")

# 5. Save Results
save_results(results, "smolvlm")

# Cleanup
del model, processor
if torch.cuda.is_available():
    torch.cuda.empty_cache()