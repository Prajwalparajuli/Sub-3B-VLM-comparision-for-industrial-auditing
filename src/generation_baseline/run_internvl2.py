import torch
from PIL import Image
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from inference_utils import load_preprocessed_metadata, get_standard_prompt, save_results

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

# 3. Processing Protocol (from research.ipynb)
def get_pixel_values_safe(image, input_size=448):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # Move to the SAME device and dtype as the model
    pixel_values = transform(image).unsqueeze(0)
    if device == "cuda":
        pixel_values = pixel_values.to(torch.bfloat16).to(device)
    return pixel_values

# 4. Load Data
dataset = load_preprocessed_metadata()
results = []

# 5. Inference Loop
print(f"Starting inference on {len(dataset)} images...")
for i, item in enumerate(dataset):
    image_path = item.get("processed_path")
    constraint = item.get("constraint")
    
    if not image_path or not os.path.exists(image_path):
        print(f"WARNING: Skipping missing image: {image_path}")
        continue
    
    # Load Image
    image = Image.open(image_path).convert("RGB")
    
    # Process Inputs
    pixel_values = get_pixel_values_safe(image)
    standard_prompt = get_standard_prompt(constraint)
    question = f'<image>\n{standard_prompt}'
    
    # Generate Response
    with torch.no_grad():
        # InternVL2 uses model.chat for simple interaction
        response = model.chat(
            tokenizer, 
            pixel_values, 
            question, 
            generation_config=dict(
                max_new_tokens=256, 
                do_sample=False, 
                repetition_penalty=1.1 # SGP Standard
            ),
            num_patches_list=[1] # Disable tiling for baseline
        )
    
    # Store result
    result_entry = item.copy()
    result_entry["model_response"] = response
    results.append(result_entry)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images...")

# 6. Save Results
save_results(results, "internvl2")

# Cleanup
del model, tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
