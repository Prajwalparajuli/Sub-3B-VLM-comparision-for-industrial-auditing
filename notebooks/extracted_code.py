import torch
from PIL import Image
import os
from io import BytesIO



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

image = Image.open("2.jpg").convert("RGB")
image = image.resize((336, 336))

display(image)

# SmolVLM

from transformers import AutoProcessor, AutoModelForVision2Seq

print("Loading model...")

smol_model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype = torch.float16,
    trust_remote_code = True
).to(device)

smol_processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")

print("Model loaded.")

# Prepare the Input prompt
prompt = "User: <image>\nWhat is the reading on this gauge? Answer briefly.\nAssistant:"

# Process and Generate
inputs = smol_processor(text = prompt,
                        images = image,
                        return_tensors = "pt").to(device)

print("Running Inference...")
generate_ids = smol_model.generate(**inputs, max_new_tokens = 100)
output_text = smol_processor.batch_decode(generate_ids, skip_special_tokens = True)[0]

display(image)
print("SmolVLM Output:")
print(output_text)

# Cleanup to free GPU memory for the next cell
del smol_model, smol_processor
torch.cuda.empty_cache()