import torch
import gc
import types
import math
import os
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from inference_utils import load_preprocessed_metadata, get_standard_prompt, save_results

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load Model & Tokenizer
local_model_path = "models/MiniCPM"
print(f"Loading MiniCPM-V-2 model from {local_model_path}...")

# 4-bit quantization config (only on CUDA)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=["vpm", "resampler"]  # Keep vision in float16
) if device == "cuda" else None

model = AutoModel.from_pretrained(
    local_model_path,
    trust_remote_code=True,
    quantization_config=quant_config,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)

# 3. Patch A: Vision Dtype Fix
#    bitsandbytes quantises lm_head to uint8; the original code reads
#    self.llm.lm_head.weight.dtype which returns uint8 and breaks vision.
#    Hardcode float16 exactly as the working notebook does.
def patched_get_vision_embedding(self, pixel_values):
    res = []
    dtype = torch.float16  # Hardcode instead of self.llm.lm_head.weight.dtype

    def process_each_pixel(pixel_value, dtype, config, vpm, resampler):
        H, W = pixel_value.shape[-2:]
        target_size = (math.ceil(H / config.patch_size), math.ceil(W / config.patch_size))
        vision_embedding = self.vpm_forward_features(pixel_value.unsqueeze(0).type(dtype))
        if hasattr(vpm, 'num_prefix_tokens') and vpm.num_prefix_tokens > 0:
            vision_embedding = vision_embedding[:, vpm.num_prefix_tokens:]
        return resampler(vision_embedding, target_size)

    for pixel_value in pixel_values:
        result = process_each_pixel(pixel_value, dtype, self.config, self.vpm, self.resampler)
        res.append(result)
    return torch.vstack(res)

model.get_vision_embedding = types.MethodType(patched_get_vision_embedding, model)

# 4. Precision Locks (matches notebook exactly)
model.vpm = model.vpm.to(torch.float16)
model.resampler = model.resampler.to(torch.float16)

# 5. Eval mode
model.eval()

# Diagnostics (same as notebook)
print(f"LM Head Dtype: {model.llm.lm_head.weight.dtype}")
print(f"Resampler Dtype: {model.resampler.ln_q.weight.dtype}")
print(f"VPM Dtype: {next(model.vpm.parameters()).dtype}")
print("Model Ready (transformers 4.40 + NF4 + Vision Fix)")

# 6. Load Data
dataset = load_preprocessed_metadata()
results = []

# 7. Inference Loop
print(f"Starting inference on {len(dataset)} images...")
for i, item in enumerate(dataset):
    image_path = item.get("processed_path")
    constraint = item.get("constraint")

    if not image_path or not os.path.exists(image_path):
        print(f"WARNING: Skipping missing image: {image_path}")
        continue

    # Load Image
    image = Image.open(image_path).convert("RGB")

    # Standard 384px Resizing
    w, h = image.size
    MAX_EDGE = 384
    scale = MAX_EDGE / max(w, h)
    proc_image = image.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)

    # Prepare Prompt
    standard_prompt = get_standard_prompt(constraint)
    msgs = [{'role': 'user', 'content': standard_prompt}]

    # Generate Response (matches notebook: sampling=False, no extra kwargs)
    with torch.no_grad():
        response, context, _ = model.chat(
            image=proc_image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            sampling=False          # Deterministic output per protocol
        )

    # Store result
    result_entry = item.copy()
    result_entry["model_response"] = response
    results.append(result_entry)

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images...")

# 8. Save Results
save_results(results, "minicpm")

# 9. Memory Hygiene
del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("VRAM Hygiene Complete.")
