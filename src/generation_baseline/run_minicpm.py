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

# 2. Load Model & Processor
local_model_path = "models/MiniCPM"
print(f"Loading MiniCPM-V-2 model from {local_model_path}...")

# BAP: 4-bit quantization for models >= 2B
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

# 3. Apply Stability Patches (from research.ipynb)

# Patch A: Fix Dtype Leak in Vision Embedding
def patched_get_vision_embedding(self, pixel_values):
    res = []
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
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

# Patch B: Nuclear Rotary Fix (Self-Healing)
if device == "cuda":
    def self_healing_rotary_forward(self, x, seq_len=None):
        if self.inv_freq.numel() == 0:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.inv_freq = inv_freq
        if not hasattr(self, 'cos_cached') or self.cos_cached.numel() == 0:
            self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device=x.device, dtype=torch.float32)
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)

    for layer in model.llm.model.layers:
        rotary = layer.self_attn.rotary_emb
        rotary.forward = types.MethodType(self_healing_rotary_forward, rotary)

# Patch C: Precision Locks
if device == "cuda":
    model.vpm = model.vpm.to(torch.float16)
    model.resampler = model.resampler.to(torch.float16)

model.eval()
print("Model Ready (All Patches Applied).")

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
    
    # Standard 384px Resizing (BAP)
    w, h = image.size
    MAX_EDGE = 384
    scale = MAX_EDGE / max(w, h)
    proc_image = image.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)
    
    # Prepare Prompt
    standard_prompt = get_standard_prompt(constraint)
    msgs = [{'role': 'user', 'content': standard_prompt}]
    
    # Generate Response
    with torch.no_grad():
        response, context, _ = model.chat(
            image=proc_image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            sampling=False, # Fair Greed
            repetition_penalty=1.1 # SGP Standard
        )
    
    # Store result
    result_entry = item.copy()
    result_entry["model_response"] = response
    results.append(result_entry)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images...")

# 6. Save Results
save_results(results, "minicpm")

# Cleanup
del model, tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
