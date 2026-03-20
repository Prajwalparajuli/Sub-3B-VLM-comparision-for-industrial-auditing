import os
# Force single-GPU mode -- same requirement as baseline to prevent
# bitsandbytes from spreading layers across cuda:0 and cuda:1,
# which causes torch.gather failures in the scatter operation.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import gc
import sys
import csv
import types
import math
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# inference_utils lives in generation_baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_baseline"))
from inference_utils import load_preprocessed_metadata, save_results

# Add the local directory for image_utils
sys.path.insert(0, os.path.dirname(__file__))
from image_utils import apply_clahe_and_concatenate

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load Model & Tokenizer
local_model_path = "models/MiniCPM"
print(f"Loading MiniCPM-V-2 model from {local_model_path}...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    llm_int8_skip_modules=["vpm", "resampler"]  # Keep vision towers in float16
) if device == "cuda" else None

# LOADING FIX: patch PreTrainedModel.to() to allow dispatch_model during load.
if device == "cuda":
    import transformers.modeling_utils as _mu
    _original_to = _mu.PreTrainedModel.to

    def _permissive_to(self, *args, **kwargs):
        if args and isinstance(args[0], (int, torch.device, str)):
            return torch.nn.Module.to(self, *args, **kwargs)
        from transformers.utils.quantization_config import QuantizationMethod
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            raise ValueError(
                "`.to` is not supported for `4-bit` or `8-bit` bitsandbytes models."
            )
        return _original_to(self, *args, **kwargs)

    _mu.PreTrainedModel.to = _permissive_to

model = AutoModel.from_pretrained(
    local_model_path,
    trust_remote_code=True,
    quantization_config=quant_config,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

# Restore original .to() restriction
if device == "cuda":
    _mu.PreTrainedModel.to = _original_to

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)

# PATCH A: Vision Dtype Fix
def patched_get_vision_embedding(self, pixel_values):
    res = []
    dtype = torch.float16
    vpm_device = next(self.vpm.parameters()).device

    def process_each_pixel(pixel_value, dtype, vpm_device, config, vpm, resampler):
        H, W = pixel_value.shape[-2:]
        target_size = (math.ceil(H / config.patch_size), math.ceil(W / config.patch_size))
        px = pixel_value.unsqueeze(0).to(device=vpm_device, dtype=dtype)
        vision_embedding = self.vpm_forward_features(px)
        if hasattr(vpm, "num_prefix_tokens") and vpm.num_prefix_tokens > 0:
            vision_embedding = vision_embedding[:, vpm.num_prefix_tokens:]
        return resampler(vision_embedding, target_size)

    for pixel_value in pixel_values:
        result = process_each_pixel(pixel_value, dtype, vpm_device, self.config, self.vpm, self.resampler)
        res.append(result)
    return torch.vstack(res)

model.get_vision_embedding = types.MethodType(patched_get_vision_embedding, model)

# Fix: cast resampler.pos_embed to float16
if hasattr(model.resampler, "pos_embed") and model.resampler.pos_embed.dtype != torch.float16:
    model.resampler.pos_embed.data = model.resampler.pos_embed.data.to(torch.float16)

# PATCH B: Multi-GPU scatter fix
def patched_get_vllm_embedding(self, data):
    if "vision_hidden_states" not in data:
        pixel_values_list = data["pixel_values"]
        vision_hidden_states = []
        for pixel_values in pixel_values_list:
            if len(pixel_values) > 0:
                vision_hidden_states.append(self.get_vision_embedding(pixel_values))
            elif self.training:
                dtype = self.llm.lm_head.weight.dtype
                dev = self.llm.lm_head.weight.device
                dummy_image = torch.zeros((1, 3, 224, 224), device=dev, dtype=dtype)
                vision_hidden_states.append(self.get_vision_embedding(dummy_image))
            else:
                vision_hidden_states.append([])
    else:
        vision_hidden_states = data["vision_hidden_states"]

    et_device = self.llm.model.embed_tokens.weight.device
    input_ids = data["input_ids"].to(et_device)
    vllm_embedding = (
        self.llm.model.embed_tokens(input_ids) * self.llm.config.scale_emb
    )

    llm_device = vllm_embedding.device
    vision_hidden_states = [
        i.type(vllm_embedding.dtype).to(llm_device) if isinstance(i, torch.Tensor) else i
        for i in vision_hidden_states
    ]

    bs = len(data["input_ids"])
    for i in range(bs):
        cur_vs_hs = vision_hidden_states[i]
        if len(cur_vs_hs) > 0:
            cur_vllm_emb = vllm_embedding[i]
            cur_image_bound = data["image_bound"][i]
            if len(cur_image_bound) > 0:
                image_indices = torch.stack([
                    torch.arange(r[0], r[1], dtype=torch.long)
                    for r in cur_image_bound
                ]).to(llm_device)
                cur_vllm_emb.scatter_(
                    0,
                    image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                    cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                )
            elif self.training:
                cur_vllm_emb += cur_vs_hs[0].mean() * 0

    return vllm_embedding, vision_hidden_states

model.get_vllm_embedding = types.MethodType(patched_get_vllm_embedding, model)
model.eval()

# 3. Load Data
dataset = load_preprocessed_metadata()
N_RUNS = 3

for run_i in range(1, N_RUNS + 1):
    results = []
    torch.manual_seed(42 + run_i)

    # 4. Inference Loop
    print(f"\n--- Starting Contrast Run {run_i}/{N_RUNS} on {len(dataset)} images ---")
    
    for i, item in enumerate(dataset):
        image_path = item.get("processed_path")
        constraint = item.get("logic_constraint") or item.get("constraint") or "Inspect this image for any safety concern."
    
        if not image_path or not os.path.exists(image_path):
            continue
    
        original_image = Image.open(image_path).convert("RGB")
    
        # APPLY DUAL-CHANNEL CLAHE INNOVATION
        # apply_clahe_and_concatenate internally enforces max_dim=512 via LANCZOS,
        # making it completely VRAM safe without an external resizing step.
        image = apply_clahe_and_concatenate(original_image, max_dim=512)
    
        # Chain-of-Thought prompt
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
    
        msgs = [{"role": "user", "content": decomp_prompt}]
    
        with torch.no_grad():
            response, _, _ = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=False
            )
    
        # Store result
        result_entry = item.copy()
        result_entry["model_response"] = response
        result_entry["run_iteration"] = run_i
        results.append(result_entry)
    
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images...")

    # 5. Save Results
    save_results(results, "minicpm_contrast", iteration=run_i, out_dir="results/innovation/contrast")

# Memory Hygiene
del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
