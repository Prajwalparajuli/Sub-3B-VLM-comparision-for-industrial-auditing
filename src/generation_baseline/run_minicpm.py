import os
# Force single-GPU mode: prevents bitsandbytes/accelerate from spreading layers
# across cuda:0 and cuda:1, which causes torch.gather failures in beam search.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import gc
import types
import math
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from inference_utils import load_preprocessed_metadata, get_standard_prompt, save_results

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

# LOADING FIX: transformers 4.40's from_pretrained calls dispatch_model() which calls
# model.to(device) via accelerate/big_modeling.py:502.
# transformers prevents .to() on quantized models (modeling_utils.py:2670) because
# it would destroy quantization — but we need dispatch_model to run so that
# non-quantized layers (embed_tokens, norm, vpm, resampler) get moved to GPU.
# Fix: temporarily patch PreTrainedModel.to() to pass through device-only .to() calls
# during loading (this is safe — bitsandbytes handles its own weights, others need moving).
if device == "cuda":
    import transformers.modeling_utils as _mu
    _original_to = _mu.PreTrainedModel.to

    def _permissive_to(self, *args, **kwargs):
        # Allow pure device moves (e.g. model.to(0), model.to("cuda:0"))
        # Block dtype changes on quantized models (original restriction)
        if args and isinstance(args[0], (int, torch.device, str)):
            # device-only move — call nn.Module.to directly (bypasses the check)
            return torch.nn.Module.to(self, *args, **kwargs)
        from transformers.utils.quantization_config import QuantizationMethod
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            raise ValueError(
                "`.to` is not supported for `4-bit` or `8-bit` bitsandbytes models. "
                "Please use the model as it is."
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

# 3. Patch A: Vision Dtype Fix
# bitsandbytes quantises lm_head to uint8; the original code reads
# self.llm.lm_head.weight.dtype which returns uint8 and breaks vision.
# Hardcode float16 exactly as the working notebook does.
def patched_get_vision_embedding(self, pixel_values):
    res = []
    dtype = torch.float16  # Hardcode instead of self.llm.lm_head.weight.dtype
    # Get the device the vision encoder is on
    vpm_device = next(self.vpm.parameters()).device

    def process_each_pixel(pixel_value, dtype, vpm_device, config, vpm, resampler):
        H, W = pixel_value.shape[-2:]
        target_size = (math.ceil(H / config.patch_size), math.ceil(W / config.patch_size))
        # Move to vpm device and cast dtype
        px = pixel_value.unsqueeze(0).to(device=vpm_device, dtype=dtype)
        vision_embedding = self.vpm_forward_features(px)
        if hasattr(vpm, 'num_prefix_tokens') and vpm.num_prefix_tokens > 0:
            vision_embedding = vision_embedding[:, vpm.num_prefix_tokens:]
        return resampler(vision_embedding, target_size)

    for pixel_value in pixel_values:
        result = process_each_pixel(pixel_value, dtype, vpm_device, self.config, self.vpm, self.resampler)
        res.append(result)
    return torch.vstack(res)

model.get_vision_embedding = types.MethodType(patched_get_vision_embedding, model)

# Fix: resampler.pos_embed is float32 by default, causes Float vs Half mismatch.
# Cast it to float16 to match all other resampler parameters.
if hasattr(model.resampler, 'pos_embed') and model.resampler.pos_embed.dtype != torch.float16:
    model.resampler.pos_embed.data = model.resampler.pos_embed.data.to(torch.float16)
    print(f"Fixed resampler.pos_embed: {model.resampler.pos_embed.dtype}")

# When device_map="auto" spreads across 2 GPUs, the scatter in get_vllm_embedding
# fails because vision tensors (cuda:0) and LLM embeddings (cuda:1) are on different devices.
# We patch get_vllm_embedding to move vision tensors to the LLM embedding device before scatter.
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

    # Move input_ids to embed_tokens device (may be CPU due to _process_list/pad construction)
    et_device = self.llm.model.embed_tokens.weight.device
    input_ids = data["input_ids"].to(et_device)
    vllm_embedding = (
        self.llm.model.embed_tokens(input_ids) * self.llm.config.scale_emb
    )

    # KEY FIX: move vision tensors to same device as LLM embeddings before scatter
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
                image_indices = torch.stack(
                    [
                        torch.arange(r[0], r[1], dtype=torch.long)
                        for r in cur_image_bound
                    ]
                ).to(llm_device)

                cur_vllm_emb.scatter_(
                    0,
                    image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                    cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                )
            elif self.training:
                cur_vllm_emb += cur_vs_hs[0].mean() * 0

    return vllm_embedding, vision_hidden_states

model.get_vllm_embedding = types.MethodType(patched_get_vllm_embedding, model)

# 5. Eval mode
model.eval()

# Diagnostics
print(f"LM Head Dtype:   {model.llm.lm_head.weight.dtype}")
print(f"Resampler Dtype: {model.resampler.ln_q.weight.dtype}")
print(f"VPM Dtype:       {next(model.vpm.parameters()).dtype}")
print("Model Ready.")

# 6. Load Data
dataset = load_preprocessed_metadata()

# Number of evaluation runs to capture generation variance
N_RUNS = 3

for run_i in range(1, N_RUNS + 1):
    results = []
    # Seed control for reproducibility across runs
    torch.manual_seed(42 + run_i)

    # 7. Inference Loop
    print(f"\n--- Starting Run {run_i}/{N_RUNS} on {len(dataset)} images ---")
    for i, item in enumerate(dataset):
        image_path = item.get("processed_path")
        # Use logic_constraint (the actual safety rule, e.g. "Alert if pressure exceeds 0.10 bar.")
        # NOT constraint (the artifact_tag like "Oblique Angle") which causes the model to
        # reason about image quality instead of reading the gauge/inspecting the surface.
        constraint = item.get("logic_constraint") or item.get("constraint") or "Inspect this image for any safety concern."

        if not image_path or not os.path.exists(image_path):
            print(f"WARNING: Skipping missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        w, h = image.size
        MAX_EDGE = 384
        scale = MAX_EDGE / max(w, h)
        proc_image = image.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)

        msgs = [{'role': 'user', 'content': get_standard_prompt(constraint)}]

        with torch.no_grad():
            response, context, _ = model.chat(
                image=proc_image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=False
            )

        result_entry = item.copy()
        result_entry["model_response"] = response
        result_entry["run_iteration"] = run_i  # Track which run this is
        results.append(result_entry)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images...")

    # 8. Save Results
    save_results(results, "minicpm", iteration=run_i)

# 9. Memory Hygiene
del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("VRAM Hygiene Complete.")
