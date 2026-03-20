import torch
import os
import sys
import csv
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig

# 0. Runtime patch for attrdict (same fix as baseline)
# Janus depends on 'attrdict', which is broken on Python 3.10+.
# We inject a local replacement before any Janus import.
from types import ModuleType

class AttrDict(dict):
    """A simple dict subclass that allows dot notation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, AttrDict):
                self[k] = AttrDict(v)
    def __getattr__(self, key):
        try: return self[key]
        except KeyError: raise AttributeError(key)
    def __setattr__(self, key, value): self[key] = value

attrdict_module = ModuleType("attrdict")
attrdict_module.AttrDict = AttrDict
sys.modules["attrdict"] = attrdict_module

# 1. Path resolution for local Janus repo
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
janus_module_path = os.path.join(project_root, "Janus")

if os.path.exists(janus_module_path):
    sys.path.append(janus_module_path)
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    print(f"[OK] Found Janus module at {janus_module_path}")
else:
    print(f"[ERROR] Janus module NOT found at {janus_module_path}")
    sys.exit(1)

# inference_utils lives in generation_baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_baseline"))
from inference_utils import load_preprocessed_metadata

# 2. Setup Device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 3. Load Model & Processor
local_model_path = "models/Janus"
print(f"Loading Janus model from {local_model_path}...")

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(local_model_path)
tokenizer = vl_chat_processor.tokenizer

# 4-bit NF4 quantization -- same as baseline (required for 4GB VRAM)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

vl_gpt = MultiModalityCausalLM.from_pretrained(
    local_model_path,
    trust_remote_code=True,
    quantization_config=quantization_config if device.startswith("cuda") else None,
    device_map=device if device.startswith("cuda") else None,
).eval()

print(f"Model loaded successfully on {device}.")

# 4. Load Data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_baseline"))
from inference_utils import load_preprocessed_metadata, save_results
dataset = load_preprocessed_metadata()

# Number of evaluation runs to capture generation variance
N_RUNS = 3

for run_i in range(1, N_RUNS + 1):
    results = []
    # Seed control for reproducibility across runs
    torch.manual_seed(42 + run_i)

    # 5. Inference Loop
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
        # Janus uses <image_placeholder> in the user turn.
        cot_prompt = (
            f"<image_placeholder>\n"
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
                "role": "User",
                "content": cot_prompt,
                "images": [image],
            },
            {"role": "Assistant", "content": ""},
        ]
    
        # Use the official Janus processor to format inputs
        prepare_inputs = vl_chat_processor(
            conversations=messages, images=[image], force_batchify=True
        ).to(vl_gpt.device, torch.float16 if device.startswith("cuda") else torch.float32)
    
        # Generate -- max_new_tokens raised to 384 for the 3-step response
        with torch.no_grad():
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=vl_gpt.prepare_inputs_embeds(**prepare_inputs),
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1,
                use_cache=True,
            )
    
        response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
        # Store result -- same fields as baseline so parse_results.py works unchanged
        result_entry = item.copy()
        result_entry["model_response"] = response
        result_entry["run_iteration"] = run_i
        results.append(result_entry)
    
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images...")
            
    # 6. Save Results
    save_results(results, "janus_cot", iteration=run_i, out_dir="results/innovation/cot")

# Cleanup
del vl_gpt, vl_chat_processor
if torch.cuda.is_available():
    torch.cuda.empty_cache()
