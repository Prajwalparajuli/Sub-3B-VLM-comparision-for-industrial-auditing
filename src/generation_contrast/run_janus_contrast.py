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
from inference_utils import load_preprocessed_metadata, save_results

# Add the local directory for image_utils
sys.path.insert(0, os.path.dirname(__file__))
from image_utils import apply_clahe_and_concatenate

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
dataset = load_preprocessed_metadata()
N_RUNS = 3

for run_i in range(1, N_RUNS + 1):
    results = []
    torch.manual_seed(42 + run_i)

    # 5. Inference Loop
    print(f"\n--- Starting Contrast Run {run_i}/{N_RUNS} on {len(dataset)} images ---")
    
    for i, item in enumerate(dataset):
        image_path = item.get("processed_path")
        constraint = item.get("logic_constraint") or item.get("constraint") or "Inspect this image for any safety concern."
    
        if not image_path or not os.path.exists(image_path):
            print(f"WARNING: Skipping missing image: {image_path}")
            continue
    
        # Load Image
        original_image = Image.open(image_path).convert("RGB")
        
        # APPLY DUAL-CHANNEL CLAHE INNOVATION
        image = apply_clahe_and_concatenate(original_image, max_dim=512)
    
        # Chain-of-Thought prompt -- identical to all other CoT scripts.
        # Janus uses <image_placeholder> in the user turn.
        decomp_prompt = (
            f"<image_placeholder>\n"
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
    
        messages = [
            {
                "role": "User",
                "content": decomp_prompt,
                "images": [image],
            },
            {"role": "Assistant", "content": ""},
        ]
    
        # Use the official Janus processor to format inputs
        prepare_inputs = vl_chat_processor(
            conversations=messages, images=[image], force_batchify=True
        ).to(vl_gpt.device, torch.float16 if device.startswith("cuda") else torch.float32)
    
        # Generate
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
    
        # Store result
        result_entry = item.copy()
        result_entry["model_response"] = response
        result_entry["run_iteration"] = run_i
        results.append(result_entry)
    
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images...")

    # 6. Save Results
    save_results(results, "janus_contrast", iteration=run_i, out_dir="results/innovation/contrast")

# Cleanup
del vl_gpt, vl_chat_processor
if torch.cuda.is_available():
    torch.cuda.empty_cache()
