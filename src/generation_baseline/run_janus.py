import torch
import os
import sys
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig
from inference_utils import load_preprocessed_metadata, get_standard_prompt, save_results

# 0. Runtime Patch for attrdict (Compatibility Fix)
# Janus depends on 'attrdict', which is broken/missing on Python 3.10+.
# Instead of modifying the Janus source or installing a dead package,
# we provide a local replacement at runtime.
from types import ModuleType
import sys

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

# 1. Path Resolution for Janus
# The notebook environment uses a local 'Janus' folder in the root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
janus_module_path = os.path.join(project_root, "Janus")

if os.path.exists(janus_module_path):
    sys.path.append(janus_module_path)
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    print(f"[OK] Found Janus module at {janus_module_path}")
else:
    print(f"[ERROR] Janus module NOT found at {janus_module_path}")
    print("Please ensure the Janus repository is cloned to the root of the project.")
    sys.exit(1)

# 2. Setup Device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 3. Load Model & Processor
local_model_path = "models/Janus"
print(f"Loading Janus model from {local_model_path}...")

# We load the CUSTOM processor that comes with the Janus repo
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(local_model_path)
tokenizer = vl_chat_processor.tokenizer

# 4-bit Quantization for sub-4GB VRAM compatibility (GTX 1060 optimized)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Official DeepSeek Way: Use MultiModalityCausalLM directly
vl_gpt = MultiModalityCausalLM.from_pretrained(
    local_model_path, 
    trust_remote_code=True, 
    quantization_config=quantization_config if device.startswith("cuda") else None,
    device_map=device if device.startswith("cuda") else None,
).eval()

print(f"Model loaded successfully on {device}.")

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
    
    # Prepare Prompt
    standard_prompt = get_standard_prompt(constraint)
    messages = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{standard_prompt}",
            "images": [image],
        },
        {"role": "Assistant", "content": ""},
    ]
    
    # Use the official processor to handle the formatting
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
            max_new_tokens=256,
            do_sample=False, # Fair Greedy comparison
            repetition_penalty=1.1, # SGP Standard
            use_cache=True,
        )
    
    response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    # Store result
    result_entry = item.copy()
    result_entry["model_response"] = response
    results.append(result_entry)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images...")

# 6. Save Results
save_results(results, "janus")

# Cleanup
del vl_gpt, vl_chat_processor
if torch.cuda.is_available():
    torch.cuda.empty_cache()
