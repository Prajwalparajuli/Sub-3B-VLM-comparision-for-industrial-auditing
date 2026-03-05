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
    constraint = item.get("logic_constraint") or item.get("constraint") or "Inspect this image for any safety concern."
    
    if not image_path or not os.path.exists(image_path):
        print(f"WARNING: Skipping missing image: {image_path}")
        continue
    
    # Load Image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare Prompt
    standard_prompt = get_standard_prompt(constraint)
    user_content = standard_prompt.replace("System: You are an industrial auditor. ", "")
    
    # Few-shot example to guide the small model
    example_format = (
        "Example:\nUser: Alert if pressure > 0.5 bar. [Image of 0.6 bar]\n"
        "Assistant: Reasoning: The needle points to 0.6. 0.6 is greater than 0.5.\n"
        "Final Verdict: UNSAFE"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"You are an industrial safety auditor. {example_format}\n\nTask: {user_content}\nAlways end with exactly one of: 'Final Verdict: SAFE' or 'Final Verdict: UNSAFE'. Keep reasoning under 2 sentences."}
            ]
        }
    ]
    
    # Apply Template
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Process Inputs
    # Nudging the model to start with reasoning
    prompt_with_nudge = prompt + "Reasoning: "
    inputs = processor(text=prompt_with_nudge, images=image, return_tensors="pt").to(device)
    
    # Generate Response
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1  # SGP Standard
        )
    
    # Trim prompt from output to extract only the assistant's response
    # For SmolVLM, the generate_ids contains the full sequence.
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Clean up any leftover "Assistant:" or "Assistant: " prefix if the model generated it
    output_text = output_text.strip()
    if output_text.lower().startswith("assistant:"):
        output_text = output_text[len("assistant:"):].strip()
    
    # Prepend our nudge back to the output text so the CSV shows the full logical flow
    output_text = "Reasoning: " + output_text
    
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