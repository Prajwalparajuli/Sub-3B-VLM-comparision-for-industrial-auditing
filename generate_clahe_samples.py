import os
import sys
from PIL import Image

# Add src to path to import image_utils
sys.path.insert(0, os.path.abspath('src/generation_contrast'))
from image_utils import apply_clahe_and_concatenate

import json
with open('src/ingestion/clean_metadata.json', 'r') as f:
    metadata = json.load(f)

# Find specific low contrast images for tuning demo
import random

# Find specific low contrast images for tuning demo
low_contrast_gauges = [item for item in metadata if item['category'] == 'guage' and ('low contrast' in str(item.get('artifact_tag', '')).lower() or 'glare' in str(item.get('artifact_tag', '')).lower())]
low_contrast_pipes = [item for item in metadata if item['category'] == 'pipeline' and ('low contrast' in str(item.get('artifact_tag', '')).lower() or 'texture overlap' in str(item.get('artifact_tag', '')).lower())]

# Take a slice of 3 uniquely different images (ignoring metadata duplicates for rules)
def get_unique_items(items):
    unique = []
    seen = set()
    for item in items:
        if item['processed_path'] not in seen:
            unique.append(item)
            seen.add(item['processed_path'])
    return unique

unique_gauges = get_unique_items(low_contrast_gauges)
unique_pipes = get_unique_items(low_contrast_pipes)

# Select random distinct samples to ensure they are new to the user
gauge_samples = random.sample(unique_gauges, min(3, len(unique_gauges)))
pipe_samples = random.sample(unique_pipes, min(3, len(unique_pipes)))

out_dir = 'results/sample_images/tuning'
os.makedirs(out_dir, exist_ok=True)

def generate_tuning_samples(item, name, idx):
    img_path = item['processed_path']
    img = Image.open(img_path).convert("RGB")
    
    # Save raw original for baseline reference
    img.save(os.path.join(out_dir, f"{name}_{idx}_raw.jpg"), "JPEG", quality=95)
    
    # Sweep clip limits
    for clip in [3.0, 6.0, 10.0]:
        processed_img = apply_clahe_and_concatenate(img, clip_limit=clip)
        save_path = os.path.join(out_dir, f"{name}_{idx}_clahe_{clip}.jpg")
        processed_img.save(save_path, "JPEG", quality=95)
        print(f"Saved tuned sample to {save_path}")

for idx, g in enumerate(gauge_samples):
    generate_tuning_samples(g, "gauge", idx)

for idx, p in enumerate(pipe_samples):
    generate_tuning_samples(p, "pipe", idx)

print("Tuning parameter sweep complete.")
