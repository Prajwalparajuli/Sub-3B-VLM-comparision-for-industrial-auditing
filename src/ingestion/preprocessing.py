import os
import json
import pandas as pd
from PIL import Image   
from data_loader import load_dataset_from_metadata

def run_preprocessing():
    """
    Create a 'Source of truth' JSON file and standardizing RGB images.
    This script ensures that the actual VLM inference scripts do not need to read from the excel file.
    """

    # Load the dataset using the data_loader.py
    dataset = load_dataset_from_metadata("Dataset")
    if not dataset:
        print("Error! Could not find the Dataset folder or the excel file, please check the path of the Dataset")
        return None

    # Create a folder with the 'preprocessed' images
    output_dir = "Data_Preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    # Create a list to store the preprocessed data
    preprocessed_data = []
    print(f"Starting Preprocessing of {len(dataset)} images")

    for item in dataset:
        img_id = item['image_id']
        img_path = item['full_path']
        # Standardizing the image to RGB format
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                save_path = os.path.join(output_dir, img_id)
                img.save(save_path)

            # Copy the metadata from the excel file
            entry = item.copy()
            entry['processed_path'] = save_path.replace("\\", "/") # Use forward slashes for cross-platform compatibility
            entry['constraint'] = entry.get('artifact_tag', 'Safety Violation')

            # Handle 'NaN' for JSON compatibility
            for key, value in entry.items():
                if pd.isna(value):
                    entry[key] = None
            
            preprocessed_data.append(entry)
            print(f"Sucessfully processed {img_id}")
        except Exception as e:
            print(f"Error processing {img_id}: {e}")

    # Save the preprocessed data to a JSON file
    output_json_path = "src/ingestion/clean_metadata.json"
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(preprocessed_data, f, indent=4)

    print(f"Preprocessing complete. Saved {len(preprocessed_data)} entries to {output_json_path}")
    return output_json_path

# Entry point for preprocessing.py
if __name__ == "__main__":
    run_preprocessing()