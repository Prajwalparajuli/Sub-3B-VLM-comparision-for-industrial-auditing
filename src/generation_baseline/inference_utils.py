import json
import csv
import os

def load_preprocessed_metadata(path="src/ingestion/clean_metadata.json"):
    """Loads the preprocessed metadata JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_standard_prompt(constraint):
    """
    The standardized 'Golden Prompt' for all models.
    Ensures that the comparison is 100% fair.
    """
    return (f"System: You are an industrial auditor. Examine this image and "
            f"determine if it violates the following safety constraint: {constraint}. "
            f"Provide a concise audit reasoning followed by a verdict [SAFE/UNSAFE].")

def save_results(results, model_name, iteration=None, out_dir="results/baseline"):
    """Saves the VLM responses to a CSV for final analysis."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if iteration is not None:
        filename = f"{model_name}_run_{iteration}_results.csv"
    else:
        filename = f"{model_name}_results.csv"
        
    output_path = os.path.join(out_dir, filename)
    
    if results:
        keys = results[0].keys()
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    print(f"Results for {model_name} saved to {output_path}")   

# Export the functions from this file to be used in other files
if __name__ == "__main__":
    # These are placeholders for illustration. Do not run this file directly.
    # To test, uncomment and provide valid path/data:
    # print(load_preprocessed_metadata("src/ingestion/clean_metadata.json"))
    # print(get_standard_prompt("No helmets allowed"))
    pass
