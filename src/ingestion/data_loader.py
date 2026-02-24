import pandas as pd
from pathlib import Path

# Defining the function to load data
def load_dataset_from_metadata(base_path, excel_filename = "data_label_constraint_image.xlsx"):
    """
    Loads the Excel metadata and finds the images in the folder
    """

    base_folder = Path(base_path)
    metadata_path = base_folder / excel_filename

    # Loading the excel metadata file
    if not metadata_path.exists():
        print(f"Metadata excel file not found at {metadata_path}, please check the path")
        return None
    
    print(f"Loading metadata from {metadata_path}")
    df = pd.read_excel(metadata_path)

    # Index all the images in the folders
    print(f"Looking for images in {base_path}") 
    image_locations = {}
    # Use glob patterns (e.g. *.jpg) instead of just extensions
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"]

    for pattern in patterns:
        for img_path in base_folder.rglob(pattern):
            # Store file names and absolute paths as dictionary
            image_locations[img_path.name] = str(img_path.absolute())

    # Match the excel rows to the actual file paths
    final_data_list = []
    for index, row in df.iterrows():

        # Get the image file names from excel
        img_id = str(row['image_id'])

        # Create a dictionary of this row and add the full path
        if img_id in image_locations:
            data_entry = row.to_dict()
            data_entry['full_path'] = image_locations[img_id]
            final_data_list.append(data_entry)
        else:
            print(f"Image {img_id} not found in the folder")

    print(f"Found {len(final_data_list)} images total")
    return final_data_list

# Entry point for data_loader.py to read the data from the metadata file
if __name__ == "__main__":

    # Call the function to load the data
    dataset = load_dataset_from_metadata("Dataset")

    if dataset is not None:
        print("Data loaded successfully")
        print(f"Example image: {dataset[1]['full_path']}")
        print(f"Found {len(dataset)} images")
    elif len(dataset) != 200:
        print(f"Found {len(dataset)} images, expected 200, missing {200 - len(dataset)} images")
    else:
        print("Data loaded successfully")
        

    