import cv2
import numpy as np
from PIL import Image

def apply_clahe_and_concatenate(image: Image.Image, max_dim: int = 512, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)) -> Image.Image:
    """
    Applies CLAHE to the L channel of the LAB color space to enhance contrast
    without shifting colors, then horizontally concatenates the original image
    with the enhanced image to create a "Dual-Channel" input.
    
    Args:
        image: PIL.Image (RGB)
        max_dim: the maximum edge length for the *concatenated* final image. 
                 The function scales the source images down so that 
                 Width_concat <= max_dim and Height <= max_dim.
        clip_limit: threshold for contrast limiting in CLAHE
        tile_grid_size: size of the grid for histogram equalization
        
    Returns:
        PIL.Image (RGB) representing the dual-channel concatenated image.
    """
    # 1. Convert PIL to cv2 (BGR)
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 2. Convert to LAB color space
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 3. Apply CLAHE only to the Lightness (L) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    
    # 4. Merge back and convert to RGB
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert back to PIL
    enhanced_pil = Image.fromarray(enhanced_rgb)
    
    # 5. Concatenate horizontally
    # width of concatenated image is 2 * width of original
    concat_width = image.width + enhanced_pil.width
    concat_height = max(image.height, enhanced_pil.height)
    
    concatenated = Image.new('RGB', (concat_width, concat_height))
    concatenated.paste(image, (0, 0))
    concatenated.paste(enhanced_pil, (image.width, 0))
    
    # 6. Resize to meet max_dim constraints (VRAM safety)
    # We want max(concat_width, concat_height) <= max_dim
    # maintain aspect ratio
    ratio = min(max_dim / concat_width, max_dim / concat_height)
    
    if ratio < 1.0:
        new_width = int(concat_width * ratio)
        new_height = int(concat_height * ratio)
        concatenated = concatenated.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    return concatenated
