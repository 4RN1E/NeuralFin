import numpy as np
from PIL import Image
import io
import torch

def process_image(image_bytes: bytes):
    """
    Process the uploaded image for AI model input.

    Args:
    - image_bytes: Bytes of the uploaded image

    Returns: 
    - Processed image or None if processing fails
    """
    try:
        # Open the image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize image (adjust size as needed)
        image = image.resize((224, 224))

        # Convert to numpy array
        image_array = np.array(image)

        # Normalize pixel values (optional, depends on your AI model)
        image_array = image_array / 255.0

        # Convert image to tensor (CHW format for PyTorch)
        image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1)  # Convert to CHW

        return image_tensor

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def validate_image(image_bytes: bytes) -> bool:
    """
    Validate if the uploaded file is a valid image.

    Args: 
    - image_bytes: Bytes of the uploaded file

    Returns: 
    - Boolean indicating image validity
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # Verifies the image is not corrupt
        return True
    except Exception as e:
        print(f"Invalid image file: {e}")
        return False