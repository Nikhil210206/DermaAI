import numpy as np
from PIL import Image
import io
import tensorflow as tf

def preprocess_image(image_bytes):
    """
    Preprocesses the image to match MobileNetV2 requirements.
    """
    # Open image from bytes
    img = Image.open(io.BytesIO(image_bytes))
    
    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize to 224x224 (Model Input Shape)
    img = img.resize((224, 224))
    
    # Convert to array
    img_array = np.array(img)
    
    # Expand dims to create batch: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess using MobileNetV2 standard (scales to -1 to 1)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return img_array