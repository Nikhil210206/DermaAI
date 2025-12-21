import numpy as np
from PIL import Image
import io
import tensorflow as tf

def preprocess_image(image_bytes):
    """
    Preprocesses the image for EfficientNetB1.
    """
    # 1. Open Image
    img = Image.open(io.BytesIO(image_bytes))
    
    # 2. Ensure RGB (removes Alpha channels if any)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # 3. Resize to 224x224
    img = img.resize((224, 224))
    
    # 4. Convert to Array & Float32
    # IMPORTANT: EfficientNet expects 0-255, but as Floats, not Integers.
    img_array = np.array(img).astype(np.float32)
    
    # 5. Expand dims (Create batch of 1)
    # Shape becomes: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 6. Preprocessing
    # EfficientNetB1 includes its own scaling layers, so we usually pass raw 0-255 float data.
    # However, to be 100% safe matching Keras defaults:
    processed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    return processed_img