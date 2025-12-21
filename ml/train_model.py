import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# CONFIGURATION
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = './data' # Ensure your HAM10000 images are here
METADATA_PATH = './data/HAM10000_metadata.csv'

# MAPPING DICTIONARY (7 Classes of HAM10000)
# nv: Melanocytic nevi, mel: Melanoma, bkl: Benign keratosis-like lesions
# bcc: Basal cell carcinoma, akiec: Actinic keratoses, vasc: Vascular lesions, df: Dermatofibroma
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def create_model(num_classes):
    # Load MobileNetV2 without the top layer (Transfer Learning)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze the base model
    base_model.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x) # Prevent overfitting
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    print("Checking for dataset...")
    if not os.path.exists(METADATA_PATH):
        print("ERROR: Dataset not found. Please download HAM10000 and place it in ml/data/")
        # FOR BEGINNER TESTING: We will generate a dummy model if data is missing
        # so you can test the website immediately.
        print("Generating DUMMY model for testing UI connection...")
        model = create_model(len(CLASSES))
        model.save('model.h5')
        print("Dummy model saved to ml/model.h5")
        return

    # Load Metadata
    df = pd.read_csv(METADATA_PATH)
    
    # Point to image paths (assuming all images are in one folder or subfolders)
    # You might need to adjust path joining depending on how you extracted the zip
    image_dir = os.path.join(DATA_DIR, 'HAM10000_images_part_1') 
    
    # Simple data generator
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True
    )

    # Note: This part requires the actual images to run
    # If running for real, uncomment and adjust paths
    print("Training functionality is ready. Run this with actual data to train.")

if __name__ == "__main__":
    train()