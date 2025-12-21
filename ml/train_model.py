import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight
import numpy as np
import os

# --- CONFIGURATION ---
IMG_SIZE = 224
BATCH_SIZE = 16 
EPOCHS = 25
DATA_DIR = 'ml/processed_data/train' # Points to the MERGED data
VAL_DIR = 'ml/processed_data/val'
MODEL_SAVE_PATH = 'ml/model.h5'

def train():
    print(f"TensorFlow Version: {tf.__version__}")
    
    # 1. Generators
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )

    val_datagen = ImageDataGenerator() # No augmentation for validation

    print("Loading merged data...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # --- CRITICAL: CALCULATE CLASS WEIGHTS ---
    # This solves the imbalance problem (e.g. 6000 moles vs 500 acne)
    print("Calculating class weights to fix imbalance...")
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Weights: {class_weights_dict}")

    # 2. Model Setup (EfficientNetB1)
    base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Phase 1: Freeze Base
    base_model.trainable = False 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    print("\n--- Phase 1: Warming up ---")
    model.fit(train_generator, epochs=5, validation_data=val_generator, class_weight=class_weights_dict)

    # Phase 2: Fine Tuning
    print("\n--- Phase 2: Unfreezing for Max Accuracy ---")
    base_model.trainable = True
    
    # Unfreeze only the top 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights_dict  # <--- This is the magic key
    )
    
    print(f"✅ Training Complete. Best model saved to {MODEL_SAVE_PATH}")
    print("⬇️ COPY THIS MAPPING FOR YOUR BACKEND ⬇️")
    print(train_generator.class_indices)

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print("ERROR: Processed data not found. Run merge_data.py first!")
    else:
        train()