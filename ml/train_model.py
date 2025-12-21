import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# --- CONFIGURATION ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15  # Increased for better accuracy
TRAIN_DIR = 'ml/processed_data/train'
VAL_DIR = 'ml/processed_data/val'
MODEL_SAVE_PATH = 'ml/model.h5'

def train():
    print(f"TensorFlow Version: {tf.__version__}")
    
    # 1. Data Generators (Augmentation for Training)
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    print("Loading data from folders...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
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

    # 2. Build Model (Transfer Learning)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze base model to keep pre-trained knowledge
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)
    # The output layer must match the number of classes (7 for HAM10000)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3. Compile
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Callbacks (Save Best Model Only)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, 
                                 monitor='val_accuracy', 
                                 save_best_only=True, 
                                 mode='max', 
                                 verbose=1)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 5. Train
    print("Starting training... (This will take time)")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    print(f"✅ Training Complete. Best model saved to {MODEL_SAVE_PATH}")

    # Optional: Save Class Indices so Backend knows which ID is which Disease
    print("Class Mapping:", train_generator.class_indices)

if __name__ == "__main__":
    if not os.path.exists(TRAIN_DIR):
        print("ERROR: Processed data not found. Please run organize_data.py first.")
    else:
        train()