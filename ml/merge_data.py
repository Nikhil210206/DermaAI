import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Progress bar

# --- CONFIGURATION ---
BASE_DIR = "ml/data"
HAM_DIR = os.path.join(BASE_DIR, "ham10000")
COMMON_DIR = os.path.join(BASE_DIR, "common_diseases")
DEST_DIR = "ml/processed_data"

# HAM10000 Class Mapping (Short code -> Readable Name)
HAM_MAP = {
    'nv': 'Melanocytic_Nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign_Keratosis',
    'bcc': 'Basal_Cell_Carcinoma',
    'akiec': 'Actinic_Keratoses',
    'vasc': 'Vascular_Lesions',
    'df': 'Dermatofibroma'
}

def setup_directories():
    # Delete existing processed data to start fresh
    if os.path.exists(DEST_DIR):
        print("Cleaning old processed data...")
        shutil.rmtree(DEST_DIR)
    
    # Create Train/Val structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

def process_ham10000():
    print("\n--- Processing HAM10000 (Cancer Data) ---")
    metadata_path = os.path.join(HAM_DIR, "HAM10000_metadata.csv")
    
    if not os.path.exists(metadata_path):
        print("⚠️ HAM10000_metadata.csv not found! Skipping HAM10000.")
        return

    df = pd.read_csv(metadata_path)
    
    # Map all image IDs to their actual file paths
    image_paths = {}
    for root, _, files in os.walk(HAM_DIR):
        for file in files:
            if file.endswith('.jpg'):
                image_id = os.path.splitext(file)[0]
                image_paths[image_id] = os.path.join(root, file)
    
    print(f"Found {len(image_paths)} HAM10000 images.")

    # Split into Train/Val
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)

    def move_files(dataframe, split):
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Moving {split}"):
            img_id = row['image_id']
            label_code = row['dx']
            label_name = HAM_MAP.get(label_code, label_code)
            
            if img_id in image_paths:
                src = image_paths[img_id]
                dest_folder = os.path.join(DEST_DIR, split, label_name)
                os.makedirs(dest_folder, exist_ok=True)
                shutil.copy(src, os.path.join(dest_folder, f"{img_id}.jpg"))

    move_files(train_df, 'train')
    move_files(val_df, 'val')

def process_common_diseases():
    print("\n--- Processing Common Diseases ---")
    if not os.path.exists(COMMON_DIR):
        print("⚠️ No 'common_diseases' folder found. Skipping.")
        return

    # Get list of diseases (folders) user added
    diseases = [d for d in os.listdir(COMMON_DIR) if os.path.isdir(os.path.join(COMMON_DIR, d))]
    
    if not diseases:
        print("⚠️ Common diseases folder is empty!")
        return

    print(f"Found classes: {diseases}")

    for disease in diseases:
        disease_path = os.path.join(COMMON_DIR, disease)
        images = [f for f in os.listdir(disease_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Split 80/20
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        # Move Train
        for img in train_imgs:
            src = os.path.join(disease_path, img)
            dest_folder = os.path.join(DEST_DIR, 'train', disease)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(src, os.path.join(dest_folder, img))
            
        # Move Val
        for img in val_imgs:
            src = os.path.join(disease_path, img)
            dest_folder = os.path.join(DEST_DIR, 'val', disease)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(src, os.path.join(dest_folder, img))
            
    print(f"✅ Processed {len(diseases)} common diseases.")

if __name__ == "__main__":
    setup_directories()
    process_ham10000()
    process_common_diseases()
    print("\n🎉 MERGE COMPLETE! Ready to train.")
    print(f"Data is ready in: {DEST_DIR}")