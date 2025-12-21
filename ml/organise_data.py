import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# Where your raw files are now:
DATA_DIR = "data" 
METADATA_FILE = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

# Where we want to move them:
DEST_DIR = "ml/processed_data"

# Map the short codes to full names for folder creation
LABEL_MAP = {
    'nv': 'Melanocytic_nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign_keratosis',
    'bcc': 'Basal_cell_carcinoma',
    'akiec': 'Actinic_keratoses',
    'vasc': 'Vascular_lesions',
    'df': 'Dermatofibroma'
}

def organize():
    print("Reading metadata...")
    if not os.path.exists(METADATA_FILE):
        print(f"ERROR: Could not find {METADATA_FILE}")
        print("Please ensure HAM10000_metadata.csv is inside ml/data/")
        return

    df = pd.read_csv(METADATA_FILE)
    
    # Create the training/validation folder structure
    for split in ['train', 'val']:
        for label in LABEL_MAP.values():
            os.makedirs(os.path.join(DEST_DIR, split, label), exist_ok=True)

    # Find all .jpg files inside ml/data (handling Part 1 & Part 2 folders automatically)
    print("Locating all images...")
    image_paths = {}
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".jpg"):
                image_id = os.path.splitext(file)[0]
                image_paths[image_id] = os.path.join(root, file)

    print(f"Found {len(image_paths)} images.")

    # Split data: 80% Training, 20% Validation
    # We balance the split based on the disease type (stratify)
    y = df['dx']
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=y)

    def copy_images(dataframe, split_name):
        print(f"Copying {split_name} images...")
        count = 0
        for _, row in dataframe.iterrows():
            img_id = row['image_id']
            label_code = row['dx']
            label_name = LABEL_MAP[label_code]
            
            if img_id in image_paths:
                src = image_paths[img_id]
                dst = os.path.join(DEST_DIR, split_name, label_name, f"{img_id}.jpg")
                shutil.copy2(src, dst)
                count += 1
        print(f"Moved {count} images to {split_name}.")

    copy_images(df_train, 'train')
    copy_images(df_val, 'val')
    print("\n✅ Data organization complete! Check ml/processed_data folder.")

if __name__ == "__main__":
    organize()