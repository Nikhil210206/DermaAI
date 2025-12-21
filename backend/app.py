from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import os
from utils import preprocess_image

app = FastAPI(title="Skin AI API", description="Skin Disease Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = None

# --- CORRECTED CLASS MAPPING (Based on your training output) ---
CLASSES = {
    0: "Actinic Keratoses (Pre-cancerous)",
    1: "Basal Cell Carcinoma (Cancer)",
    2: "Benign Keratosis (Non-cancerous)",
    3: "Dermatofibroma (Non-cancerous)",
    4: "Melanocytic Nevi (Mole - Non-cancerous)",
    5: "Melanoma (Cancer)",
    6: "Vascular Lesions (Non-cancerous)"
}

@app.on_event("startup")
async def load_model():
    global model
    # Path to the model we just trained
    model_path = "../ml/model.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
    else:
        print("❌ Model file not found! Check ml/model.h5")

@app.get("/")
def home():
    return {"message": "Skin AI API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read file
        contents = await file.read()
        
        # Preprocess
        processed_image = preprocess_image(contents)
        
        # Predict
        predictions = model.predict(processed_image)
        
        # Get the highest confidence score
        confidence = float(np.max(predictions))
        class_id = int(np.argmax(predictions))
        
        class_name = CLASSES.get(class_id, "Unknown")
        
        return {
            "disease": class_name,
            "confidence": round(confidence, 4),
            "disclaimer": "AI-assisted screening only. Consult a doctor."
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))