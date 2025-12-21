from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import os
from utils import preprocess_image

app = FastAPI(title="Skin AI API", description="Skin Disease Detection API")

# Enable CORS (Allows your frontend to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Global Variable
model = None

# Class Names (Must match training order)
CLASSES = {
    0: "Actinic Keratoses (akiec)",
    1: "Basal Cell Carcinoma (bcc)",
    2: "Benign Keratosis (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic Nevi (nv)",
    6: "Vascular Lesions (vasc)"
}

@app.on_event("startup")
async def load_model():
    global model
    model_path = "../ml/model.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    else:
        print("Model file not found! Please run train_model.py")

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
        confidence = float(np.max(predictions))
        class_id = int(np.argmax(predictions))
        class_name = CLASSES.get(class_id, "Unknown")
        
        return {
            "disease": class_name,
            "confidence": round(confidence, 4),
            "disclaimer": "AI-assisted screening only. Consult a doctor."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))