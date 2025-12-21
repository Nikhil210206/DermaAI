from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import os
from utils import preprocess_image

app = FastAPI(title="DermaAI API", description="AI Skin Disease Detection")

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

# --- ⚠️ FINAL TRAINED MAPPING (Copied from your terminal) ---
CLASSES = {
    0: "Acne",
    1: "Actinic Keratoses (Pre-cancerous)",
    2: "Basal Cell Carcinoma (Cancer)",
    3: "Benign Keratosis (Non-cancerous)",
    4: "Dermatofibroma (Non-cancerous)",
    5: "Eczema",
    6: "Melanocytic Nevi (Mole)",
    7: "Melanoma (Cancer)",
    8: "Nail Fungus",
    9: "Psoriasis",
    10: "Ringworm (Fungal)",
    11: "Vascular Lesions"
}

@app.on_event("startup")
async def load_model():
    global model
    # Load the trained model
    model_path = "../ml/model.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("✅ High-Accuracy Model Loaded Successfully!")
    else:
        print("❌ Error: ml/model.h5 not found. Did training finish?")

@app.get("/")
def home():
    return {"message": "DermaAI API is active"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model is loading... please wait.")
    
    try:
        # 1. Read Image
        contents = await file.read()
        
        # 2. Preprocess (same as training)
        processed_image = preprocess_image(contents)
        
        # 3. Predict
        predictions = model.predict(processed_image)
        
        # 4. Get Top Result
        top_index = int(np.argmax(predictions[0]))
        top_confidence = float(predictions[0][top_index])
        top_disease = CLASSES.get(top_index, "Unknown")
        
        # 5. Get Alternatives (Second & Third best guess)
        # This helps if the AI is unsure (e.g., 55% Acne, 40% Rosacea)
        sorted_indices = np.argsort(predictions[0])[::-1]
        alternatives = []
        for i in range(1, 3): # Get 2nd and 3rd best
            idx = sorted_indices[i]
            score = float(predictions[0][idx])
            if score > 0.05: # Only show if score is significant (>5%)
                alternatives.append({
                    "disease": CLASSES.get(idx, "Unknown"),
                    "probability": f"{round(score * 100, 1)}%"
                })

        return {
            "disease": top_disease,
            "confidence": round(top_confidence, 4),
            "alternatives": alternatives,
            "disclaimer": "AI Analysis Only. Consult a Dermatologist."
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))