from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from preprocess import preprocess_audio
import numpy as np
import shutil
import tempfile
import os

app = FastAPI()

# CORS configuration (agar zaroorat ho)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model/cnn_lstm_mfcc.h5"

# Model load kar rahe hain app start hote hi
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def predict_audio(audio_path):
    features = preprocess_audio(audio_path)
    prediction = model.predict(features)
    score = float(prediction[0][0])
    label = "FAKE" if score > 0.5 else "REAL"
    confidence = score * 100 if score > 0.5 else (1 - score) * 100
    
    return {
        "prediction": label,
        "confidence": round(confidence, 2)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = None
    try:
        # 1. Uploaded audio file ko temporary file mein save karein
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # 2. Model prediction function ko call karein
        result = predict_audio(temp_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 3. Cleanup: Temporary file ko delete kar dein taaki space na bhare
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)