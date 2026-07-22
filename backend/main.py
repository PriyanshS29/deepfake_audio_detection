from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model
from preprocess import preprocess_audio
import numpy as np
import shutil
import tempfile
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model/cnn_lstm_mfcc.h5"

# Load Model
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

# 1. Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        result = predict_audio(temp_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# 2. Serve React Frontend (Yeh part miss ho gaya tha!)
# Check if static folder exists, then mount assets if using Vite
if os.path.isdir("static"):
    # Agar Vite use kiya hai toh assets folder mount karna padta hai
    if os.path.isdir("static/assets"):
        app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

# Catch-all route to serve index.html for React Router / main page
@app.get("/{catchall:path}")
async def serve_react_app(catchall: str):
    file_path = os.path.join("static", catchall)
    # Agar specific file mangi hai aur wo exist karti hai (jaise .css, .js)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    # Warna default React index.html serve karo
    return FileResponse("static/index.html")