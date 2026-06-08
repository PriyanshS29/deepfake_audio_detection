from tensorflow.keras.models import load_model
from preprocess import preprocess_audio
import numpy as np

MODEL_PATH = "model/cnn_lstm_mfcc.h5"

model = load_model(MODEL_PATH)

def predict_audio(audio_path):

    features = preprocess_audio(audio_path)

    prediction = model.predict(features)

    score = float(prediction[0][0])

    label = "FAKE" if score > 0.5 else "REAL"

    confidence = score * 100 if score > 0.5 else (1-score)*100

    return {
        "prediction": label,
        "confidence": round(confidence,2)
    }