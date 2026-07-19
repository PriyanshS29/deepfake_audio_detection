from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from predict import predict_audio 

app = FastAPI()

# --- ABSOLUTE PATH CALCULATION ---
# Get the directory where main.py is currently located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the path to your static folder (assuming it's in the same folder as main.py)
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")
ASSETS_DIR = os.path.join(STATIC_DIR, "assets")

# DEBUG: Print to logs to help us verify paths
print(f"DEBUG: BASE_DIR is {BASE_DIR}")
print(f"DEBUG: Looking for index at {INDEX_PATH}")
print(f"DEBUG: Does index exist? {os.path.exists(INDEX_PATH)}")
# ---------------------------------

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Uploads
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount Assets
# This tells FastAPI to serve the 'assets' folder
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# Route for the root and all React-side routes
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    # If the path is for the API, return 404 (or handle as needed) so it doesn't try to serve index.html
    if full_path.startswith("predict"):
        return {"error": "Not Found"}
    
    return FileResponse(INDEX_PATH)

# Your API Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Add your existing logic here
    return {"message": "Predict endpoint working"}