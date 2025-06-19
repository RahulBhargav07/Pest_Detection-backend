from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import os
from ultralytics import YOLO
import clip
import torchvision.transforms as transforms
import urllib.request  # <-- Add this

app = FastAPI()

# === Ensure YOLO model file exists (download if not) ===
MODEL_PATH = "weights/best.pt"
if not os.path.exists(MODEL_PATH):
    os.makedirs("weights", exist_ok=True)
    print("Downloading model...")
    urllib.request.urlretrieve("https://drive.google.com/file/d/1v1uiDq-_iE7qFhrPGXCfd75B_QNKErO2/view?usp=sharing", MODEL_PATH)

# === Load YOLOv11 model ===
yolo_model = YOLO(MODEL_PATH)

# === Load CLIP model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
