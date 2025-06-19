from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import uvicorn
import clip
from ultralytics import YOLO
import torchvision.transforms as transforms

app = FastAPI()

# Load YOLOv11 model
yolo_model = YOLO("weights/best.pt")  # Ensure this path exists in Render build

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# Text labels
text_labels = [
    "a photo of a stem borer insect pest",
    "a photo of a pink bollworm moth pest", 
    "a photo of a leaf folder caterpillar pest",
    "a photo of a healthy green plant leaf"
]
text_tokens = clip.tokenize(text_labels).to(device)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run YOLO detection
        results = yolo_model.predict(image)
        detections = results[0].boxes

        response_data = []

        if detections is not None:
            for i, box in enumerate(detections):
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Only process confident detections
                if conf > 0.3:
                    cropped = image.crop((x1, y1, x2, y2))
                    clip_input = preprocess(cropped).unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_feat = clip_model.encode_image(clip_input)
                        text_feat = clip_model.encode_text(text_tokens)

                        image_feat /= image_feat.norm(dim=-1, keepdim=True)
                        text_feat /= text_feat.norm(dim=-1, keepdim=True)

                        similarity = (100.0 * image_feat @ text_feat.T).softmax(dim=-1)
                        label_idx = similarity.argmax().item()
                        label_score = similarity[0, label_idx].item()
                        label = text_labels[label_idx]

                        result = {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "yolo_conf": round(conf, 2),
                            "clip_conf": round(label_score, 2),
                            "label": label
                        }
                        response_data.append(result)

        return JSONResponse(content={"detections": response_data})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
