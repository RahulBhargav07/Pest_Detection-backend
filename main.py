from fastapi import FastAPI, WebSocket
import base64, cv2, torch, numpy as np
from PIL import Image
import io
from ultralytics import YOLO
import clip

app = FastAPI()

# Load models (CPU mode—Render Starter plan)
device = "cpu"
yolo_model = YOLO("best.pt")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Define pest labels
text_labels = [
    "a photo of a stem borer insect pest",
    "a photo of a pink bollworm moth pest",
    "a photo of a leaf folder caterpillar pest",
    "a photo of a healthy green plant leaf"
]
text_tokens = clip.tokenize(text_labels).to(device)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            msg = await ws.receive_text()
        except:
            break  # Connection closed

        if not msg.startswith("data:image"):
            await ws.send_json({"error": "invalid frame"})
            continue

        # Decode incoming frame
        header, data = msg.split(",", 1)
        image_bytes = base64.b64decode(data)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Run YOLO
        results = yolo_model(frame)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            crop = pil_img.crop((x1, y1, x2, y2))
            clip_input = clip_preprocess(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                img_f = clip_model.encode_image(clip_input)
                text_f = clip_model.encode_text(text_tokens)
                img_f /= img_f.norm(dim=-1, keepdim=True)
                text_f /= text_f.norm(dim=-1, keepdim=True)

                sim = (100.0 * img_f @ text_f.T).softmax(dim=-1)
                idx = sim.argmax().item()
                label = text_labels[idx].split(" a photo of a ")[-1].split(" pest")[0]
                score = float(sim[0, idx].cpu().numpy())

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "yolo_confidence": round(conf, 2),
                "clip_label": label,
                "clip_score": round(score, 2)
            })

        await ws.send_json({"detections": detections})

    await ws.close()

@app.get("/")
def root():
    return {"message": "Pest detection WebSocket server is running"}
