from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import requests

app = FastAPI()

# Helper function to annotate image
def create_annotated_image(image_data, predictions):
    try:
        image = Image.open(BytesIO(image_data))
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for pred in predictions:
            center_x = pred['x']
            center_y = pred['y']
            width = pred['width']
            height = pred['height']
            left = center_x - width / 2
            top = center_y - height / 2
            right = center_x + width / 2
            bottom = center_y + height / 2

            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            label = f"{pred['class']}: {pred['confidence']:.2f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle([left, top - text_height - 10, left + text_width + 10, top], fill="red")
            draw.text((left + 5, top - text_height - 5), label, fill="white", font=font)

        return image
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img_b64 = base64.b64encode(image_bytes).decode('utf-8')

        api_key = "NVfp8h9atJEAWzsw1eZ0"
        model_id = "insect-identification-rweyy/7"
        url = f"https://detect.roboflow.com/{model_id}"
        params = {
            "api_key": api_key,
            "confidence": 0.4,
            "overlap": 0.3,
            "format": "json"
        }

        response = requests.post(url, params=params, data=img_b64,
                                 headers={"Content-Type": "application/json"})

        result = response.json()
        predictions = result.get("predictions", [])
        if not predictions:
            return {"status": "error", "message": "No insects detected"}

        image = create_annotated_image(image_bytes, predictions)
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        annotated_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            "status": "success",
            "detections": predictions,
            "annotated_image_base64": annotated_base64
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
