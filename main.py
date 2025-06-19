from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64

app = FastAPI()

class Prediction(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str

class AnnotateRequest(BaseModel):
    image_base64: str
    predictions: List[Prediction]

def create_annotated_image(image_data, predictions):
    try:
        image = Image.open(BytesIO(image_data))
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for pred in predictions:
            center_x = pred.x
            center_y = pred.y
            width = pred.width
            height = pred.height
            left = center_x - width / 2
            top = center_y - height / 2
            right = center_x + width / 2
            bottom = center_y + height / 2

            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            label = f"{pred.class_name}: {pred.confidence:.2f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle([left, top - text_height - 10, left + text_width + 10, top], fill="red", outline="red")
            draw.text((left + 5, top - text_height - 5), label, fill="white", font=font)

        return image

    except Exception as e:
        print(f"❌ Error creating annotated image: {e}")
        return None

@app.get("/")
def home():
    return {"message": "🦋 Insect Detection API is running"}

@app.post("/annotate")
def annotate(data: AnnotateRequest):
    try:
        img_bytes = base64.b64decode(data.image_base64)
        predictions = data.predictions
        annotated_img = create_annotated_image(img_bytes, predictions)

        if not annotated_img:
            return {"status": "error", "message": "Failed to create image"}

        buffer = BytesIO()
        annotated_img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"status": "success", "annotated_image_base64": img_str}

    except Exception as e:
        return {"status": "error", "message": str(e)}
