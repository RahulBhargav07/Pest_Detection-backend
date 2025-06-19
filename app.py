from flask import Flask, request, jsonify
from main import create_annotated_image
import base64
from io import BytesIO

app = Flask(__name__)

@app.route("/")
def home():
    return "🦋 Insect Detection Flask API running"

@app.route("/annotate", methods=["POST"])
def annotate():
    try:
        data = request.get_json()
        image_b64 = data["image_base64"]
        predictions = data["predictions"]

        image_bytes = base64.b64decode(image_b64)
        with open("temp.jpg", "wb") as f:
            f.write(image_bytes)

        annotated = create_annotated_image("temp.jpg", predictions)
        if not annotated:
            return jsonify({"error": "Annotation failed"}), 500

        buffer = BytesIO()
        annotated.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return jsonify({"annotated_image_base64": encoded})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
