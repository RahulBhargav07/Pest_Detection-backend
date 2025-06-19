from flask import Flask, request, jsonify, send_file
from PIL import Image
import base64
import os
from io import BytesIO

from main import create_annotated_image  # Import your function from your existing file

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Read base64 image and predictions from request JSON
        data = request.get_json()
        img_data = base64.b64decode(data['image_base64'])
        predictions = data['predictions']

        # Save temporary image
        image_path = "temp.jpg"
        with open(image_path, "wb") as f:
            f.write(img_data)

        # Annotate
        annotated_img = create_annotated_image(image_path, predictions)
        os.remove(image_path)  # Clean up

        if annotated_img:
            # Save to buffer and return base64
            buffer = BytesIO()
            annotated_img.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return jsonify({"status": "success", "annotated_image_base64": img_str})

        return jsonify({"status": "error", "message": "Annotation failed"}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def home():
    return "🦋 Insect Detection API Running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
