import os
from flask import Flask, request, jsonify
from PIL import Image
import io
import json
import numpy as np
import tf_keras as keras


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_FILE = os.path.join(ARTIFACTS_DIR, "best_model.keras")
PREPROCESSING_FILE = os.path.join(ARTIFACTS_DIR, "preprocessing.json")
MODEL_CARD_FILE = os.path.join(ARTIFACTS_DIR, "model_card.json")

app = Flask(__name__)

best_model = keras.models.load_model(MODEL_FILE)

with open(PREPROCESSING_FILE, "r") as ppfile:
    preprocessing_config = json.load(ppfile)

with open(MODEL_CARD_FILE, "r") as mcfile:
    model_metadata = json.load(mcfile)

WIDTH = preprocessing_config["img_size"][0]
HEIGHT = preprocessing_config["img_size"][1]
SCALE = preprocessing_config["scale"]


@app.route("/summary", methods=["GET"])
def get_summary():
    return jsonify({
        "model_name": model_metadata.get("best_model_name", "Unknown"),
        "test_auc": model_metadata.get("test_auc"),
        "input_size": [WIDTH, HEIGHT, 3],
        "classes": ["no_damage", "damage"],
        "preprocessing": {
            "resize": [WIDTH, HEIGHT],
            "scale": SCALE
        }
    })


@app.route("/inference", methods=["POST"])
def perform_inference():
    if 'image' in request.files:
        image_bytes = request.files['image'].read()
    else:
        image_bytes = request.get_data()
    
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized_image = pil_image.resize((WIDTH, HEIGHT))
    image_array = np.asarray(resized_image, dtype=np.float32) * float(SCALE)
    batch_input = image_array[None, ...]
    
    prediction_output = best_model.predict(batch_input, verbose=0)
    probability = float(prediction_output.ravel()[0])
    class_label = "no_damage" if probability >= 0.5 else "damage"
    
    return jsonify({"prediction": class_label})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
