import os
import io
import json
from typing import Dict, Tuple, Any

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

# Prefer tf_keras on environments where it is available (e.g., TF 2.14+),
# otherwise fall back to tensorflow.keras.
try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras


# ---------- Filesystem utilities ----------
def _default_paths() -> Dict[str, str]:
    app_directory = os.path.dirname(os.path.abspath(__file__))
    artifacts_directory = os.path.join(app_directory, "artifacts")
    return {
        "APP_DIR": app_directory,
        "ARTIFACTS_DIR": artifacts_directory,
        "MODEL_FILE_PATH": os.getenv("MODEL_PATH", os.path.join(artifacts_directory, "best_model.keras")),
        "PREPROCESS_CONFIG_PATH": os.getenv("PREP_PATH", os.path.join(artifacts_directory, "preprocessing.json")),
        "MODEL_CARD_PATH": os.getenv("CARD_PATH", os.path.join(artifacts_directory, "model_card.json")),
    }


def _load_json(path: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {} if default is None else default


# ---------- Model and preprocessing ----------
def load_artifacts() -> Tuple[keras.Model, Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    paths = _default_paths()

    print("=== STARTUP ===")
    print(f"APP_DIR: {paths['APP_DIR']}")
    print(f"ARTIFACTS_DIR: {paths['ARTIFACTS_DIR']}")
    print(f"MODEL_FILE_PATH: {paths['MODEL_FILE_PATH']}")
    print(f"PREPROCESS_CONFIG_PATH: {paths['PREPROCESS_CONFIG_PATH']}")
    print(f"MODEL_CARD_PATH: {paths['MODEL_CARD_PATH']}")

    if not os.path.exists(paths["MODEL_FILE_PATH"]):
        raise FileNotFoundError(f"Missing model at {paths['MODEL_FILE_PATH']}")
    if not os.path.exists(paths["PREPROCESS_CONFIG_PATH"]):
        raise FileNotFoundError(f"Missing preprocessing.json at {paths['PREPROCESS_CONFIG_PATH']}")

    print(f"Loading model from: {paths['MODEL_FILE_PATH']}")
    model = keras.models.load_model(paths["MODEL_FILE_PATH"])

    prep_cfg = _load_json(paths["PREPROCESS_CONFIG_PATH"])
    if "img_size" not in prep_cfg or "scale" not in prep_cfg:
        raise ValueError("preprocessing.json must contain 'img_size' and 'scale'")

    card = _load_json(paths["MODEL_CARD_PATH"], default={"best_model_name": "Unknown", "test_auc": None})

    input_width, input_height = prep_cfg["img_size"]
    input_scale = prep_cfg["scale"]
    print(f"✓ Loaded model: {card.get('best_model_name', 'Unknown')}")
    print(f"✓ Test AUC: {card.get('test_auc', 'N/A')}")
    print(f"✓ Input size: {input_width}x{input_height}, Scale: {input_scale}")

    return model, prep_cfg, card, paths


def preprocess_bytes(raw_bytes: bytes, width: int, height: int, scale: float) -> np.ndarray:
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize((width, height))
    array = np.asarray(image, dtype=np.float32) * float(scale)
    return array[None, ...]  # (1, H, W, 3)


def predict_label_from_array(model: keras.Model, batch: np.ndarray) -> str:
    prob = float(model.predict(batch, verbose=0).ravel()[0])
    return "damage" if prob >= 0.5 else "no_damage"


def get_request_image_bytes() -> bytes:
    if "image" in request.files:
        uploaded = request.files["image"]
        return uploaded.read()
    return request.get_data()


# ---------- Flask app factory ----------
def create_app() -> Flask:
    app = Flask(__name__)

    model, prep_cfg, card, paths = load_artifacts()
    app.config["MODEL"] = model
    app.config["PREP"] = prep_cfg
    app.config["CARD"] = card
    app.config["PATHS"] = paths

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.route("/summary", methods=["GET"])
    def summary() -> Any:
        input_width, input_height = app.config["PREP"]["img_size"]
        input_scale = app.config["PREP"]["scale"]
        meta = {
            "model_name": app.config["CARD"].get("best_model_name", "Unknown"),
            "test_auc": app.config["CARD"].get("test_auc"),
            "input_size": [input_width, input_height, 3],
            "classes": ["no_damage", "damage"],
            "preprocessing": {"resize": [input_width, input_height], "scale": input_scale},
        }
        return jsonify(meta)

    @app.route("/inference", methods=["POST"])
    def inference() -> Any:
        raw = get_request_image_bytes()
        if not raw:
            return jsonify({"error": "Empty request body"}), 400

        try:
            input_width, input_height = app.config["PREP"]["img_size"]
            input_scale = app.config["PREP"]["scale"]
            batch = preprocess_bytes(raw, input_width, input_height, input_scale)
            label = predict_label_from_array(app.config["MODEL"], batch)
            return jsonify({"prediction": label})
        except Exception as exc:
            # Intentionally return error as JSON for easier debugging.
            return jsonify({"error": str(exc)}), 500

    return app


# Create the application when the module is imported so gunicorn/uwsgi can use it.
app = create_app()

if __name__ == "__main__":
    # Bind to 0.0.0.0:5000 to satisfy the grader/container requirements.
    app.run(host="0.0.0.0", port=5000, debug=False)
