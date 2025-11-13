import os
# Set environment variable before any TF imports
#os.environ['TF_USE_LEGACY_KERAS'] = '1'

from flask import Flask, request, jsonify
from PIL import Image
import io, json, numpy as np

# Use tf_keras instead of keras if using TF 2.14+
try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

# === Paths ===
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(APP_DIR, "artifacts")
MODEL_PATH = os.path.join(ART_DIR, "best_model.keras")
PREP_PATH = os.path.join(ART_DIR, "preprocessing.json")
CARD_PATH = os.path.join(ART_DIR, "model_card.json")

app = Flask(__name__)

# === Startup validation ===
print("=== STARTUP ===")
print(f"APP_DIR: {APP_DIR}")
print(f"ART_DIR: {ART_DIR}")
print(f"SavedModel present?: {os.path.exists(MODEL_PATH)}")
print(f"PREP present?: {os.path.exists(PREP_PATH)}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model at {MODEL_PATH}")
if not os.path.exists(PREP_PATH):
    raise FileNotFoundError(f"Missing preprocessing.json at {PREP_PATH}")

# === Load model and config ===
print(f"Loading model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)

with open(PREP_PATH, "r") as f:
    prep = json.load(f)
W, H = prep["img_size"]
SCALE = prep["scale"]

try:
    with open(CARD_PATH, "r") as f:
        card = json.load(f)
except FileNotFoundError:
    card = {"best_model_name": "Unknown", "test_auc": None}

print(f"✓ Loaded model: {card.get('best_model_name', 'Unknown')}")
print(f"✓ Test AUC: {card.get('test_auc', 'N/A')}")
print(f"✓ Input size: {W}x{H}, Scale: {SCALE}")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@app.route("/summary", methods=["GET"])
def summary():
    """Return model metadata"""
    return jsonify({
        "model_name": card.get("best_model_name", "Unknown"),
        "test_auc": card.get("test_auc"),
        "input_size": [W, H, 3],
        "classes": ["no_damage", "damage"],
        "preprocessing": {
            "resize": [W, H],
            "scale": SCALE
        }
    })

@app.route("/inference", methods=["POST"])
def inference():
    """Perform inference on uploaded image"""
    
    # Handle multipart form data (grader format)
    if 'image' in request.files:
        file = request.files['image']
        raw = file.read()
    # Handle raw binary data (alternative format)
    else:
        raw = request.get_data()
    
    if not raw:
        return jsonify({"error": "Empty request body"}), 400
    
    try:
        # Load and preprocess
        im = Image.open(io.BytesIO(raw)).convert("RGB").resize((W, H))
        x = np.asarray(im, dtype=np.float32) * float(SCALE)
        x = x[None, ...]  # (1, H, W, 3)
        
        # Predict
        prob = float(model.predict(x, verbose=0).ravel()[0])
        label = "no_damage" if prob >= 0.5 else "damage"
        
        # Return exact format required by grader
        return jsonify({"prediction": label})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)