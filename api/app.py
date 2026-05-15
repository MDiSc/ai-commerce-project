import os
import sys
import joblib
import numpy as np
from flask import Flask, jsonify

# ─────────────────────────────────────────────
# Bootstrap — load model artifacts
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "pipeline", "model")

try:
    model    = joblib.load(os.path.join(MODEL_DIR, "mlp_model.pkl"))
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
except Exception as e:
    print(f"[API] Error loading artifacts: {e}")
    sys.exit(1)

FEATURE_COLS  = encoders["feature_cols"]
MONTH_ORDER   = encoders["month_order"]
month_enc     = encoders["month_encoder"]
visitor_enc   = encoders["visitor_encoder"]
THRESHOLD     = encoders.get("threshold", 0.5)

# ─────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _build_message(probability: float, classification: str) -> str:
    pct = round(probability * 100)
    if classification == "purchase":
        if probability >= 0.80: level = "muy probable"
        elif probability >= 0.60: level = "bastante probable"
        else: level = "posible"
    else:
        if probability >= 0.40: level = "algo posible"
        elif probability >= 0.20: level = "poco probable"
        else: level = "muy improbable"
    return f"El usuario presenta un {pct}% de probabilidades de hacer la compra, lo que lo hace {level}."

def _encode_and_scale(data: dict) -> np.ndarray:
    month_val = data.get("Month", "Nov")
    month_encoded = month_enc.transform([[month_val]])[0][0]
    
    visitor_val = data.get("VisitorType", "Returning_Visitor")
    visitor_encoded = visitor_enc.transform([visitor_val])[0]
    
    weekend_val = data.get("Weekend", False)
    if isinstance(weekend_val, str):
        weekend_val = weekend_val.lower() in ("true", "1", "yes")
    weekend_encoded = int(bool(weekend_val))

    row = [
        float(data.get("Administrative", 0)),
        float(data.get("Administrative_Duration", 0.0)),
        float(data.get("Informational", 0)),
        float(data.get("Informational_Duration", 0.0)),
        float(data.get("ProductRelated", 0)),
        float(data.get("ProductRelated_Duration", 0.0)),
        float(data.get("BounceRates", 0.0)),
        float(data.get("ExitRates", 0.0)),
        float(data.get("PageValues", 0.0)),
        float(data.get("SpecialDay", 0.0)),
        float(month_encoded),
        float(data.get("OperatingSystems", 2)),
        float(data.get("Browser", 2)),
        float(data.get("Region", 1)),
        float(data.get("TrafficType", 2)),
        float(visitor_encoded),
        float(weekend_encoded),
    ]
    X_scaled = scaler.transform(np.array(row).reshape(1, -1))
    return X_scaled

# ─────────────────────────────────────────────
# /health endpoint
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": type(model).__name__}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)