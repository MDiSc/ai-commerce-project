"""
E-commerce Purchase Intention — Flask REST API & Dashboard
==========================================================
Endpoints:
  GET  /            → Interactive Premium Dashboard
  GET  /health      → Health check
  POST /predict     → Prediction endpoint

Response contract (POST /predict):
  {
    "classification": "purchase" | "no_purchase",
    "probability":    float (0.0–1.0),
    "message":        str   (human-readable)
  }
"""

import os
import sys
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

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
# HTML Dashboard Template (Premium Design)
# ─────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E-commerce Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --accent: #38bdf8;
            --text: #f1f5f9;
            --success: #4ade80;
            --warning: #fbbf24;
            --danger: #f87171;
        }
        * { box-sizing: border-box; font-family: 'Outfit', sans-serif; }
        body { 
            background-color: var(--bg); 
            color: var(--text); 
            margin: 0; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            padding: 40px 20px;
        }
        .container { max-width: 1000px; width: 100%; }
        header { text-align: center; margin-bottom: 40px; }
        h1 { font-weight: 600; margin: 0; color: var(--accent); letter-spacing: -1px; font-size: 2.5rem; }
        p.subtitle { opacity: 0.7; margin-top: 8px; font-size: 1.1rem; }
        
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
            gap: 20px; 
            background: var(--card);
            padding: 40px;
            border-radius: 32px;
            box-shadow: 0 25px 60px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.05);
        }
        .field { display: flex; flex-direction: column; gap: 10px; }
        label { font-size: 0.75rem; font-weight: 600; opacity: 0.6; text-transform: uppercase; letter-spacing: 1.5px; }
        input, select {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(255,255,255,0.1);
            padding: 14px;
            border-radius: 14px;
            color: white;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        input:focus, select:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.2); }
        
        button {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, var(--accent), #0ea5e9);
            color: var(--bg);
            border: none;
            padding: 18px;
            border-radius: 16px;
            font-size: 1.2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin-top: 20px;
            box-shadow: 0 10px 20px rgba(56, 189, 248, 0.2);
        }
        button:hover { transform: translateY(-3px); box-shadow: 0 15px 30px rgba(56, 189, 248, 0.4); filter: brightness(1.1); }
        button:active { transform: translateY(-1px); }

        #result-card {
            margin-top: 40px;
            padding: 40px;
            border-radius: 32px;
            background: rgba(56, 189, 248, 0.05);
            border: 1px solid rgba(56, 189, 248, 0.3);
            display: none;
            animation: fadeInScale 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes fadeInScale { 
            from { opacity: 0; transform: scale(0.95) translateY(30px); } 
            to { opacity: 1; transform: scale(1) translateY(0); } 
        }
        
        .res-header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 25px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 20px; }
        .prob-badge { font-size: 3rem; font-weight: 600; line-height: 1; }
        .res-status { font-size: 1.2rem; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; }
        .msg-text { font-size: 1.4rem; line-height: 1.5; font-weight: 300; }
        
        .raw-container { margin-top: 30px; }
        .raw-label { font-size: 0.7rem; opacity: 0.4; text-transform: uppercase; margin-bottom: 10px; display: block; }
        pre { background: rgba(0,0,0,0.4); padding: 20px; border-radius: 16px; font-size: 0.85rem; overflow-x: auto; color: var(--accent); border: 1px solid rgba(255,255,255,0.05); }

        @media (max-width: 600px) {
            .grid { padding: 25px; }
            h1 { font-size: 1.8rem; }
            .prob-badge { font-size: 2.2rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>IA Purchase Predictor</h1>
            <p class="subtitle">Análisis Predictivo de Intención de Compra en Tiempo Real</p>
        </header>

        <form id="predict-form" class="grid">
            <div class="field">
                <label>Administrativas</label>
                <input type="number" name="Administrative" value="2" min="0">
            </div>
            <div class="field">
                <label>Duración Admin (s)</label>
                <input type="number" step="0.1" name="Administrative_Duration" value="30.5" min="0">
            </div>
            <div class="field">
                <label>Informativas</label>
                <input type="number" name="Informational" value="0" min="0">
            </div>
            <div class="field">
                <label>Duración Info (s)</label>
                <input type="number" step="0.1" name="Informational_Duration" value="0.0" min="0">
            </div>
            <div class="field">
                <label>Productos</label>
                <input type="number" name="ProductRelated" value="12" min="0">
            </div>
            <div class="field">
                <label>Duración Productos (s)</label>
                <input type="number" step="0.1" name="ProductRelated_Duration" value="450.0" min="0">
            </div>
            <div class="field">
                <label>Tasa Rebote</label>
                <input type="number" step="0.001" name="BounceRates" value="0.01" min="0" max="1">
            </div>
            <div class="field">
                <label>Tasa Salida</label>
                <input type="number" step="0.001" name="ExitRates" value="0.02" min="0" max="1">
            </div>
            <div class="field">
                <label>Valor de Página</label>
                <input type="number" step="0.1" name="PageValues" value="35.5" min="0">
            </div>
            <div class="field">
                <label>Día Especial</label>
                <input type="number" step="0.1" name="SpecialDay" value="0.0" min="0" max="1">
            </div>
            <div class="field">
                <label>Mes</label>
                <select name="Month">
                    <option value="Feb">Febrero</option><option value="Mar">Marzo</option>
                    <option value="May">Mayo</option><option value="Jun">Junio</option>
                    <option value="Jul">Julio</option><option value="Aug">Agosto</option>
                    <option value="Sep">Septiembre</option><option value="Oct">Octubre</option>
                    <option value="Nov" selected>Noviembre</option><option value="Dec">Diciembre</option>
                </select>
            </div>
            <div class="field">
                <label>Sistema Operativo</label>
                <input type="number" name="OperatingSystems" value="2" min="1" max="8">
            </div>
            <div class="field">
                <label>Navegador</label>
                <input type="number" name="Browser" value="2" min="1" max="13">
            </div>
            <div class="field">
                <label>Región</label>
                <input type="number" name="Region" value="1" min="1" max="9">
            </div>
            <div class="field">
                <label>Tipo de Tráfico</label>
                <input type="number" name="TrafficType" value="2" min="1" max="20">
            </div>
            <div class="field">
                <label>Tipo de Visitante</label>
                <select name="VisitorType">
                    <option value="Returning_Visitor" selected>Recurrente</option>
                    <option value="New_Visitor">Nuevo</option>
                    <option value="Other">Otro</option>
                </select>
            </div>
            <div class="field">
                <label>Fin de Semana</label>
                <select name="Weekend">
                    <option value="false">No</option>
                    <option value="true">Sí</option>
                </select>
            </div>

            <button type="submit" id="btn-submit">Analizar Intención de Compra</button>
        </form>

        <div id="result-card">
            <div class="res-header">
                <div id="res-status" class="res-status"></div>
                <div id="res-prob" class="prob-badge"></div>
            </div>
            <p id="res-msg" class="msg-text"></p>
            
            <div class="raw-container">
                <span class="raw-label">Respuesta JSON de la API:</span>
                <pre id="res-raw"></pre>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('predict-form');
        const btn = document.getElementById('btn-submit');
        const resCard = document.getElementById('result-card');

        form.onsubmit = async (e) => {
            e.preventDefault();
            btn.innerText = 'Procesando...';
            btn.disabled = true;
            
            const formData = new FormData(form);
            const data = {};
            
            formData.forEach((value, key) => {
                if (['Administrative', 'Informational', 'ProductRelated', 'OperatingSystems', 'Browser', 'Region', 'TrafficType'].includes(key)) {
                    data[key] = parseInt(value);
                } else if (['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'].includes(key)) {
                    data[key] = parseFloat(value);
                } else if (key === 'Weekend') {
                    data[key] = value === 'true';
                } else {
                    data[key] = value;
                }
            });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();

                const isPurchase = result.classification === 'purchase';
                
                document.getElementById('res-status').innerText = isPurchase ? '✅ Intención Detectada' : '❌ Sin Intención';
                document.getElementById('res-status').style.color = isPurchase ? 'var(--success)' : 'var(--danger)';
                document.getElementById('res-prob').innerText = (result.probability * 100).toFixed(0) + '%';
                document.getElementById('res-msg').innerText = result.message;
                document.getElementById('res-raw').innerText = JSON.stringify(result, null, 2);
                
                resCard.style.borderColor = isPurchase ? 'var(--success)' : 'var(--danger)';
                resCard.style.background = isPurchase ? 'rgba(74, 222, 128, 0.05)' : 'rgba(248, 113, 113, 0.05)';
                resCard.style.display = 'block';
                resCard.scrollIntoView({ behavior: 'smooth' });

            } catch (err) {
                alert('Error al conectar con la API de predicción');
            } finally {
                btn.innerText = 'Analizar Intención de Compra';
                btn.disabled = false;
            }
        };
    </script>
</body>
</html>
"""

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
# Routes
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": type(model).__name__}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json()
    try:
        X_scaled = _encode_and_scale(data)
        prob_purchase = float(model.predict_proba(X_scaled)[0][1])
        classification = "purchase" if prob_purchase >= THRESHOLD else "no_purchase"
        return jsonify({
            "classification": classification,
            "probability": round(prob_purchase, 4),
            "message": _build_message(prob_purchase, classification)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 422

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)