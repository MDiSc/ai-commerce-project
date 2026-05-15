#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# start.sh — E-commerce Purchase Intention API
# Usage:
#   ./start.sh          → starts the API (assumes model is trained)
#   ./start.sh --train  → runs the full training pipeline first, then starts API
# ─────────────────────────────────────────────────────────────

set -e

VENV="./venv"
PYTHON="$VENV/bin/python3"

# Create venv if it doesn't exist
if [ ! -d "$VENV" ]; then
  echo "[setup] Creating virtual environment …"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -r requirements.txt
  echo "[setup] Dependencies installed."
fi

# Optional: retrain
if [ "$1" == "--train" ]; then
  echo "[pipeline] Running training pipeline …"
  "$PYTHON" pipeline/train.py
  echo "[pipeline] Training complete."
fi

# Check model artifacts exist
if [ ! -f "pipeline/model/mlp_model.pkl" ]; then
  echo "[error] No model found. Run: ./start.sh --train"
  exit 1
fi

echo "[api] Starting prediction server on http://0.0.0.0:5000 …"
"$PYTHON" api/app.py