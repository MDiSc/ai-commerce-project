import os, joblib, numpy as np, pandas as pd
from collections import Counter

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET   = os.path.join(BASE_DIR, "dataset shop.csv")
MODEL_DIR = os.path.join(BASE_DIR, "pipeline", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("[1] Loading dataset …")
df = pd.read_csv(DATASET)
print(f"    Shape: {df.shape}")
print(f"    Target distribution:\n{df['Revenue'].value_counts()}")

print("\n[2] Cleaning data …")
bool_map = {"TRUE": True, "FALSE": False, True: True, False: False}
df["Weekend"] = df["Weekend"].map(bool_map)
df["Revenue"]  = df["Revenue"].map(bool_map)

null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ["object", "bool"]:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    print("    Imputation complete.")
else:
    print("    No missing values detected — no imputation needed.")
