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


print("\n[3] Encoding categorical features …")

MONTH_ORDER = ["Feb", "Mar", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_encoder = OrdinalEncoder(
    categories=[MONTH_ORDER],
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)
df["Month_enc"] = month_encoder.fit_transform(df[["Month"]])

visitor_encoder = LabelEncoder()
df["VisitorType_enc"] = visitor_encoder.fit_transform(df["VisitorType"])

df["Weekend_enc"] = df["Weekend"].astype(int)
df["Revenue_enc"] = df["Revenue"].astype(int)


FEATURE_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month_enc", "OperatingSystems", "Browser",
    "Region", "TrafficType",
    "VisitorType_enc", "Weekend_enc",
]

X = df[FEATURE_COLS].values
y = df["Revenue_enc"].values

print(f"    Feature matrix: {X.shape} | Class dist: {Counter(y)}")

print("\n[4] Splitting data (60/20/20 stratified) …")

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)
print(f"    Train:{X_train.shape} | Val:{X_val.shape} | Test:{X_test.shape}")

print("\n[5] Scaling with StandardScaler …")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print("\n[6] Applying SMOTE to training split …")
print(f"    Before: {Counter(y_train)}")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"    After : {Counter(y_train_res)}")