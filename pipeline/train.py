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

print("\n[7] Hyperparameter tuning (GradientBoostingClassifier) …")
print("    This may take several minutes …")

param_grid = {
    "n_estimators":    [200, 300],
    "learning_rate":   [0.05, 0.1],
    "max_depth":       [4, 5],
    "min_samples_leaf":[10, 20],
    "subsample":       [0.8],
}

base_gbc = GradientBoostingClassifier(random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=base_gbc,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
    refit=True,
)
grid_search.fit(X_train_res, y_train_res)

print(f"\n    Best params:   {grid_search.best_params_}")
print(f"    Best CV score: {grid_search.best_score_:.4f}")
best_model = grid_search.best_estimator_

print("\n[8] Optimising decision threshold on real-distribution val set …")
val_probs = best_model.predict_proba(X_val_scaled)[:, 1]

best_threshold = 0.5
best_val_acc   = 0.0

for thresh in np.arange(0.20, 0.71, 0.01):
    preds = (val_probs >= thresh).astype(int)
    acc   = accuracy_score(y_val, preds)
    if acc > best_val_acc:
        best_val_acc   = acc
        best_threshold = round(thresh, 2)

print(f"    Best threshold: {best_threshold} → Val Accuracy: {best_val_acc * 100:.2f}%")

print("\n[9] Final evaluation on held-out test set …")
test_probs = best_model.predict_proba(X_test_scaled)[:, 1]
y_pred     = (test_probs >= best_threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"\n    Test Accuracy (threshold={best_threshold}): {acc * 100:.2f}%")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Purchase", "Purchase"]))
print("    Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

if acc >= 0.90:
    print("\n    Target accuracy ≥90% REACHED!")
elif acc >= 0.88:
    print("\n    Accuracy within acceptable range.")
else:
    print("\n    Below target.")


print("\n[10] Saving model artifacts …")

joblib.dump(best_model, os.path.join(MODEL_DIR, "mlp_model.pkl"))
joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))

encoders = {
    "month_encoder":   month_encoder,
    "visitor_encoder": visitor_encoder,
    "feature_cols":    FEATURE_COLS,
    "month_order":     MONTH_ORDER,
    "threshold":       best_threshold,
}
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

print(f"    Artifacts saved to: {MODEL_DIR}")
print(f"    Threshold: {best_threshold}")
print("\n" + "=" * 60)
print("Training pipeline complete.")
print("=" * 60)