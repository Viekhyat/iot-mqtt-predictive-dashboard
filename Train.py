import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import numpy as np
# ===============================
# 1. Load & Clean Data
# ===============================
df = pd.read_excel("iiot_dataset.xlsx")
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r"[()/%-]", "", regex=True)
print("✅ Columns after renaming:", df.columns.tolist())

df = df.dropna()  # Remove rows with missing values

X = df.drop("Maintenance_Required", axis=1)
y = df["Maintenance_Required"]

# Give more weight to energy & carbon
X["Energy_Consumption_kWh"] *= 1.2
X["Carbon_Emission_kg"] *= 1.2

numeric_features = ["Temperature_C", "Vibration_Level_mms",
                   "Energy_Consumption_kWh", "Carbon_Emission_kg", "Downtime_Hours"]

categorical_features = ["Industry", "Region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)
# ===============================
# 2. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# ===============================
# 3. Handle Imbalance with scale_pos_weight
# ===============================
neg, pos = np.bincount(y_train)
scale = neg / pos
print(f"⚖️ Class balance: {neg} no-maintenance, {pos} maintenance → scale_pos_weight={scale:.2f}")
# ===============================
# 4. Build Model with Calibration
# ===============================
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
calibrated_model = CalibratedClassifierCV(xgb_model, cv=3, method="sigmoid")

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", calibrated_model)
])
# ===============================
# 5. Train Model
# ===============================
print("⏳ Training model...")
clf.fit(X_train, y_train)
print("✅ Training complete.")
# ===============================
# 6. Save Model + Test Data
# ===============================
joblib.dump(clf, "maintenance_model.pkl")
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# ===============================
# 7. Predictions with Custom Threshold
# ===============================
# Predict probabilities
y_proba = clf.predict_proba(X_test)[:, 1]
# Debug check
print("🔎 Sample probabilities:", y_proba[:20])
print("⚖️ Mean probability:", y_proba.mean())
# Set your threshold
threshold = 0.55
y_pred_custom = (y_proba >= threshold).astype(int)
# Save predictions
np.savetxt("y_pred_custom.csv", y_pred_custom, delimiter=",", fmt="%d")
# ===============================
# 8. Report
# ===============================
from sklearn.metrics import classification_report, confusion_matrix
print(f"\n✅ Custom threshold applied: {threshold}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred_custom))
print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))
