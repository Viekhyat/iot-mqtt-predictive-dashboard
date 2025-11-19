import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)

# 1. Load model & test data
clf = joblib.load("maintenance_model.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# 2. Predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# 3. Evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# 4. Ensure result folder exists
os.makedirs("result", exist_ok=True)

# 5. Save metrics to text file (UTF-8 to support emojis)
with open("result/test_results.txt", "w", encoding="utf-8") as f:
    f.write("📊 Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n\n")
    f.write("📋 Classification Report:\n")
    f.write(class_report + "\n")
    f.write(f"✅ ROC-AUC: {roc_auc:.4f}\n")

# 6. Save predictions to CSV
pred_df = pd.DataFrame({
    "y_true": y_test.values.ravel(),
    "y_pred": y_pred,
    "y_prob": y_prob
})
pred_df.to_csv("result/predictions.csv", index=False)

# 7. Plot confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No Maintenance", "Maintenance Required"],
            yticklabels=["No Maintenance", "Maintenance Required"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.savefig("result/confusion_matrix_heatmap.png")
plt.close()

# 8. Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("result/roc_curve.png")
plt.close()

# 9. Plot Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(rec, prec, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("result/precision_recall_curve.png")
plt.close()

print("✅ Test results, predictions, and plots saved in 'result/' folder.")