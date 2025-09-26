import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, roc_auc_score,
    roc_curve
)

# Paths
DATA_PATH = os.path.join("..", "data", "data.csv")
OUTPUT_DIR = os.path.join("..", "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

# Drop unnecessary columns
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# Convert target: M=1, B=0
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Features & target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# 2. Train-test split & standardization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

# Predictions & probabilities
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 4. Evaluation
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Plot Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign (0)", "Malignant (1)"], yticklabels=["Benign (0)", "Malignant (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.show()

# 5. Threshold tuning example
custom_threshold = 0.6
y_pred_custom = (y_prob >= custom_threshold).astype(int)
cm_custom = confusion_matrix(y_test, y_pred_custom)
print(f"\nConfusion Matrix with threshold={custom_threshold}:\n", cm_custom)

# Explain sigmoid
print("\n🔎 Sigmoid Function: σ(z) = 1 / (1 + e^(-z))")
print("It converts any real number into (0,1), interpreted as probability of being Malignant.")
