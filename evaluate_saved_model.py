"""
Evaluates the ALREADY TRAINED model (heart_model.keras) without retraining.
It uses the same data split as main.py (random_state=42, stratified),
so the test patients are the ones the model never saw during training.

Put this file in the project folder (next to main.py) and run:
    python evaluate_saved_model.py
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

# 1. Load data (same steps as main.py)
df = pd.read_csv("heart_disease.csv")
if "num" in df.columns:
    df = df.drop("num", axis=1)

X = df.drop("target_binary", axis=1)
y = df["target_binary"]

# 2. Same split as main.py: 20% test, then 20% of the rest for validation
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=42, stratify=y_temp)

print("Training rows  :", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows   :", len(X_test))

# 3. Load the saved scaler and model
scaler = joblib.load("scaler.pkl")
model = tf.keras.models.load_model("heart_model.keras")

# 4. Scale the data with the SAVED scaler (transform only, never fit)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# 5. Predict on the unseen test set
proba = model.predict(X_test_s, verbose=0).flatten()
y_pred = (proba >= 0.5).astype(int)

# 6. Scores
print("\n========== MODEL TEST RESULTS (unseen test data) ==========")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4),
      f"({accuracy_score(y_test, y_pred) * 100:.2f}%)")
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1 score :", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("AUC      :", round(roc_auc_score(y_test, proba), 4))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives : {tn}  (healthy, predicted healthy)")
print(f"False Positives: {fp}  (healthy, predicted disease)")
print(f"False Negatives: {fn}  (disease, predicted healthy)")
print(f"True Positives : {tp}  (disease, predicted disease)")

print("\nClassification report:")
print(classification_report(y_test, y_pred,
                            target_names=["No Heart Disease", "Heart Disease"],
                            zero_division=0))

# 7. Overfitting check: training accuracy vs testing accuracy
train_pred = (model.predict(X_train_s, verbose=0).flatten() >= 0.5).astype(int)
print("Training accuracy:", round(accuracy_score(y_train, train_pred), 4))
print("Testing accuracy :", round(accuracy_score(y_test, y_pred), 4))

# 8. Save and show the confusion matrix picture
ConfusionMatrixDisplay(cm, display_labels=["No Heart Disease", "Heart Disease"]).plot(cmap="Blues")
plt.title("Heart Disease Prediction - Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("\nSaved confusion_matrix.png in the project folder")
plt.show()
