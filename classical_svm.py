import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

import joblib


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv("processed_data.csv")

FEATURES = [
    "PC1",
    "PC2",
    "PC3",
    "PC4",
    "PC5",
    "PC6",
    "PC7",
    "PC8"
]

X = data[FEATURES]
y = data["Label"]


print("Dataset shape:", data.shape)

print("\nClass distribution:")
print("Control:", (y == 0).sum())
print("Lung:", (y == 1).sum())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# BALANCED CLASSICAL SVM
# ============================================================

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced"
)


print("\nTraining Balanced Classical SVM...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================================
# TEST PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

cm = confusion_matrix(y_test, y_pred)


print("\n========================================")
print("BALANCED CLASSICAL SVM RESULTS")
print("========================================")

print(f"Accuracy          : {accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Lung Recall       : {recall:.4f}")
print(f"F1 Score          : {f1:.4f}")
print(f"Balanced Accuracy : {balanced_accuracy:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Control", "Lung"],
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "classical_svm_model.pkl"
)

print("\nModel saved as:")
print("classical_svm_model.pkl")