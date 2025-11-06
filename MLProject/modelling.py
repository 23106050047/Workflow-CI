# modelling.py - CI/CD version (Simplified from modelling_tuning.py)
import os
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_auc_score, log_loss,
    roc_curve, precision_recall_curve, auc
)

# Load data preprocessing
if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = "diabetes_prediction_dataset_preprocessing.csv"

try:
    df = pd.read_csv(data_path)
    if "diabetes" not in df.columns:
        raise ValueError("Kolom 'diabetes' tidak ditemukan di dataset.")
except Exception as e:
    print(f"Error saat membaca dataset: {e}")
    sys.exit(1)

TARGET_COL = "diabetes"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter Tuning
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

# Training + Logging
start_time = time.time()
grid.fit(X_train, y_train)
train_time = time.time() - start_time

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

# ===== Metrics =====
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)
test_size_ratio = len(X_test) / len(X)

# ROC-AUC: cek binary atau multi-class
n_classes = len(np.unique(y))
if n_classes == 2:
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
else:
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro")

logloss = log_loss(y_test, y_pred_proba)

# Log ke MLflow
mlflow.log_params(grid.best_params_)
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("f1_macro", f1)
mlflow.log_metric("precision_macro", precision)
mlflow.log_metric("recall_macro", recall)
mlflow.log_metric("matthews_corrcoef", mcc)
mlflow.log_metric("train_time_sec", train_time)
mlflow.log_metric("test_size_ratio", test_size_ratio)
mlflow.log_metric("roc_auc", roc_auc)
mlflow.log_metric("log_loss", logloss)
mlflow.log_metric("n_classes", n_classes)

mlflow.set_tags({
    "stage": "production",
    "framework": "scikit-learn",
    "task": "classification",
    "data_rows": len(df),
    "data_features": X.shape[1]
})

# ===== Artifacts =====
# Buat folder cm untuk artifacts
os.makedirs("cm", exist_ok=True)

# 1) Confusion Matrix plot
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax_cm, values_format="d", colorbar=False)
plt.tight_layout()
cm_path = "cm/confusion_matrix.png"
plt.savefig(cm_path)
mlflow.log_artifact(cm_path)
plt.close(fig_cm)

# 2) Feature Importances plot
importances = getattr(best_model, "feature_importances_", None)
if importances is not None:
    fi = pd.Series(importances, index=X.columns).sort_values(ascending=True)
    fig_fi, ax_fi = plt.subplots(figsize=(7, max(4, len(fi) * 0.25)))
    fi.plot(kind="barh", ax=ax_fi)
    ax_fi.set_title("RandomForest Feature Importances")
    ax_fi.set_xlabel("Importance")
    plt.tight_layout()
    fi_path = "cm/feature_importances.png"
    plt.savefig(fi_path)
    mlflow.log_artifact(fi_path)
    plt.close(fig_fi)

# 3) ROC Curve Plot
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
classes = np.unique(y)

if len(classes) == 2:  # Binary
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc_val = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_val:.3f})', linewidth=2)
else:  # Multi-class
    for i, cls in enumerate(classes):
        y_test_binary = (y_test == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc_val:.3f})')

ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve')
ax_roc.legend(loc='lower right')
ax_roc.grid(alpha=0.3)
plt.tight_layout()
roc_path = "cm/roc_curve.png"
plt.savefig(roc_path)
mlflow.log_artifact(roc_path)
plt.close(fig_roc)

# 4) Precision-Recall Curve Plot
fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

if len(classes) == 2:  # Binary
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    pr_auc = auc(recall_curve, precision_curve)
    ax_pr.plot(recall_curve, precision_curve, label=f'PR curve (AUC = {pr_auc:.3f})', linewidth=2)
else:  # Multi-class
    for i, cls in enumerate(classes):
        y_test_binary = (y_test == cls).astype(int)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_binary, y_pred_proba[:, i])
        pr_auc = auc(recall_curve, precision_curve)
        ax_pr.plot(recall_curve, precision_curve, label=f'Class {cls} (AUC = {pr_auc:.3f})')

ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision')
ax_pr.set_title('Precision-Recall Curve')
ax_pr.legend(loc='best')
ax_pr.grid(alpha=0.3)
plt.tight_layout()
pr_path = "cm/precision_recall_curve.png"
plt.savefig(pr_path)
mlflow.log_artifact(pr_path)
plt.close(fig_pr)

# 5) Metrics summary JSON
import json
metrics_summary = {
    "best_params": grid.best_params_,
    "metrics": {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "matthews_corrcoef": float(mcc),
        "roc_auc": float(roc_auc),
        "log_loss": float(logloss),
        "train_time_sec": float(train_time),
        "test_size_ratio": float(test_size_ratio)
    },
    "data_info": {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "n_classes": int(n_classes)
    }
}

metrics_path = "cm/metrics_summary.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics_summary, f, indent=2)
mlflow.log_artifact(metrics_path)

# 6) Classification report
cls_report = classification_report(y_test, y_pred, zero_division=0)
report_path = "cm/classification_report.txt"
with open(report_path, 'w') as f:
    f.write(cls_report)
mlflow.log_artifact(report_path)

# ===== Log model dengan signature =====
signature = infer_signature(X_test, best_model.predict(X_test))
input_example = X_test.iloc[:5]
mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="model",
    signature=signature,
    input_example=input_example
)

print("=" * 50)
print("Training Completed Successfully!")
print("=" * 50)
print(f"Best params: {grid.best_params_}")
print(f"Accuracy: {acc:.4f}")
print(f"F1-macro: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("=" * 50)
