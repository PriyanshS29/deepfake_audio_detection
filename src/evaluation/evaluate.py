import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from tensorflow.keras.models import load_model


FEATURE_PATH = "data/features/mfcc"


X_test = []
y_test = []


# Load Test Data

for label, class_name in enumerate(

    ["real", "fake"]

):

    folder = os.path.join(

        FEATURE_PATH,
        "test",
        class_name

    )

    for file in os.listdir(folder):

        file_path = os.path.join(
            folder,
            file
        )

        feature = np.load(file_path)

        X_test.append(feature)

        y_test.append(label)


# Convert to numpy arrays

X_test = np.array(X_test)

y_test = np.array(y_test)


# Add channel dimension

X_test = X_test[..., np.newaxis]


print("\nTest Shape:", X_test.shape)


# Load Best Model

model = load_model(
    "saved_models/improved_mfcc_cnn.h5"
)


# Predictions

pred_probs = model.predict(X_test)

predictions = (
    pred_probs > 0.5
).astype(int)


# Classification Report

print("\nClassification Report:\n")

print(

    classification_report(
        y_test,
        predictions
    )

)


# Confusion Matrix

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(6, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

os.makedirs(
    "results/plots",
    exist_ok=True
)

plt.savefig(
    "results/plots/confusion_matrix.png"
)

plt.close()


# ROC Curve

fpr, tpr, thresholds = roc_curve(
    y_test,
    pred_probs
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    "results/plots/roc_curve.png"
)

plt.close()


print(f"\nROC-AUC Score: {roc_auc:.4f}")

print("\nEvaluation Completed Successfully")