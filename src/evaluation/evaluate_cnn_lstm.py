

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

        # transpose for LSTM

        feature = feature.T

        X_test.append(feature)

        y_test.append(label)


X_test = np.array(X_test)
y_test = np.array(y_test)


print("Test Shape:", X_test.shape)


model = load_model(
    "saved_models/cnn_lstm_mfcc.h5"
)


pred_probs = model.predict(X_test)

predictions = (
    pred_probs > 0.5
).astype(int)


print("\nClassification Report:\n")

print(

    classification_report(
        y_test,
        predictions
    )

)


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
    "results/plots/cnn_lstm_confusion_matrix.png"
)

plt.close()


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

plt.title("CNN-LSTM ROC Curve")

plt.legend()

plt.savefig(
    "results/plots/cnn_lstm_roc_curve.png"
)

plt.close()


print(f"\nROC-AUC Score: {roc_auc:.4f}")

print("\nCNN-LSTM Evaluation Completed")