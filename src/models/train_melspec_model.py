import os
import numpy as np

from tensorflow.keras.callbacks import (
    ModelCheckpoint
)

from src.models.cnn_model import (
    build_cnn_model
)

FEATURE_PATH = "data/features/mel_spectrogram"

X_train = []
y_train = []

X_val = []
y_val = []


# Load Train + Validation Data

for split, X, y in [

    ("train", X_train, y_train),

    ("validation", X_val, y_val)

]:

    for label, class_name in enumerate(

        ["real", "fake"]

    ):

        folder = os.path.join(

            FEATURE_PATH,
            split,
            class_name

        )

        for file in os.listdir(folder):

            file_path = os.path.join(
                folder,
                file
            )

            feature = np.load(file_path)

            X.append(feature)

            y.append(label)


# Convert to numpy arrays

X_train = np.array(X_train)
y_train = np.array(y_train)

X_val = np.array(X_val)
y_val = np.array(y_val)


# Add channel dimension

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]


print("\nTrain Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)


# Input Shape

input_shape = X_train.shape[1:]


# Build CNN

model = build_cnn_model(
    input_shape
)


# Save Best Model

checkpoint = ModelCheckpoint(

    "saved_models/best_melspec_model.h5",

    monitor='val_accuracy',

    save_best_only=True,

    mode='max',

    verbose=1
)


# Train

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=10,

    batch_size=32,

    callbacks=[checkpoint]

)