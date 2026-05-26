

import os
import numpy as np

from tensorflow.keras.callbacks import (

    EarlyStopping,

    ReduceLROnPlateau,

    ModelCheckpoint

)

from src.models.cnn_lstm_model import (
    build_cnn_lstm_model
)

FEATURE_PATH = "data/features/mfcc"


X_train = []
y_train = []

X_val = []
y_val = []


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

            # transpose for LSTM

            feature = feature.T

            X.append(feature)

            y.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)

X_val = np.array(X_val)
y_val = np.array(y_val)


print("Train Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)


input_shape = X_train.shape[1:]


model = build_cnn_lstm_model(
    input_shape
)


checkpoint = ModelCheckpoint(

    "saved_models/cnn_lstm_mfcc.h5",

    monitor='val_accuracy',

    save_best_only=True,

    mode='max',

    verbose=1

)


early_stopping = EarlyStopping(

    monitor='val_loss',

    patience=3,

    restore_best_weights=True,

    verbose=1

)


reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=2,

    verbose=1

)


history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=20,

    batch_size=32,

    callbacks=[

        checkpoint,

        early_stopping,

        reduce_lr

    ]

)