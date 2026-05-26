

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (

    Conv1D,

    MaxPooling1D,

    LSTM,

    Dense,

    Dropout

)


def build_cnn_lstm_model(input_shape):

    model = Sequential()

    # CNN Layers

    model.add(

        Conv1D(

            64,

            kernel_size=3,

            activation='relu',

            input_shape=input_shape

        )

    )

    model.add(
        MaxPooling1D(pool_size=2)
    )

    model.add(
        Dropout(0.3)
    )

    model.add(

        Conv1D(

            128,

            kernel_size=3,

            activation='relu'

        )

    )

    model.add(
        MaxPooling1D(pool_size=2)
    )

    model.add(
        Dropout(0.3)
    )

    # LSTM Layer

    model.add(

        LSTM(

            64,

            return_sequences=False

        )

    )

    model.add(
        Dropout(0.3)
    )

    # Dense Layer

    model.add(

        Dense(

            64,

            activation='relu'

        )

    )

    model.add(
        Dropout(0.3)
    )

    # Output

    model.add(

        Dense(

            1,

            activation='sigmoid'

        )

    )

    model.compile(

        optimizer='adam',

        loss='binary_crossentropy',

        metrics=['accuracy']

    )

    return model