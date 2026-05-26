# from tensorflow.keras.models import Sequential

# from tensorflow.keras.layers import (
#     Conv2D,
#     MaxPooling2D,
#     Flatten,
#     Dense,
#     Dropout
# )


# def build_cnn_model(input_shape):

#     model = Sequential()

#     # First CNN Layer

#     model.add(
#         Conv2D(
#             32,
#             (3, 3),
#             activation='relu',
#             input_shape=input_shape
#         )
#     )

#     model.add(
#         MaxPooling2D((2, 2))
#     )

#     # Second CNN Layer

#     model.add(
#         Conv2D(
#             64,
#             (3, 3),
#             activation='relu'
#         )
#     )

#     model.add(
#         MaxPooling2D((2, 2))
#     )

#     # Flatten

#     model.add(
#         Flatten()
#     )

#     # Dense Layer

#     model.add(
#         Dense(
#             128,
#             activation='relu'
#         )
#     )

#     # Dropout

#     model.add(
#         Dropout(0.3)
#     )

#     # Output Layer

#     model.add(
#         Dense(
#             1,
#             activation='sigmoid'
#         )
#     )

#     # Compile Model

#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )

#     return model

# from tensorflow.keras.models import Sequential

# from tensorflow.keras.layers import (

#     Conv2D,
#     MaxPooling2D,

#     Flatten,

#     Dense,

#     Dropout,

#     BatchNormalization

# )


# def build_cnn_model(input_shape):

#     model = Sequential()

#     # First Block

#     model.add(

#         Conv2D(

#             32,

#             (3, 3),

#             activation='relu',

#             padding='same',

#             input_shape=input_shape

#         )

#     )

#     model.add(
#         BatchNormalization()
#     )

#     model.add(
#         MaxPooling2D((2, 2))
#     )

#     model.add(
#         Dropout(0.25)
#     )

#     # Second Block

#     model.add(

#         Conv2D(

#             64,

#             (3, 3),

#             activation='relu',

#             padding='same'

#         )

#     )

#     model.add(
#         BatchNormalization()
#     )

#     model.add(
#         MaxPooling2D((2, 2))
#     )

#     model.add(
#         Dropout(0.25)
#     )

#     # Flatten

#     model.add(
#         Flatten()
#     )

#     # Dense Layer

#     model.add(

#         Dense(

#             128,

#             activation='relu'

#         )

#     )

#     model.add(
#         Dropout(0.5)
#     )

#     # Output

#     model.add(

#         Dense(

#             1,

#             activation='sigmoid'

#         )

#     )

#     model.compile(

#         optimizer='adam',

#         loss='binary_crossentropy',

#         metrics=['accuracy']

#     )

#     return model


from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (

    Conv2D,
    MaxPooling2D,

    Flatten,

    Dense,

    Dropout

)


def build_cnn_model(input_shape):

    model = Sequential()

    # First CNN Layer

    model.add(

        Conv2D(

            32,

            (3, 3),

            activation='relu',

            input_shape=input_shape

        )

    )

    model.add(
        MaxPooling2D((2, 2))
    )

    # Second CNN Layer

    model.add(

        Conv2D(

            64,

            (3, 3),

            activation='relu'

        )

    )

    model.add(
        MaxPooling2D((2, 2))
    )

    # Flatten

    model.add(
        Flatten()
    )

    # Dense Layer

    model.add(

        Dense(

            128,

            activation='relu'

        )

    )

    # Dropout

    model.add(
        Dropout(0.3)
    )

    # Output Layer

    model.add(

        Dense(

            1,

            activation='sigmoid'

        )

    )

    # Compile

    model.compile(

        optimizer='adam',

        loss='binary_crossentropy',

        metrics=['accuracy']

    )

    return model