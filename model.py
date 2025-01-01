from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop


def create_model():
    model = Sequential()

    # لایه 1: Convolutional + Batch Normalization + Max Pooling
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(128, 128, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # لایه 2: Convolutional + Batch Normalization + Max Pooling
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # لایه 3: Convolutional + Batch Normalization + Max Pooling
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # لایه 4: Convolutional + Batch Normalization + Max Pooling
    model.add(Conv2D(256, (2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # لایه 5: Flatten
    model.add(Flatten())

    # لایه 6: Dense (512 neurons) + Dropout
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # لایه 7: Dense (256 neurons) + Dropout
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # لایه 8: Dense (128 neurons) + Dropout
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    # لایه 9: Dense (2 neurons for binary classification) + Softmax
    model.add(Dense(2, activation='softmax'))

    # کامپایل مدل با RMSprop
    model.compile(optimizer=RMSprop(),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load_existing_model(model_path):
    try:
        from tensorflow.keras.models import load_model
        return load_model(model_path)
    except:
        return None
