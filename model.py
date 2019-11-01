from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Flatten, Dense, Activation, Dropout
import utils


def get_model():
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(utils.IMG_HT, utils.IMG_WIDTH, utils.IMG_CH)))

    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu", strides=(1, 1)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Dense(1))

    return model
