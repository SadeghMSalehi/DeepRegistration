import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Input, Dense
from keras.layers.normalization import BatchNormalization


def Pitanh(x):
    q = 3.1415 * tf.nn.tanh(x)
    return q


def model_3d(width, height, depth, channel):
    model = Sequential()

    model.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same'
                     , input_shape=(width, height, depth, channel)))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(3, activation= Pitanh))
    return model