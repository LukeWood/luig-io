import tensorflow as tf
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Layer
from keras.layers import MaxPool2D
from keras.models import Model
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers

import luig_io


def build_network(input_shape, output_size, name=None):
    """Builds a CNN using `luig_io.models.SimpleCNN`.

    Args:
        input_shape: the input shape for the model.
        output_size: int, the dimension of the output.
        name: the model name.
    Returns:
        A keras.Model representing the network.
    """
    inputs = layers.Input(input_shape)
    encoder = luig_io.models.SimpleCNN()
    # encoder = luig_io.models.ResNet18()
    x = encoder(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(output_size, activation=None)(x)
    return keras.Model(inputs, x, name=name)
