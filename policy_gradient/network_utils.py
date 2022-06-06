import tensorflow as tf
from keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Layer,
    MaxPool2D,
)
from keras.models import Model, Sequential
from tensorflow import keras
from tensorflow.keras import layers

import luig_io


def build_network(input_shape, output_size, name=None):
    """
    Args:
        output_size: int, the dimension of the output
        n_layers: int, the number of hidden layers of the network
        size: int, the size of each hidden layer
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
