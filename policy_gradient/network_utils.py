from tensorflow import keras
from keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    GlobalAveragePooling2D,
    BatchNormalization,
    Layer,
    Add,
)
from keras.models import Sequential
from keras.models import Model
from tensorflow.keras import layers
import luig_io
import tensorflow as tf


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
