from tensorflow import keras
from tensorflow.keras import layers

def build_mlp(output_size, n_layers, size, name=None):
    """
    Args:
        output_size: int, the dimension of the output
        n_layers: int, the number of hidden layers of the network
        size: int, the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.
    """
    layers = [keras.layers.Dense(size, activation='relu') for _ in range(n_layers)]
    layers.append(keras.layers.Dense(output_size, activation=None))
    return keras.Sequential(layers, name=name)
