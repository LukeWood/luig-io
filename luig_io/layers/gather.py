from tensorflow import keras
from tensorflow.keras import layers

class Gather(layers.Layer):

    def __init__(self, gather_dim):
        self.gather_dim = gather_dim
        if len(gather_dim) != 3:
            raise ValueError("Right now `gather_dim` must be of len 3. "
            f"Got `gather_dim={gather_dim}`")
    def build(self, input_shape):
        for is, gd in zip(input_shape[1:], gather_dim):
            if is < gd:
                raise ValueError(
                    "`gather_dim` must be <= all dimensions of `input_shape`. "
                    f"Instead, got `gather_dim={gather_dim}`, "
                    f"input_shape={input_shape}."
                )

    def call(self, inputs):
        x, y, c = self.gather_dim
        return inputs[:, :x, :y, :c]
