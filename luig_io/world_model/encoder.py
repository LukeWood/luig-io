from tensorflow import keras
from keras import layers
import numpy as np
from luig_io.layers import ResnetBlock, GroupNormalization, AttentionBlock

def compute_layers(input_shape, latent_dim, downscale_blocks=[128, 256, 512], final_block=512):
    result = []
    result += [layers.Input(input_shape)]

    # final block
    for block_size in downscale_blocks:
        result+=[
            layers.Conv2D(block_size, 3, padding='same'),
            ResnetBlock(block_size),
            ResnetBlock(block_size),
            layers.Conv2D(block_size, 3, padding'same', strides=2)
        ]

    # final block
    result += [
        AttentionBlock(final_block),
        ResnetBlock(final_block),
        GroupNormalization(epsilon=1e-5),
        layers.Activation('swish'),
    ]

    result += [
        layers.Conv2D(8, 3, padding='same'),
        layers.Conv2D(8, 1, padding='same')
    ]

    #filter_amounts = np.linspace(input_shape)

class Encoder(keras.Sequential):
    def __init__(self, input_shape, latent_dim, **kwargs):
        if len(input_shape) != 3 or len(latent_dim) != 3:
            raise ValueError(
                "Expected both `input_shape` and `latent_dim` to be "
                "tuples of 3 elements representing HWC dimensions. "
                f"Got `input_shape={input_shape}` and `latent_dim={latent_dim}`."
            )

        super().__init__(
            layers=compute_layers(input_shape, latent_dim),
            **kwargs
        )
