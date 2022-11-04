from tensorflow import keras


def compute_layers(img_height, img_width, latent_dim, channels=3):
    pass


class Encoder(keras.Sequential):
    def __init__(self, img_height, img_width, latent_dim, channels=3, **kwargs):
        super().__init__(
            layers=compute_layers(img_height, img_width, latent_dim, channels=3),
            **kwargs
        )
