import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def SimpleCNN(**kwargs):
    return tf.keras.Sequential(
        [
            layers.Conv2D(
                32,
                3,
                strides=2,
                activation="relu",
                initializer=tf.keras.initializers.Orthogonal(),
            ),
            layers.Conv2D(
                32,
                3,
                strides=2,
                activation="relu",
                initializer=tf.keras.initializers.Orthogonal(),
            ),
            layers.Conv2D(
                32,
                3,
                strides=2,
                activation="relu",
                initializer=tf.keras.initializers.Orthogonal(),
            ),
            layers.Conv2D(
                32,
                3,
                strides=2,
                activation="relu",
                initializer=tf.keras.initializers.Orthogonal(),
            ),
        ],
        **kwargs
    )
