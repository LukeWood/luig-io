import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def SimpleCNN(**kwargs):
    return tf.keras.Sequential(
        [
            layers.Conv2D(
                256,
                3,
                activation="relu",
                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            ),
            layers.MaxPool2D(),
            layers.Conv2D(
                128,
                3,
                activation="relu",
                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            ),
            layers.MaxPool2D(),
            layers.Conv2D(
                64,
                3,
                activation="relu",
                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            ),
        ],
        **kwargs
    )
