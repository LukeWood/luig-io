import tensorflow as tf
from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Layer
from keras.layers import MaxPool2D


class ResnetBlock(Model):
    def __init__(self, channels: int, use_batchnorm=True, down_sample=False):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(
            self.__channels,
            strides=self.__strides[0],
            kernel_size=KERNEL_SIZE,
            padding="same",
            kernel_initializer=INIT_SCHEME,
        )
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(
            self.__channels,
            strides=self.__strides[1],
            kernel_size=KERNEL_SIZE,
            padding="same",
            kernel_initializer=INIT_SCHEME,
        )
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels,
                strides=2,
                kernel_size=(1, 1),
                kernel_initializer=INIT_SCHEME,
                padding="same",
            )
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        if self.use_batchnorm:
            x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        if self.use_batchnorm:
            x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            if self.use_batchnorm:
                res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


def SimpleCNN(**kwargs):
    return tf.keras.Sequential(
        [
            Conv2D(256, 3, activation="relu"),
            MaxPool2D(),
            Conv2D(128, 3, activation="relu"),
            MaxPool2D(),
            Conv2D(64, 3, activation="relu"),
        ]
    )


class ResNet18(Model):
    def __init__(self, use_batchnorm=False, **kwargs):
        super().__init__(**kwargs)
        self.use_batchnorm = use_batchnorm
        self.conv_1 = Conv2D(
            64, (7, 7), strides=2, padding="same", kernel_initializer="he_normal"
        )
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64, use_batchnorm=use_batchnorm)
        self.res_1_2 = ResnetBlock(64, use_batchnorm=use_batchnorm)
        self.res_2_1 = ResnetBlock(128, use_batchnorm=use_batchnorm, down_sample=True)
        self.res_2_2 = ResnetBlock(128, use_batchnorm=use_batchnorm)
        self.res_3_1 = ResnetBlock(256, use_batchnorm=use_batchnorm, down_sample=True)
        self.res_3_2 = ResnetBlock(256, use_batchnorm=use_batchnorm)
        self.res_4_1 = ResnetBlock(512, use_batchnorm=use_batchnorm, down_sample=True)
        self.res_4_2 = ResnetBlock(512, use_batchnorm=use_batchnorm)
        self.avg_pool = GlobalAveragePooling2D()

    def call(self, inputs):
        out = self.conv_1(inputs)
        if self.use_batchnorm:
            out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [
            self.res_1_1,
            self.res_1_2,
            self.res_2_1,
            self.res_2_2,
            self.res_3_1,
            self.res_3_2,
            self.res_4_1,
            self.res_4_2,
        ]:
            out = res_block(out)
        out = self.avg_pool(out)
        return out
