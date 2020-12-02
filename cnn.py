from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *


class InstanceNormalization_(keras.layers.Layer):
    """Batch Instance Normalization Layer (https://arxiv.org/abs/1805.07925)."""

    def __init__(self, trainable=None, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.trainable = trainable
        self.gamma, self.beta = None, None

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer=keras.initializers.RandomNormal(1, 0.02),
            trainable=self.trainable)

        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer=keras.initializers.RandomNormal(0, 0.02),
            trainable=self.trainable)

    def call(self, x, trainable=None):
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) * (tf.math.rsqrt(ins_sigma + self.epsilon))
        return x_ins * self.gamma + self.beta

try:
    from tensorflow_addons.layers import InstanceNormalization
except ImportError:
    InstanceNormalization = InstanceNormalization_


W_INIT = keras.initializers.RandomNormal(0, 0.02)


def dc_d(input_shape, name):
    def add_block(filters, ins_norm=True):
        model.add(Conv2D(filters, 3, 2, padding='same', kernel_initializer=W_INIT))
        if ins_norm:
            model.add(InstanceNormalization())
        model.add(LeakyReLU(alpha=0.2))

    model = keras.Sequential([Input(input_shape)], name=name)
    # [n, 128, 128, 3]
    # model.add(GaussianNoise(0.02))
    add_block(64, ins_norm=False)
    # 64
    add_block(128)
    # 32
    add_block(256)
    # 16
    add_block(512)
    # 8
    # add_block(512)
    # 4
    model.add(Conv2D(1, 3, 1, "valid"))
    return model


class ResBlock(keras.layers.Layer):
    def __init__(self, filters, bottlenecks=2):
        super().__init__()
        assert bottlenecks > 0
        self.bs = keras.Sequential(
            [ResBottleneck(filters) for _ in range(bottlenecks)]
        )

    def call(self, x, training=None):
        o = self.bs(x, training=training)
        return o


class ResBottleneck(keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        c = filters // 3
        self.b = keras.Sequential([
            Conv2D(c, 3, 1, padding="same", kernel_initializer=W_INIT),
            InstanceNormalization(),
            LeakyReLU(0.2),
            Conv2D(filters, 3, 1, padding="same", kernel_initializer=W_INIT),
            InstanceNormalization(),
        ])

    def call(self, x, training=None):
        o = self.b(x, training=training) + x
        o = tf.nn.relu(o)
        return o


def unet(input_shape=(128, 128, 3), name="unet"):
    def en_block(inputs, filters, kernels=3, trainable=True):
        _e = Conv2D(filters, kernels, 2, "same", trainable=trainable)(inputs)
        _e = InstanceNormalization(trainable=trainable)(_e)
        _e = LeakyReLU(0.2)(_e)
        return _e

    def de_block(inputs, encoding, filters=64):
        _u = tf.concat((inputs, encoding), axis=3)
        _u = UpSampling2D((2, 2), interpolation="bilinear")(_u)
        _u = Conv2D(filters, 3, 1, "same")(_u)
        _u = InstanceNormalization()(_u)
        _u = LeakyReLU(0.2)(_u)
        return _u

    i = keras.Input(shape=input_shape, dtype=tf.float32)
    train_encoder = True
    # attempt: encoder to extract image random encoding with some its distribution. trainable = False
    i_ = Conv2D(64, 7, 1, "same")(i)
    e1 = en_block(i_, 128, 7, trainable=train_encoder)   # 64
    e2 = en_block(e1, 128, trainable=train_encoder)    # 32

    m = ResBlock(filters=128, bottlenecks=2)(e2)
    for _ in range(3):
        m = ResBlock(128, 2)(m)

    d2 = de_block(m, e2, 128)
    d1 = de_block(d2, e1, 128)

    o = Conv2D(64, 7, 1, "same")(d1)
    o = InstanceNormalization()(o)
    o = LeakyReLU(0.2)(o)
    # o = Conv2D(64, 5, 1, "same")(o)
    # o = InstanceNormalization()(o)
    # o = LeakyReLU(0.2)(o)
    o = Conv2D(input_shape[-1], 7, 1, "same", activation=keras.activations.tanh)(o)
    unet = keras.Model(i, o, name=name)
    return unet

