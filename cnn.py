from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *


class InstanceNormalization(keras.layers.Layer):
    def __init__(self, axis=(1, 2), epsilon=1e-6):
        super().__init__()
        # NHWC
        self.axis = axis
        self.epsilon = epsilon
        self.gamma, self.beta = None, None

    def build(self, input_shape):
        # NHWC
        shape = [1, 1, 1, input_shape[-1]]
        self.gamma = self.add_weight(
            name='gamma',
            shape=shape,
            initializer=keras.initializers.RandomNormal(1, 0.02))

        self.beta = self.add_weight(
            name='beta',
            shape=shape,
            initializer=keras.initializers.RandomNormal(0, 0.02))

    def call(self, x, trainable=None):
        mean = tf.math.reduce_mean(x, axis=self.axis, keepdims=True)
        x -= mean
        variance = tf.math.reduce_mean(tf.math.square(x), axis=self.axis, keepdims=True)
        x *= tf.math.rsqrt(variance + self.epsilon)
        return x * self.gamma + self.beta


W_INIT = keras.initializers.HeNormal()


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
    add_block(128)
    # 16
    add_block(256)
    # 8
    add_block(512)
    # 4
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
        self.filters = filters
        self.b = None
        self.projection = None

    def call(self, x, **kwargs):
        o = self.b(x)
        if self.projection is not None:
            o += self.projection(x)
        return o

    def build(self, input_shape):
        c = input_shape[-1]
        if c != self.filters:
            self.projection = Conv2D(self.filters, 1, 1)
        self.b = keras.Sequential([
            Conv2D(c, 3, 1, padding="same", kernel_initializer=W_INIT),
            LeakyReLU(0.2),
            InstanceNormalization(),
            Conv2D(self.filters, 1, 1, kernel_initializer=W_INIT),
        ])


def unet(input_shape=(128, 128, 3), name="unet"):
    def en_block(inputs, filters, kernels=3):
        _e = Conv2D(filters, kernels, 2, "same")(inputs)
        _e = LeakyReLU(0.2)(_e)
        _e = InstanceNormalization()(_e)
        return _e

    def de_block(inputs, encoding, filters=64):
        _u = tf.concat((inputs, encoding), axis=3)
        _u = UpSampling2D((2, 2), interpolation="bilinear")(_u)
        _u = Conv2D(filters, 3, 1, "same", kernel_initializer=W_INIT)(_u)
        _u = LeakyReLU(0.2)(_u)
        _u = InstanceNormalization()(_u)
        return _u

    i = keras.Input(shape=input_shape, dtype=tf.float32)
    i_ = Conv2D(32, 5, 1, "same", kernel_initializer=W_INIT)(i)
    e1 = en_block(i_, 64, 3)   # 64
    e2 = en_block(e1, 128)    # 32

    m = ResBlock(filters=128, bottlenecks=2)(e2)
    for _ in range(2):
        m = ResBlock(128, 2)(m)

    o = de_block(m, e2, 128)
    o = de_block(o, e1, 64)

    o = Conv2D(64, 3, 1, "same", kernel_initializer=W_INIT)(o)
    o = LeakyReLU(0.2)(o)
    o = InstanceNormalization()(o)
    o = Conv2D(input_shape[-1], 5, 1, "same", activation=keras.activations.tanh, kernel_initializer=W_INIT)(o)
    unet = keras.Model(i, o, name=name)
    return unet

