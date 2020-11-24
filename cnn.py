from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *


class InstanceNormalization_(keras.layers.Layer):
    """Batch Instance Normalization Layer (https://arxiv.org/abs/1805.07925)."""

    def __init__(self, trainable=None, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.trainable = trainable

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=self.trainable)

        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
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
        model.add(Conv2D(filters, 4, strides=2, padding='same', kernel_initializer=W_INIT))
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
    model.add(Conv2D(1, 4, 1, "same"))
    return model


class ResBlock(keras.layers.Layer):
    def __init__(self, filters, activation=None, bottlenecks=2):
        super().__init__()
        self.activation = activation
        self.bn = keras.Sequential(
            [ResBottleneck(filters)]
        )
        if bottlenecks > 1:
            self.bn.add(ReLU())
            for _ in range(1, bottlenecks):
                self.bn.add(ResBottleneck(filters))

    def call(self, x, training=None):
        o = self.bn(x, training=training)
        if self.activation is not None:
            o = self.activation(o)
        return o


class ResBottleneck(keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        c = filters // 2
        self.b = keras.Sequential([Conv2D(c, 1, strides=1, padding="same", kernel_initializer=W_INIT)])
        self.b.add(InstanceNormalization())
        self.b.add(ReLU())
        self.b.add(Conv2D(filters, 4, strides=1, padding="same", kernel_initializer=W_INIT))
        self.ins_norm = InstanceNormalization()

    def call(self, x, training=None):
        b = self.b(x, training=training)
        x = self.ins_norm(b + x)
        return x


def unet(input_shape=(128, 128, 3), name="unet"):
    def en_block(inputs, filters=64, trainable=True):
        _o = Conv2D(filters, 3, 2, "same", trainable=trainable)(inputs)
        _o = InstanceNormalization(trainable=trainable)(_o)
        _o = ReLU()(_o)
        return _o

    def de_block(inputs, encoding, filters=64):
        _u = tf.concat((inputs, encoding), axis=3)
        _u = Conv2DTranspose(filters, 3, 2, "same")(_u)
        _u = InstanceNormalization()(_u)
        _u = ReLU()(_u)
        return _u

    def c7s164(inputs, trainable=True):
        # e = GaussianNoise(0.02)(inputs)
        e = Conv2D(64, 7, 2, "same", trainable=trainable)(inputs)
        e = InstanceNormalization(trainable=trainable)(e)
        e = ReLU()(e)
        return e

    i = keras.Input(shape=input_shape, dtype=tf.float32)
    # attempt: encoder to extract image random encoding. trainable = False
    encoder = [c7s164(i, trainable=False)]   # 64
    encoder.append(en_block(encoder[-1], 128, trainable=False))    # 32
    encoder.append(en_block(encoder[-1], 256, trainable=False))  # 16

    m = ResBlock(filters=256, activation=tf.nn.relu, bottlenecks=1)(encoder[-1])  # 16
    for _ in range(3):
        m = ResBlock(256, tf.nn.relu, 1)(m)  # 16

    decoder = [de_block(m, encoder.pop(), 256)]
    decoder.append(de_block(decoder[-1], encoder.pop(), 128))
    decoder.append(de_block(decoder[-1], encoder.pop(), 64))

    o = Conv2D(32, 4, 1, "same")(decoder[-1])
    o = InstanceNormalization()(o)
    o = ReLU()(o)
    o = Conv2D(input_shape[-1], 7, 1, "same", activation=keras.activations.tanh)(o)
    unet = keras.Model(i, o, name=name)
    return unet

