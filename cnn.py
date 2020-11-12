from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *


W_INIT = keras.initializers.RandomNormal(0, 0.02)


def dc_d(use_bn=True):
    def add_block(filters):
        model.add(Conv2D(filters, 4, strides=2, padding='same', kernel_initializer=W_INIT))
        if use_bn:
            model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.4))

    model = keras.Sequential()
    # [n, 128, 128, 3]
    if use_bn:
        model.add(BatchNormalization())
    model.add(GaussianNoise(0.02))
    add_block(32)
    # 64
    add_block(64)
    # 32
    add_block(64)
    # 16
    add_block(128)
    # 8
    add_block(128)
    # 4
    model.add(Flatten())
    return model


def dc_g(input_shape):
    block = lambda filters: [
        UpSampling2D((2, 2)),
        Conv2D(filters, 4, 1, "same", kernel_initializer=W_INIT),
        BatchNormalization(), ReLU()
    ]
    return keras.Sequential([
        # [n, latent]
        Dense(4 * 4 * 128, input_shape=input_shape, kernel_initializer=W_INIT),
        BatchNormalization(), ReLU(),
        Reshape((4, 4, 128)),
        # 4
        *block(64),
        # 8
        *block(64),
        # 16
        *block(64),
        # 32
        *block(32),
        # 64
        *block(16),
        # 128
        Conv2D(3, 5, 1, padding="same",  activation=keras.activations.tanh, kernel_initializer=W_INIT)
    ])


class ResBlock(keras.layers.Layer):
    def __init__(self, filters, activation=None, bottlenecks=2, use_bn=True, inner_relu=True):
        super().__init__()
        self.activation = activation
        self.bn = keras.Sequential(
            [ResBottleneck(filters, use_bn, inner_relu=inner_relu)]
        )
        if bottlenecks > 1:
            self.bn.add(ReLU())
            for _ in range(1, bottlenecks):
                self.bn.add(ResBottleneck(filters, use_bn, inner_relu=inner_relu))

    def call(self, x, training=None):
        o = self.bn(x, training=training)
        if self.activation is not None:
            o = self.activation(o)
        return o


class ResBottleneck(keras.layers.Layer):
    def __init__(self, filters, use_bn=True, inner_relu=True):
        super().__init__()
        self.use_bn = use_bn
        c = filters // 4
        self.b = keras.Sequential([Conv2D(c, 1, strides=1, padding="same", kernel_initializer=W_INIT)])
        if use_bn:
            self.b.add(BatchNormalization())
        self.b.add(ReLU() if inner_relu else LeakyReLU())
        self.b.add(Conv2D(filters, 4, strides=1, padding="same", kernel_initializer=W_INIT))
        if use_bn:
            self.b.add(BatchNormalization())

        self.out_bn = None
        if use_bn:
            self.out_bn = BatchNormalization()

    def call(self, x, training=None):
        b = self.b(x, training=training)
        x = b + x
        if self.out_bn is not None:
            x = self.out_bn(x)
        return x


def resnet_g(input_shape, img_shape=(128, 128, 3), use_bn=True):
    h, w = img_shape[0], img_shape[1]
    _h, _w = 4, 4
    m = keras.Sequential([
        keras.Input(input_shape),   # [40 + 100]
        Dense(_h * _w * 256, input_shape=input_shape, kernel_initializer=W_INIT),
        BatchNormalization(),
        Reshape((_h, _w, 256)),
    ], name="resnet_g")
    c = _c = 256
    while True:
        strides = [1, 1]
        if _h < h:
            _h *= 2
            strides[0] = 2
        if _w < w:
            _w *= 2
            strides[1] = 2
        m.add(UpSampling2D(strides))
        if c != _c:
            c = _c
            m.add(Conv2D(_c, 1, 1, "same"))
            if use_bn:
                m.add(BatchNormalization())
        m.add(ResBlock(filters=c, bottlenecks=1, use_bn=use_bn))
        if _w >= w and _h >= h:
            break
        _c = max(int(c / 2), 128)

    m.add(Conv2D(3, 5, 1, "same"))
    if use_bn:
        m.add(BatchNormalization())
    m.add(Activation(keras.activations.tanh))
    return m


def resnet_d(input_shape=(128,128,3), use_bn=True, name="resnet_d"):
    _h, _w = input_shape[0], input_shape[1]
    h, w = 8, 8
    m = keras.Sequential(name=name)
    if use_bn:
        m.add(BatchNormalization())
    c = 64
    while True:
        strides = [1, 1]
        if _h > h:
            _h //= 2
            strides[0] = 2
        if _w > w:
            _w //= 2
            strides[1] = 2
        m.add(Conv2D(c, 4, strides, "same", kernel_initializer=W_INIT))
        if use_bn:
            m.add(BatchNormalization())
        m.add(ResBlock(filters=c, bottlenecks=1, use_bn=use_bn, inner_relu=False))
        c = min(int(2 * c), 128)
        if _w <= w and _h <= h:
            break

    m.add(Conv2D(1, 5, 1, "same"))
    return m

def unet(input_shape=(128, 128, 3), name="unet"):
    def en_block(inputs, filters=64):
        _o = Conv2D(filters, 4, 2, "same")(inputs)
        _o = BatchNormalization()(_o)
        _o = LeakyReLU()(_o)
        return _o

    def de_block(inputs, encoding, filters=64):
        _u = tf.concat((inputs, encoding), axis=3)
        _u = UpSampling2D((2, 2))(_u)
        _u = Conv2D(filters, 4, 1, "same")(_u)
        _u = BatchNormalization()(_u)
        _u = ReLU()(_u)
        return _u

    i = keras.Input(shape=input_shape, dtype=tf.float32)
    l1 = en_block(i, 32)    # 64
    l2 = en_block(l1, 64)  # 32
    l3 = en_block(l2, 128)  # 16
    l4 = en_block(l3, 128)  # 8

    m = ResBlock(filters=128, activation=tf.nn.relu, bottlenecks=1, use_bn=True, inner_relu=False)(l3)  # 8
    m = ResBlock(128, tf.nn.relu, 1, True, inner_relu=False)(m)     # 8

    u1 = de_block(m, l4, 128)   # 16
    u2 = de_block(u1, l3, 128)  # 32
    u3 = de_block(u2, l2, 64)   # 64
    u4 = de_block(u3, l1, 32)   # 128

    o = Conv2D(input_shape[-1], 5, 1, "same", activation=keras.activations.tanh)(u4)
    unet = keras.Model(i, o, name=name)
    return unet

