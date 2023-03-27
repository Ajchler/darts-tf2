import tensorflow as tf
from python.keras.layers.fake_convolutional import FakeApproxConv2D, FakeApproxDepthwiseConv2D

OP_DICT = {
    'none' : lambda C_curr, C_prev, stride, approx: Zero(stride),
    'avg_pool_3x3' : lambda C_curr, C_prev, stride, approx: tf.keras.layers.AveragePooling2D(3, strides=stride, padding='same'),
    'max_pool_3x3' : lambda C_curr, C_prev, stride, approx: tf.keras.layers.MaxPool2D(3, strides=stride, padding='same'),
    'skip_connect' : lambda C_curr, C_prev, stride, approx: Identity() if stride[0] == 1 else FactorizedReduce(C_curr),
    #'sep_conv_3x3' : lambda C_curr, C_prev, stride, approx: keras.layers.SeparableConv2D(C_curr, kernel_size=3, strides=stride, padding='same'),#SepConv(C_curr, C_prev, 3, stride),
    #'sep_conv_5x5' : lambda C_curr, C_prev, stride, approx: keras.layers.SeparableConv2D(C_curr, kernel_size=5, strides=stride, padding='same'),#SepConv(C_curr, C_prev, 5, stride),
    'sep_conv_3x3' : lambda C_curr, C_prev, stride, approx: SepConv(C_curr, C_prev, 3, stride, approx),
    'sep_conv_5x5' : lambda C_curr, C_prev, stride, approx: SepConv(C_curr, C_prev, 5, stride, approx),
    #'sep_conv_7x7' : lambda C_curr, C_prev, stride: keras.layers.SeparableConv2D(C_curr, kernel_size=7, strides=stride, padding='same'),#SepConv(C_curr, C_prev, 7, stride),
    #'dil_conv_3x3' : lambda C_curr, C_prev, stride, approx: keras.layers.SeparableConv2D(C_curr, kernel_size=3, strides=stride, padding='same', dilation_rate=2),#DilConv(C_curr, 3, stride, 2),
    #'dil_conv_5x5' : lambda C_curr, C_prev, stride, approx: keras.layers.SeparableConv2D(C_curr, kernel_size=5, strides=stride, padding='same', dilation_rate=2),#DilConv(C_curr, 5, stride, 2),
    'dil_conv_3x3' : lambda C_curr, C_prev, stride, approx: DilConv(C_curr, 3, stride, 2, approx),
    'dil_conv_5x5' : lambda C_curr, C_prev, stride, approx: DilConv(C_curr, 5, stride, 2, approx),
}

class DilConv(tf.keras.layers.Layer):
    def __init__(self, C_curr, kernel_size, stride, rate, approx):
        super().__init__()
        self.relu = tf.keras.layers.ReLU()
        self.dw = FakeApproxDepthwiseConv2D(kernel_size, (1, 1), dilation_rate=rate,padding='same', approx_mul_table_file='mul8u_1JFF.bin')
        self.pw = FakeApproxConv2D(C_curr, 1, strides=stride, padding='same', approx_mul_table_file='mul8u_1JFF.bin')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        x = self.relu(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return x

class SepConv(tf.keras.layers.Layer):
    def __init__(self, C_curr, C_prev, kernel_size, stride, approx):
        super().__init__()
        self.relu = tf.keras.layers.ReLU()
        self.dw = FakeApproxDepthwiseConv2D(kernel_size, stride, padding='same', approx_mul_table_file='mul8u_1JFF.bin')
        self.pw = FakeApproxConv2D(C_curr, 1, padding='same', approx_mul_table_file='mul8u_1JFF.bin')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        x = self.relu(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return x

class Identity(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, training=None):
        return x

class Zero(tf.keras.layers.Layer):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def call(self, x, training=None):
        return tf.zeros_like(x)[:, ::self.stride[0], ::self.stride[1], :]

class ReLUConvBN(tf.keras.layers.Layer):
    def __init__(self, C_out, kernel_size, stride, padding):
        super().__init__()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(C_out, kernel_size, stride, padding, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.15)

    def call(self, x, training=None):
        op = self.relu(x)
        op = self.conv(op)
        return self.bn(op)

class FactorizedReduce(tf.keras.layers.Layer):
    def __init__(self, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = tf.keras.layers.ReLU()
        self.conv_1 = tf.keras.layers.Conv2D(C_out // 2, 1, 2, 'valid')
        self.conv_2 = tf.keras.layers.Conv2D(C_out // 2, 1, 2, 'valid')
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.15)

    def call(self, x, training=None):
        x = self.relu(x)
        out = tf.concat([self.conv_1(x), self.conv_2(x[:,1:,1:,:])], -1)
        out = self.bn(out)
        return out