import tensorflow as tf
import tensorflow.keras as keras
#from python.keras.layers.fake_convolutional import FakeApproxConv2D

OP_DICT = {
    'none': lambda C, _, stride, normalize: Zero(stride),
    'avg_pool_3x3' : lambda C_curr, C_prev, stride, normalize: AvgPool(3, stride=stride, normalize=normalize),
    'max_pool_3x3' : lambda C_curr, C_prev, stride, normalize: MaxPool(3, stride=stride, normalize=normalize),
    'sep_conv_3x3': lambda C, _, stride, normalize: SepConv(C, 5, stride),
    'rel_attention': lambda C, _, stride, normalize: RelAttention(C, stride),
    'ffn': lambda C, _, stride, normalize: FeedForwardNet(C, C, stride),
    'skip_connect' : lambda C_curr, C_prev, stride, normalize: Identity() if stride[0] == 1 else FactorizedReduce(C_curr),
}

#OP_DICT = {
#    'none' : lambda C_curr, C_prev, stride, approx: Zero(stride),
#    'avg_pool_3x3' : lambda C_curr, C_prev, stride, approx: keras.layers.AveragePooling2D(3, strides=stride, padding='same'),
#    'max_pool_3x3' : lambda C_curr, C_prev, stride, approx: keras.layers.MaxPool2D(3, strides=stride, padding='same'),
#    'skip_connect' : lambda C_curr, C_prev, stride, approx: Identity() if stride[0] == 1 else FactorizedReduce(C_curr),
#    #'sep_conv_3x3' : lambda C_curr, C_prev, stride, approx: keras.layers.SeparableConv2D(C_curr, kernel_size=3, strides=stride, padding='same'),#SepConv(C_curr, C_prev, 3, stride),
#    #'sep_conv_5x5' : lambda C_curr, C_prev, stride, approx: keras.layers.SeparableConv2D(C_curr, kernel_size=5, strides=stride, padding='same'),#SepConv(C_curr, C_prev, 5, stride),
#    'sep_conv_3x3' : lambda C_curr, C_prev, stride, approx: SepConv(C_curr, 3, stride, approx),
#    'sep_conv_5x5' : lambda C_curr, C_prev, stride, approx: SepConv(C_curr, 5, stride, approx),
#    #'sep_conv_7x7' : lambda C_curr, C_prev, stride: keras.layers.SeparableConv2D(C_curr, kernel_size=7, strides=stride, padding='same'),#SepConv(C_curr, C_prev, 7, stride),
#    #'dil_conv_3x3' : lambda C_curr, C_prev, stride, approx: keras.layers.SeparableConv2D(C_curr, kernel_size=3, strides=stride, padding='same', dilation_rate=2),#DilConv(C_curr, 3, stride, 2),
#    #'dil_conv_5x5' : lambda C_curr, C_prev, stride, approx: keras.layers.SeparableConv2D(C_curr, kernel_size=5, strides=stride, padding='same', dilation_rate=2),#DilConv(C_curr, 5, stride, 2),
#    'dil_conv_3x3' : lambda C_curr, C_prev, stride, approx: DilConv(C_curr, 3, stride, 2, approx),
#    'dil_conv_5x5' : lambda C_curr, C_prev, stride, approx: DilConv(C_curr, 5, stride, 2, approx),
#}

class DilConv(keras.layers.Layer):
    def __init__(self, C_curr, kernel_size, stride, rate):
        super().__init__()
        self.relu = keras.layers.ReLU()
        #if approx:
        #    self.sep_conv = FakeApproxConv2D(C_curr, kernel_size, stride, 'same')
        #else:
        self.sep_conv = keras.layers.SeparableConv2D(C_curr, kernel_size, stride, dilation_rate=rate, padding='same')
        self.bn = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        x = self.relu(x)
        x = self.sep_conv(x)
        x = self.bn(x)
        return x

class SepConv(keras.layers.Layer):
    def __init__(self, C_curr, kernel_size, stride):
        super().__init__()
        self.relu = keras.layers.ReLU()
        #if approx:
        #    self.sep_conv = FakeApproxConv2D(C_curr, kernel_size, stride, 'same')
        #else:
        self.sep_conv = keras.layers.SeparableConv2D(C_curr, kernel_size=kernel_size, strides=stride, padding='same')
        self.bn = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        x = self.relu(x)
        x = self.sep_conv(x)
        x = self.bn(x)
        return x

class RelAttention(keras.layers.Layer):
    def __init__(self, C_curr, stride, conv_short_cut=True, head_dim=32, drop_rate=0, activation='gelu'):
        super().__init__()
        self.C_curr = C_curr
        self.stride = stride
        self.conv_short_cut = conv_short_cut
        self.head_dim = head_dim
        self.drop_rate = drop_rate
        self.activation = activation
        self.head_n = 2
        #self.head_n = C_curr // head_dim
        #self.head_n = self.head_n if self.head_n > 0 else 1

        self.preact = keras.layers.LayerNormalization(epsilon=1e-5)
        self.max_pool_2 = keras.layers.MaxPool2D(pool_size=2, strides=self.stride, padding='same')
        self.multihead_attn = keras.layers.MultiHeadAttention(self.head_n, self.head_dim, output_shape=C_curr, use_bias=False)
        #self.dropout = keras.layers.Dropout(self.drop_rate)

    def call(self, x):
        preact = self.preact(x)
        if self.stride != 1:
            op = self.max_pool_2(preact)
        op = self.multihead_attn(op, op)
        #op = self.dropout(op)
        return op

class FeedForwardNet(keras.layers.Layer):
    def __init__(self, C_curr, C_out, stride):
        super().__init__()
        self.dense_1 = keras.layers.Dense(C_out, activation='relu')
        self.dense_2 = keras.layers.Dense(C_curr)
        self.maxpool = keras.layers.MaxPool2D(1, stride, padding='same')

    def call(self, x):
        op = self.dense_1(x)
        op = self.dense_2(op)
        return self.maxpool(op)

class Identity(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, training=None):
        return x

class Zero(keras.layers.Layer):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def call(self, x, training=None):
        return tf.zeros_like(x)[:, ::self.stride[0], ::self.stride[1], :]

class AvgPool(keras.layers.Layer):
    def __init__(self, kernel_size, stride, normalize):
        super().__init__()
        self.normalize = normalize
        self.pool = keras.layers.AveragePooling2D(kernel_size, strides=stride, padding='same')
        if normalize:
            self.bn = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        x = self.pool(x)
        if self.normalize:
            x = self.bn(x)
        return x

class MaxPool(keras.layers.Layer):
    def __init__(self, kernel_size, stride, normalize):
        super().__init__()
        self.normalize = normalize
        self.pool = keras.layers.MaxPooling2D(kernel_size, strides=stride, padding='same')
        if normalize:
            self.bn = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        x = self.pool(x)
        if self.normalize:
            x = self.bn(x)
        return x

class Conv(keras.layers.Layer):
    def __init__(self, C_out, stride, kernel_size, padding):
        super().__init__()
        self.conv = keras.layers.Conv2D(C_out, kernel_size, strides=stride, padding=padding)
        self.relu = keras.layers.ReLU()

    def call(self, x):
        op = self.conv(x)
        return self.relu(op)

class ReLUConvBN(keras.layers.Layer):
    def __init__(self, C_out, kernel_size, stride, padding):
        super().__init__()
        self.relu = keras.layers.ReLU()
        self.conv = keras.layers.Conv2D(C_out, kernel_size, stride, padding, use_bias=False)
        self.bn = keras.layers.BatchNormalization(momentum=0.15)

    def call(self, x, training=None):
        op = self.relu(x)
        op = self.conv(op)
        return self.bn(op)

class FactorizedReduce(keras.layers.Layer):
    def __init__(self, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = keras.layers.ReLU()
        self.conv_1 = keras.layers.Conv2D(C_out // 2, 1, 2, 'valid')
        self.conv_2 = keras.layers.Conv2D(C_out // 2, 1, 2, 'valid')
        self.bn = keras.layers.BatchNormalization(momentum=0.15)

    def call(self, x, training=None):
        x = self.relu(x)
        out = tf.concat([self.conv_1(x), self.conv_2(x[:,1:,1:,:])], -1)
        out = self.bn(out)
        return out