from multiprocessing import pool
from numpy import short
import tensorflow as tf
import tensorflow.keras as keras

OP_DICT = {
    'none': lambda C, stride: Zero(stride),
    'conv_3x3': lambda C, stride: Conv(C, stride, kernel_size=3, padding='same'),
    'conv_1x1': lambda C, stride: Conv(C, stride, kernel_size=1, padding='valid'),
    'dconv_3x3': lambda C, stride: MBConv(C, C, stride, kernel_size=3),
    'rel_attention': lambda C, stride: RelAttention(C, stride),
    'ffn': lambda C, stride: FeedForwardNet(C, C, stride)
}

class RelAttention(keras.layers.Layer):
    def __init__(self, C_curr, stride, conv_short_cut=True, head_dim=32, drop_rate=0, activation='gelu'):
        super().__init__()
        self.C_curr = C_curr
        self.stride = stride
        self.conv_short_cut = conv_short_cut
        self.head_dim = head_dim
        self.drop_rate = drop_rate
        self.activation = activation
        self.head_n = C_curr // head_dim

        self.preact = keras.layers.LayerNormalization(epsilon=1e-5)
        self.max_pool_1 = keras.layers.MaxPool2D(pool_size=stride, strides=stride, padding='same')
        self.conv_1 = keras.layers.Conv2D(self.C_curr, kernel_size=[1,1], strides=1, padding='valid', use_bias=False)
        self.max_pool_2 = keras.layers.MaxPool2D(pool_size=2, strides=self.stride, padding='same')
        self.multihead_attn = keras.layers.MultiHeadAttention(self.head_n, self.head_dim, output_shape=C_curr, use_bias=False)
        self.dropout = keras.layers.Dropout(self.drop_rate)
        self.add = keras.layers.Add()

    def call(self, x):
        preact = self.preact(x)
        if self.conv_short_cut:
            shortcut = self.max_pool_1(x) if self.stride[0] > 1 else x
            shortcut = self.conv_1(shortcut)
        else:
            shortcut = x

        if self.stride != 1:
            op = self.max_pool_2(preact)
        op = self.multihead_attn(op, op)
        op = self.dropout(op)
        return self.add([shortcut, op])

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

class Zero(keras.layers.Layer):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def call(self, x):
        return tf.zeros_like(x)[:, ::self.stride[0], ::self.stride[1], :]

class SEBlock(keras.layers.Layer):
    def __init__(self, C_curr, ratio=0.25):
        super().__init__()
        self._reduced_channels = max(1, int(C_curr * ratio))
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = tf.keras.layers.Conv2D(self._reduced_channels, kernel_size=1, strides=1, padding='same')
        self.expand_conv = tf.keras.layers.Conv2D(C_curr, kernel_size=1, strides=1, padding='same')

    def call(self, x):
        out = self.global_pooling(x)
        out = tf.expand_dims(out, axis=1)
        out = tf.expand_dims(out, axis=1)
        out = self.reduce_conv(out)
        out = out * tf.nn.sigmoid(out)
        out = self.expand_conv(out)
        out = tf.nn.sigmoid(out)
        return x * out

class MBConv(keras.layers.Layer):
    def __init__(self, C_curr, C_out, stride, kernel_size, expand_ratio=1, drop_connect_rate=None):
        super().__init__()
        self._stride = stride
        self._C_curr = C_curr
        self._C_out = C_out
        self._stride = stride
        self._drop_connect_rate = drop_connect_rate

        self.conv_1 = keras.layers.Conv2D(C_curr * expand_ratio, 1, 1, padding='same')
        self.bn_1 = keras.layers.BatchNormalization()
        self.depth_wise_conv = keras.layers.DepthwiseConv2D(kernel_size, stride, 'same')
        self.bn_2 = keras.layers.BatchNormalization()
        self.se_block = SEBlock(C_curr * expand_ratio)
        self.conv_2 = keras.layers.Conv2D(C_out, 1, 1, padding='same')
        self.bn_3 = keras.layers.BatchNormalization()
        self.add = keras.layers.Add()

        # dropout is probably useless since this value will probably never be
        # used, but in case it was in some of the experiments, it stays for now
        self.droupout = keras.layers.Dropout(drop_connect_rate)

    def call(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = tf.nn.gelu(out, approximate=True)
        out = out * tf.sigmoid(out)
        out = self.depth_wise_conv(out)
        out = self.bn_2(out)
        out = tf.nn.gelu(out, approximate=True)
        out = self.se_block(out)
        out = out * tf.sigmoid(out)
        out = self.conv_2(out)
        #out = self.bn_3(out)
        #out = tf.nn.gelu(out, approximate=True)

        if self._stride == 1 and self._C_curr == self._C_out:
            if self._drop_connect_rate:
                out = self.droupout(out)
            out = self.add([out, x])
        return out

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

    def call(self, x):
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

    def call(self, x):
        x = self.relu(x)
        out = tf.concat([self.conv_1(x), self.conv_2(x[:,1:,1:,:])], -1)
        out = self.bn(out)
        return out