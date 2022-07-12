from turtle import forward
import tensorflow as tf
import tensorflow.keras as keras

OP_DICT = {
    'none': lambda C, stride: Zero(stride),
    'conv_3x3': lambda C, stride: Conv(C, kernel_size=3, padding='same'),
    'conv_1x1': lambda C, stride: Conv(C, kernel_size=1, padding='valid'),
    'dconv_3x3': lambda C, stride: DConv(C, stride, kernel_size=3),
    'rel_attention': lambda C, stride: RelAttention(C, stride),
    'ffn': lambda C, stride: FeedForwardNet(C, stride)
}

class Conv(tf.Module):
    def __init__(self, C_out, kernel_size, padding):
        super().__init__()
        self.op = keras.Sequential()
        self.op.add(keras.layers.Conv2D(C_out, kernel_size, padding=padding))
        self.op.add(keras.layers.ReLU())

    def forward(self, x):
        return self.op(x)

class ReLUConvBN(tf.Module):
    def __init__(self, input_shape, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = keras.Sequential()
        self.op.add(keras.layers.ReLU())
        self.op.add(keras.layers.Conv2D(C_out, kernel_size, stride, padding, use_bias=False))
        self.op.add(keras.layers.BatchNormalization(momentum=0.15))

    def forward(self, x):
        return self.op(x)

class FactorizedReduce(tf.Module):
    def __init__(self, C_out, input_shape):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = keras.layers.ReLU()
        self.conv_1 = keras.layers.Conv2D(C_out // 2, 1, 2, 'valid')
        self.conv_2 = keras.layers.Conv2D(C_out // 2, 1, 2, 'valid')
        self.bn = keras.layers.BatchNormalization(momentum=0.15)

    def forward(self, x):
        x = self.relu(x)
        out = tf.concat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], 1)
        out = self.bn(out)
        return out 