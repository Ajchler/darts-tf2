import tensorflow as tf
import tensorflow.keras as keras

OP_DICT = {
    'none': lambda C, stride: Zero(stride),
    'conv_3x3': lambda C, stride: Conv(C, kernel_size=3, padding='same'),
    'conv_1x1': lambda C, stride: Conv(C, kernel_size=1, padding='valid'),
    'dconv_3x3': lambda C, stride: MBConv(C, C, stride, kernel_size=3),
    'rel_attention': lambda C, stride: RelAttention(C, stride),
    'ffn': lambda C, stride: FeedForwardNet(C, stride)
}

class Zero(tf.Module):
    def __init__(self, stride):
        self.stride = stride

    def __call__(self, x):
        if self.stride == 1:
            return tf.math.multiply(x, 0.0)
        return tf.math.multiply(x[:,::self.stride,::self.stride,:], 0.0)

class SEBlock(tf.Module):
    def __init__(self, C_curr, ratio=0.25):
        super().__init__()
        self._reduced_channels = max(1, int(C_curr * ratio))
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = tf.keras.layers.Conv2D(self._reduced_channels, kernel_size=1, strides=1, padding='same')
        self.expand_conv = tf.keras.layers.Conv2D(C_curr, kernel_size=1, strides=1, padding='same')

    def __call__(self, x):
        out = self.global_pooling(x)
        out = tf.expand_dims(out, axis=1)
        out = tf.expand_dims(out, axis=1)
        out = self.reduce_conv(out)
        out = out * tf.nn.sigmoid(out)
        out = self.expand_conv(out)
        out = tf.nn.sigmoid(out)
        return x * out

class MBConv(tf.Module):
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

        # dropout is probably useless since this value will probably never be
        # used, but in case it was in some of the experiments, it stays for now
        self.droupout = keras.layers.Dropout(drop_connect_rate)

    def __call__(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = out * tf.sigmoid(out)
        out = self.depth_wise_conv(out)
        out = self.bn2(out)
        out = self.se_block(out)
        out = out * tf.sigmoid(out)
        out = self.conv_2(out)
        out = self.bn_3(out)
        if self._stride == 1 and self._C_curr == self._C_out:
            if self._drop_connect_rate:
                out = self.droupout(out)
            out = tf.keras.layers.add([out, x])
        return out

class Conv(tf.Module):
    def __init__(self, C_out, kernel_size, padding):
        super().__init__()
        self.op = keras.Sequential()
        self.op.add(keras.layers.Conv2D(C_out, kernel_size, padding=padding))
        self.op.add(keras.layers.ReLU())

    def __call__(self, x):
        return self.op(x)

class ReLUConvBN(tf.Module):
    def __init__(self, input_shape, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = keras.Sequential()
        self.op.add(keras.layers.ReLU())
        self.op.add(keras.layers.Conv2D(C_out, kernel_size, stride, padding, use_bias=False))
        self.op.add(keras.layers.BatchNormalization(momentum=0.15))

    def __call__(self, x):
        return self.op(x)

class FactorizedReduce(tf.Module):
    def __init__(self, C_out, input_shape):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = keras.layers.ReLU()
        self.conv_1 = keras.layers.Conv2D(C_out // 2, 1, 2, 'valid')
        self.conv_2 = keras.layers.Conv2D(C_out // 2, 1, 2, 'valid')
        self.bn = keras.layers.BatchNormalization(momentum=0.15)

    def __call__(self, x):
        x = self.relu(x)
        out = tf.concat([self.conv_1(x), self.conv_2(x[:,1:,1:,:])], 1)
        out = self.bn(out)
        return out