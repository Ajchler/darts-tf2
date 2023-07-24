"""
Module providing many operations used in preprocessing and in search space

This code is part of reimplementation of original DARTS
and is based on it, the original implementation
can be found here: https://github.com/quark0/darts
and is licensed under Apache 2.0

Author: Vojtech Eichler
Date: April 2023
"""

import tensorflow as tf
import tensorflow.keras as keras

OP_DICT = {
    'none' : lambda C_curr, stride, normalize: Zero(stride),
    'avg_pool_3x3' : lambda C_curr, stride, normalize: AvgPool(3, stride=stride, normalize=normalize),
    'max_pool_3x3' : lambda C_curr, stride, normalize: MaxPool(3, stride=stride, normalize=normalize),
    'skip_connect' : lambda C_curr, stride, normalize: Identity() if stride[0] == 1 else FactorizedReduce(C_curr),
    'sep_conv_3x3' : lambda C_curr, stride, normalize: SepConv(C_curr, 3, stride),
    'sep_conv_5x5' : lambda C_curr, stride, normalize: SepConv(C_curr, 5, stride),
    'dil_conv_3x3' : lambda C_curr, stride, normalize: DilConv(C_curr, 3, stride, 2),
    'dil_conv_5x5' : lambda C_curr, stride, normalize: DilConv(C_curr, 5, stride, 2),
}

class AvgPool(keras.layers.Layer):
    """Average pooling operation with batch normalization

    """
    def __init__(self, kernel_size, stride, normalize):
        super().__init__()
        self.normalize = normalize
        self.pool = keras.layers.AveragePooling2D(kernel_size, strides=stride, padding='same')
        if normalize:
            self.bn = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        x = self.pool(x)
        if self.normalize:
            x = self.bn(x)
        return x

class MaxPool(keras.layers.Layer):
    """Max pooling operation with batch normalization

    """
    def __init__(self, kernel_size, stride, normalize):
        super().__init__()
        self.normalize = normalize
        self.pool = keras.layers.MaxPooling2D(kernel_size, strides=stride, padding='same')
        if normalize:
            self.bn = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        x = self.pool(x)
        if self.normalize:
            x = self.bn(x)
        return x

class DilConv(keras.layers.Layer):
    """Separable convolution operation with dilation applied on depthwise convolution
    with ReLU activation at the beginning of the operation and batch normalization at the end

    """
    def __init__(self, C_curr, kernel_size, stride, rate):
        super().__init__()
        self.relu = keras.layers.ReLU()
        self.dw = keras.layers.DepthwiseConv2D(kernel_size, (1, 1), dilation_rate=rate, padding='same')
        self.pw = keras.layers.Conv2D(C_curr, 1, stride, padding='same')
        self.bn = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        x = self.relu(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return x

class SepConv(keras.layers.Layer):
    """Separable convolution operation with ReLU activation
    at the beginning of the operation and batch normalization at the end

    """
    def __init__(self, C_curr, kernel_size, stride):
        super().__init__()
        self.relu = keras.layers.ReLU()
        self.dw = keras.layers.DepthwiseConv2D(kernel_size, stride, padding='same')
        self.pw = keras.layers.Conv2D(C_curr, 1, padding='same')
        self.bn = keras.layers.BatchNormalization()
        self.relu2 = keras.layers.ReLU()
        self.dw2 = keras.layers.DepthwiseConv2D(kernel_size, 1, padding='same')
        self.pw2 = keras.layers.Conv2D(C_curr, 1, padding='same')
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        x = self.relu(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.relu2(x)
        x = self.dw2(x)
        x = self.pw2(x)
        x = self.bn2(x)
        return x

class Identity(keras.layers.Layer):
    """Shortcut operation for normal cells

    """
    def __init__(self):
        super().__init__()

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        return x

class Zero(keras.layers.Layer):
    """Special zero operation representing no connection

    """
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        return tf.zeros_like(x)[:, ::self.stride[0], ::self.stride[1], :]


class ReLUConvBN(keras.layers.Layer):
    """Operation which first applies ReLU activation function, then convolution
    and batch normalization

    """
    def __init__(self, C_out, kernel_size, stride, padding):
        super().__init__()
        self.relu = keras.layers.ReLU()
        self.conv = keras.layers.Conv2D(C_out, kernel_size, stride, padding, use_bias=False)
        self.bn = keras.layers.BatchNormalization(momentum=0.15)

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        op = self.relu(x)
        op = self.conv(op)
        return self.bn(op)

class FactorizedReduce(keras.layers.Layer):
    """Shortcut operation for reduction cells

    """
    def __init__(self, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = keras.layers.ReLU()
        self.conv_1 = keras.layers.Conv2D(C_out // 2, 1, 2, 'valid')
        self.conv_2 = keras.layers.Conv2D(C_out // 2, 1, 2, 'valid')
        self.bn = keras.layers.BatchNormalization(momentum=0.15)

    def call(self, x, training=None):
        """Forward pass method

        Args:
            x : Input images
            training : Specify training or inference mode. Defaults to None.

        Returns:
            Feature maps
        """
        x = self.relu(x)
        out = tf.concat([self.conv_1(x), self.conv_2(x[:,1:,1:,:])], -1)
        out = self.bn(out)
        return out