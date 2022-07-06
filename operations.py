import tensorflow as tf
import tensorflow.keras as keras

class ReLUConvBN(tf.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = keras.Sequential()
        self.op.add(keras.layers.ReLU())
        self.op.add(keras.layers.Conv2D(C_out, ))