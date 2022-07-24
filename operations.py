import tensorflow as tf
import tensorflow.keras as keras

OP_DICT = {
    'none': lambda x, C, stride: Zero(x, stride),
    'conv_3x3': lambda x, C, stride: Conv(x, C, stride, kernel_size=3, padding='same'),
    'conv_1x1': lambda x, C, stride: Conv(x, C, stride, kernel_size=1, padding='valid'),
    'dconv_3x3': lambda x, C, stride: MBConv(x, C, C, stride, kernel_size=3),
    'rel_attention': lambda x, C, stride: RelAttention(x, C, stride),
    'ffn': lambda x, C, stride: FeedForwardNet(x, C, stride)
}

def RelAttention(x, C, stride, conv_short_cut=True, head_dim=32, drop_rate=0, activation='gelu'):
    preact = keras.layers.LayerNormalization(epsilon=1e-5)(x)

    if conv_short_cut:
        shortcut = keras.layers.MaxPool2D(stride, strides=stride, padding='same')(x) if stride[0] > 1 else x
        shortcut = keras.layers.Conv2D(C, [1,1], strides=1, padding='valid', use_bias=False)(shortcut)
    else:
        shortcut = x

    op = preact
    if stride != 1:
        op = keras.layers.MaxPool2D(pool_size=2, strides=stride, padding='same')(op)
    head_n = op.shape[-1] // head_dim
    op = keras.layers.MultiHeadAttention(head_n, head_dim, output_shape=C, use_bias=False)(op, op)
    op = keras.layers.Dropout(drop_rate)(op)
    return keras.layers.Add()([shortcut, op])

def FeedForwardNet(x, C, stride):
    op = keras.layers.Dense(C, activation='relu')(x)
    op = keras.layers.Dense(C)(op)
    op = keras.layers.MaxPool2D([1,1], stride, padding='same')(op)
    return op

def Zero(x, stride):
    return tf.zeros_like(x)[:, ::stride[0], ::stride[1],:]

def SEBlock(x, C, ratio=0.25):
    op = keras.layers.GlobalAveragePooling2D()(x)
    op = tf.expand_dims(op, axis=1)
    op = tf.expand_dims(op, axis=1)
    op = keras.layers.Conv2D(max(1, int(C * ratio)), kernel_size=[1,1], strides=[1,1], padding='same')(op)
    op = op * tf.nn.sigmoid(op)
    op = keras.layers.Conv2D(C, kernel_size=[1,1], strides=[1,1], padding='same')(op)
    op = tf.nn.sigmoid(op)
    return x * op

def MBConv(x, C_curr, C_out, stride, kernel_size, expand_ratio=1, drop_connect_rate=None):
    op = keras.layers.Conv2D(C_curr * expand_ratio, [1,1], [1,1], padding='same')(x)
    op = keras.layers.BatchNormalization()(op)
    op = keras.activations.gelu(op, approximate=True)
    op = op * tf.sigmoid(op)
    op = keras.layers.DepthwiseConv2D(kernel_size, stride, 'same')(op)
    op = keras.layers.BatchNormalization()(op)
    op = keras.activations.gelu(op, approximate=True)
    op = SEBlock(op, C_curr * expand_ratio)
    op = op * tf.sigmoid(op)
    op = keras.layers.Conv2D(C_out, [1,1], [1,1], padding='same')(op)
    if stride == 1 and C_curr == C_out:
        if drop_connect_rate:
            op = keras.layers.Dropout(drop_connect_rate)(op)
        op = keras.layers.Add([op, x])
    return op

def Conv(x, C, stride, kernel_size, padding):
    op = keras.layers.Conv2D(C, kernel_size, strides=stride, padding=padding)(x)
    op = keras.layers.ReLU()(op)
    return op

def ReLUConvBN(x, C):
    op = keras.layers.ReLU()(x)
    op = keras.layers.Conv2D(C, kernel_size=[1,1])(op)
    op = keras.layers.BatchNormalization()(op)
    return op

def FactorizedReduce(x, C):
    op = keras.layers.ReLU()(x)
    conv_1 = keras.layers.Conv2D(C // 2, [1,1], strides=[2,2])(op)
    conv_2 = keras.layers.Conv2D(C // 2, [1,1], strides=[2,2])(op[:,1:,1:,:])
    op = tf.concat([conv_1, conv_2], -1)
    op = keras.layers.BatchNormalization()(op)
    return op