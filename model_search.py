from genotypes import Genotype
from genotypes import PRIMITIVES
from operations import *
import tensorflow as tf
import tensorflow.keras as keras

null_scope = tf.compat.v1.VariableScope("")

def MixedOp(x, C_out, stride, index, reduction):
    ops = []
    with tf.compat.v1.variable_scope(null_scope):
        with tf.compat.v1.variable_scope('arch_var', reuse=tf.compat.v1.AUTO_REUSE):
            weights = tf.compat.v1.get_variable("weight{}_{}".format(2 if reduction else 1, index), [len(PRIMITIVES)], initializer=tf.random_uniform_initializer(0,1e-3), regularizer=keras.regularizers.l2(0.0001))
    weights = tf.nn.softmax(weights)
    weights = tf.reshape(weights, [-1, 1, 1, 1])
    index = 0
    for prim in PRIMITIVES:
        op = OP_DICT[prim](x, C_out, stride)
        mask = [i == index for i in range(len(PRIMITIVES))]
        w_mask = tf.constant(mask, tf.bool)
        w = tf.boolean_mask(weights, w_mask)
        ops.append(op * w)
        index += 1
    return tf.add_n(ops)


def Cell(s0, s1, cells_n, multiplier, C_out, reduction, reduction_prev):
    if reduction_prev:
        s0 = FactorizedReduce(s0, C_out)
    else:
        s0 = ReLUConvBN(s0, C_out)
    s1 = ReLUConvBN(s0, C_out)

    states = [s0, s1]
    offset = 0
    for i in range(cells_n):
        temp = []
        for j in range(i + 2):
            stride = [2,2] if reduction and j < 2 else [1,1]
            temp.append(MixedOp(states[j], C_out, stride, offset+j, reduction))
        offset+=len(states)
        states.append(tf.add_n(temp))
    out=tf.concat(states[-multiplier:], axis=-1)
    return out

#def Model(x, y, TODO: this might not be needed: is_training (used in BatchNormalization), C_init, classes_n, layers_n, cells_n=4, multiplier=4, stem_multiplier=3, name='Model'):
def Model(x, C_init, classes_n, layers_n, cells_n=4, multiplier=4, stem_multiplier=3, name='Model'):
    C_curr = stem_multiplier * C_init
    # stem stages
    s0 = keras.layers.Conv2D(C_curr, (3,3), strides=(1,1), padding='same', activation='relu')(x)
    s0 = keras.layers.BatchNormalization()(s0)
    s1 = keras.layers.Conv2D(C_curr, (3,3), strides=(1,1), padding='same', activation='relu')(x)
    s1 = keras.layers.BatchNormalization()(s1)
    reduction_prev = False
    for i in range(layers_n):
        if i in [layers_n // 3, 2 * layers_n // 3]:
            C_curr *= 2
            reduction = True
        else:
            reduction = False
        s0, s1 = s1, Cell(s0, s1, cells_n, multiplier, C_curr, reduction, reduction_prev)
        reduction_prev = reduction
        out = keras.layers.GlobalAveragePooling2D()(s1)
        logits = keras.layers.Dense(classes_n)(out)
    #train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    return logits