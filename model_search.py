from genotypes import Genotype
from genotypes import PRIMITIVES
from operations import *
import tensorflow as tf
import tensorflow.keras as keras

#class MixedOp(keras.layers.Layer):
#    def __init__(self, C_curr, stride):
#        super().__init__()
#        self._ops = []
#        for prim in PRIMITIVES:
#            op = OP_DICT[prim](C_curr, stride)
#            self._ops.append(op)
#
#    def call(self, x, weights):
#        return sum(w * op(x) for w, op in zip(weights, self._ops))
#
#class Cell(keras.layers.Layer):
#    def __init__(self, n_nodes, multiplier, C_curr, C_prev, C_prev_prev, reduction, reduction_prev, input_shape):
#        super().__init__()
#        self.reduction = reduction
#        self._n_nodes = n_nodes
#        self._multiplier = multiplier
#        if reduction_prev:
#            self.preprocess0 = FactorizedReduce(C_curr, input_shape)
#        else:
#            self.preprocess0 = ReLUConvBN(input_shape, C_curr, 1, 1, 'valid')
#        self.preprocess1 = ReLUConvBN(input_shape, C_curr, 1, 1, 'valid')
#
#        self._ops = []
#        self._bns = []
#        for i in range(self._n_nodes):
#            for j in range (i + 2):
#                stride = 2 if reduction and j < 2 else 1
#                self._ops.append(MixedOp(C_curr, stride))
#
#    def call(self, s0, s1, weights):
#        s0 = self.preprocess0(s0)
#        s1 = self.preprocess1(s1)
#
#        states = [s0, s1]
#        offset = 0
#        for i in range(self._n_nodes):
#            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
#            offset += len(states)
#            states.append(s)
#
#        return tf.concat(states[-self._multiplier:], 1)
#
#class Network(keras.Model):
#    def __init__(self, C, criterion, input_shape, n_classes, n_layers, n_nodes=4, multiplier=4, stem_multiplier=3):
#        super(Network, self).__init__()
#        self._C = C
#        self._input_shape = input_shape
#        self._n_classes = n_classes
#        self._n_layers = n_layers
#        self._n_nodes = n_nodes
#        self._multiplier = multiplier
#        self._criterion = criterion
#
#        C_curr = C * stem_multiplier
#        self.stem = keras.Sequential()
#        self.stem.add(keras.layers.Conv2D(C_curr, kernel_size=(3,3), padding='same', strides=1, use_bias=False))
#        self.stem.add(keras.layers.BatchNormalization())
#
#        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
#
#        self.cells = []
#        reduction_prev = False
#        for i in range(n_layers):
#            if i in [n_layers // 3, 2 * n_layers // 3]:
#                C_curr *= 2
#                reduction = True
#            else:
#                reduction = False
#
#            cell = Cell(n_nodes, multiplier, C_curr, C_prev, C_prev_prev, reduction, reduction_prev, input_shape)
#            reduction_prev = reduction
#            self.cells.append(cell)
#            C_curr_out = C_curr * n_nodes
#            C_prev_prev, C_prev = C_prev, C_curr_out
#
#        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
#        # using Dense layer from tensorflow as a replacement of nn.Linear()
#        self.classifier = tf.keras.layers.Dense(n_classes, activation=None)
#
#        self._initialize_alphas()
#
#    def new(self):
#        new_model = Network(self._C, self._criterion, self._input_shape, self._n_classes, self._n_layers)
#        for x, y in zip(new_model.arch_params(), self.arch_params()):
#            x.data.copy_(y.data)
#        return new_model
#
#    def call(self, x):
#        s0 = s1 = self.stem(x)
#        for i, cell in enumerate(self.cells):
#            if cell.reduction:
#                weights = tf.nn.softmax(self.alphas_reduce, axis=-1)
#            else:
#                weights = tf.nn.softmax(self.alphas_normal, axis=-1)
#            s0, s1 = s1, cell(s0, s1, weights)
#        out = self.global_pooling(s1)
#        logits = self.classifier(tf.reshape(out, [tf.size(out)[0], -1]))
#        return logits
#
#    def _loss(self, x, target):
#        logits = self(x)
#        return self._criterion(logits, target)
#
#    def _initialize_alphas(self):
#        k = sum(1 for i in range(self._n_nodes) for n in range(i + 2))
#        n_ops = len(PRIMITIVES)
#
#        self.alphas_normal = 1e-3*tf.random.uniform([k, n_ops])
#        self.alphas_reduce = 1e-3*tf.random.uniform([k, n_ops])
#        self._arch_params = [self.alphas_normal, self.alphas_reduce]
#
#    def arch_params(self):
#        return self._arch_params

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

#def Model(x, y, is_training, C_init, classes_n, layers_n, cells_n=4, multiplier=4, stem_multiplier=3, name='Model'):
def Model(x, is_training, C_init, classes_n, layers_n, cells_n=4, multiplier=4, stem_multiplier=3, name='Model'):
    C_curr = stem_multiplier * C_init
    # stem stages
    s0 = keras.layers.Conv2D(C_curr, (3,3), strides=(1,1), padding='same', activation='relu')(x)
    s0 = keras.layers.BatchNormalization(trainable=is_training)(s0)
    s1 = keras.layers.Conv2D(C_curr, (3,3), strides=(1,1), padding='same', activation='relu')(x)
    s1 = keras.layers.BatchNormalization(trainable=is_training)(s1)
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
        logits = keras.layers.Dense(classes_n)
    #train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    return logits