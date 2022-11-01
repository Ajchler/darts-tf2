from genotypes import PRIMITIVES, Genotype
from operations import *
import tensorflow as tf
import tensorflow.keras as keras

class Cell(keras.layers.Layer):
    def __init__(self, n_nodes, multiplier, C_curr, reduction, reduction_prev, genotype):
        super().__init__()
        self._reduction_prev = reduction_prev
        self.reduction = reduction
        self._n_nodes = n_nodes
        self._multiplier = multiplier
        self.dropout = tf.keras.layers.Dropout(0.4)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_curr)
        else:
            self.preprocess0 = ReLUConvBN(C_curr, 1, 1, 'valid')
        self.preprocess1 = ReLUConvBN(C_curr, 1, 1, 'valid')

        gene = genotype.reduce if reduction else genotype.normal
        self.concat = genotype.reduce_concat if reduction else genotype.normal_concat
        self.ops = []
        self.indices = []
        for edges in gene:
            for op in edges:
                stride = [2, 2] if reduction and op[1] < 2 else [1, 1]
                self.ops.append(OP_DICT[op[0]](C_curr, stride))
                self.indices.append(op[1])

    def call(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._n_nodes):
            s_cur = 0
            for j in range(2):
                temp = self.dropout(states[self.indices[i * 2 +j]])
                s_cur += self.ops[i * 2 + j](temp)
            states.append(s_cur)
        return tf.concat([states[i] for i in self.concat], -1)

class Network(keras.Model):
    def __init__(self, C, criterion, n_classes, n_layers, genotype, n_nodes=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._n_classes = n_classes
        self._n_layers = n_layers
        self._n_nodes = n_nodes
        self._multiplier = multiplier
        self._criterion = criterion

        C_curr = C * stem_multiplier
        self.stem_1 = keras.layers.Conv2D(C_curr, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)
        self.stem_2 = keras.layers.BatchNormalization()

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        self.cells = []
        reduction_prev = False
        for i in range(n_layers):
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(n_nodes, multiplier, C_curr, reduction, reduction_prev, genotype)
            reduction_prev = reduction
            self.cells.append(cell)
            C_curr_out = C_curr * self._multiplier
            C_prev_prev, C_prev = C_prev, C_curr_out

        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        # using Dense layer from tensorflow as a replacement of nn.Linear()
        self.classifier = tf.keras.layers.Dense(n_classes, activation=None)

    def new(self):
        new_model = Network(self._C, self._criterion, self._input_shape, self._n_classes, self._n_layers)
        for x, y in zip(new_model.arch_params(), self.arch_params()):
            x.data.copy_(y.data)
        return new_model

    def call(self, x):
        op = self.stem_1(x)
        op = self.stem_2(op)
        s0 = s1 = op
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out)
        return logits

    def _loss(self, x, target):
        logits = self(x, training=True)
        return self._criterion(target, logits)