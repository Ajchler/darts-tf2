from genotypes import PRIMITIVES, Genotype
from operations import *
import tensorflow as tf
import tensorflow.keras as keras

class MixedOp(keras.layers.Layer):
    def __init__(self, C_curr, stride):
        super().__init__()
        self._ops = []
        for prim in PRIMITIVES:
            op = OP_DICT[prim](C_curr, stride)
            self._ops.append(op)

    def call(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(keras.layers.Layer):
    def __init__(self, n_nodes, multiplier, C_curr, C_prev, C_prev_prev, reduction, reduction_prev):
        super().__init__()
        self._reduction_prev = reduction_prev
        self.reduction = reduction
        self._n_nodes = n_nodes
        self._multiplier = multiplier
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_curr)
        else:
            self.preprocess0 = ReLUConvBN(C_curr, 1, 1, 'valid')
        self.preprocess1 = ReLUConvBN(C_curr, 1, 1, 'valid')

        self._ops = []
        self._bns = []
        for i in range(self._n_nodes):
            for j in range (i + 2):
                stride = [2,2] if reduction and j < 2 else [1,1]
                self._ops.append(MixedOp(C_curr, stride))

    def call(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._n_nodes):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return tf.concat(states[-self._multiplier:], -1)

class Network(keras.Model):
    def __init__(self, C, criterion, n_classes, n_layers, n_nodes=4, multiplier=4, stem_multiplier=3):
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

            cell = Cell(n_nodes, multiplier, C_curr, C_prev, C_prev_prev, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_curr_out = C_curr * self._multiplier
            C_prev_prev, C_prev = C_prev, C_curr_out

        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        # using Dense layer from tensorflow as a replacement of nn.Linear()
        self.classifier = tf.keras.layers.Dense(n_classes, activation=None)

        self._initialize_alphas()

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
            if cell.reduction:
                weights = tf.nn.softmax(self.alphas_reduce, axis=-1)
            else:
                weights = tf.nn.softmax(self.alphas_normal, axis=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out)
        return logits

    def _loss(self, x, target):
        logits = self(x, training=True)
        return self._criterion(target, logits)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._n_nodes) for n in range(i + 2))
        n_ops = len(PRIMITIVES)

        self.alphas_normal = tf.Variable(1e-3*tf.random.uniform([k, n_ops]), trainable=False)
        self.alphas_reduce = tf.Variable(1e-3*tf.random.uniform([k, n_ops]), trainable=False)
        self._arch_params = [self.alphas_normal, self.alphas_reduce]

    def arch_params(self):
        return self._arch_params

    def _parse_alphas(self, weights):
        weights = tf.constant(weights)
        gene = []
        n = 2
        start = 0
        for i in range(self._n_nodes):
            end = start + n
            W = weights[start:end]
            edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
            node_gene = []
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                node_gene.append((PRIMITIVES[k_best], j))
            gene.append(node_gene)
            start = end
            n += 1
        return gene

    def genotypes(self):
        gene_normal = self._parse_alphas(tf.nn.softmax(self.alphas_normal))
        gene_reduce = self._parse_alphas(tf.nn.softmax(self.alphas_reduce))

        concat = range(2 + self._n_nodes - self._multiplier, self._n_nodes + 2)
        return Genotype(gene_normal, concat, gene_reduce, concat)
