from genotypes import PRIMITIVES, Genotype
from operations import *
import tensorflow as tf
import tensorflow.keras as keras

class AuxiliaryHeadCifar(keras.layers.Layer):
    def __init__(self, n_classes):
        super().__init__()
        self.relu1 = keras.layers.ReLU()
        self.avg_pool = keras.layers.AveragePooling2D(5, strides=3, padding='valid')
        self.conv1 = keras.layers.Conv2D(128, kernel_size=1, use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.relu2 = keras.layers.ReLU()
        self.conv2 = keras.layers.Conv2D(768, 2, use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()
        self.relu3 = keras.layers.ReLU()
        self.criterion = keras.layers.Dense(n_classes)

    def call(self, x):
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.criterion(x)
        return x

class Cell(keras.layers.Layer):
    def __init__(self, n_nodes, multiplier, C_curr, C_prev, reduction, reduction_prev, genotype, drop_rate):
        super().__init__()
        self._reduction_prev = reduction_prev
        self.reduction = reduction
        self._n_nodes = n_nodes
        self._multiplier = multiplier
        self.dropout = tf.keras.layers.Dropout(drop_rate)

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
                self.ops.append(OP_DICT[op[0]](C_curr, C_prev, stride))
                self.indices.append(op[1])

    def call(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._n_nodes):
            s_cur = 0
            for j in range(2):
                s_tmp = self.ops[i * 2 + j](states[self.indices[i * 2 +j]])
                if not isinstance(self.ops[i * 2 + j], Identity):
                    s_cur += self.dropout(s_tmp)
                else:
                    s_cur += s_tmp
            states.append(s_cur)
        return tf.concat([states[i] for i in self.concat], -1)

class Network(keras.Model):
    def __init__(self, C, criterion, n_classes, n_layers, genotype, drop_rate, n_nodes=4, multiplier=4, stem_multiplier=3, auxiliary=False):
        super(Network, self).__init__()
        self._C = C
        self._n_classes = n_classes
        self._n_layers = n_layers
        self._n_nodes = n_nodes
        self._multiplier = multiplier
        self._criterion = criterion
        self._drop_rate = drop_rate
        self._auxiliary = auxiliary

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

            cell = Cell(n_nodes, multiplier, C_curr, C_prev, reduction, reduction_prev, genotype, drop_rate)
            reduction_prev = reduction
            self.cells.append(cell)
            C_curr_out = C_curr * self._multiplier
            C_prev_prev, C_prev = C_prev, C_curr_out

        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        # using Dense layer from tensorflow as a replacement of nn.Linear()
        self.classifier = tf.keras.layers.Dense(n_classes, activation=None)

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCifar(n_classes)

    def new(self):
        new_model = Network(self._C, self._criterion, self._input_shape, self._n_classes, self._n_layers)
        for x, y in zip(new_model.arch_params(), self.arch_params()):
            x.data.copy_(y.data)
        return new_model

    def call(self, x, training=True):
        logits_aux = None
        op = self.stem_1(x)
        op = self.stem_2(op)
        s0 = s1 = op
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == (2 * self._n_layers // 3):
                if self._auxiliary and training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out)
        return logits, logits_aux

    def _loss(self, x, target):
        logits = self(x, training=True)
        return self._criterion(target, logits)