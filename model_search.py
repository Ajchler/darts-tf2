from genotypes import Genotype
from genotypes import PRIMITIVES
from operations import *
import tensorflow as tf
import tensorflow.keras as keras

class MixedOp(tf.Module):
    def __init__(self, C_curr, stride):
        super().__init__()
        self._ops = []
        for prim in PRIMITIVES:
            self._ops.append(prim)

class Cell(tf.Module):
    def __init__(self, n_nodes, multiplier, C_curr, C_prev, C_prev_prev, reduction, reduction_prev, input_shape):
        super().__init__()
        self.reduction = reduction
        self._n_nodes = n_nodes
        self._multiplier = multiplier
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_curr, input_shape)
        else:
            self.preprocess0 = ReLUConvBN(input_shape, C_curr, 1, 1, 'valid')
        self.preprocess1 = ReLUConvBN(input_shape, C_curr, 1, 1, 'valid')

        self._ops = []
        self._bns = []
        for i in range(self._n_nodes):
            for j in range (i + 2):
                stride = 2 if reduction and j < 2 else 1
                self._ops.append(MixedOp(C_curr, stride))

        #TODO: def forward(...):

class Network(tf.Module):
    def __init__(self, C, criterion, input_shape, n_classes, n_layers, n_nodes=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._input_shape = input_shape
        self._n_classes = n_classes
        self._n_layers = n_layers
        self._n_nodes = n_nodes
        self._multiplier = multiplier
        self._criterion = criterion

        C_curr = C * stem_multiplier
        self.stem = keras.Sequential()
        self.stem.add(keras.layers.Conv2D(C_curr, kernel_size=(3,3), padding='same', strides=1, use_bias=False, input_shape=input_shape))
        self.stem.add(keras.layers.BatchNormalization())

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        
        self.layers = []
        reduction_prev = False
        for i in range(n_layers):
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
                
            cell = Cell(n_nodes, multiplier, C_curr, C_prev, C_prev_prev, reduction, reduction_prev, input_shape)
            reduction_prev = reduction
            self.layers.append(cell)
            C_curr_out = C_curr * n_nodes
            C_prev_prev, C_prev = C_prev, C_curr_out
 
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        # using Dense layer from tensorflow as a replacement of nn.Linear()
        self.classifier = tf.keras.layers.Dense(n_classes, activation=None)
 
        #TODO: initialize alphas