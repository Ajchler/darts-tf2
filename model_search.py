from genotypes import Genotype
from genotypes import PRIMITIVES
import tensorflow as tf
import tensorflow.keras as keras

class Cell(tf.Module):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction
        #TODO
        

class Network(tf.Module):
    def __init__(self, C, input_shape, n_classes, n_layers, n_nodes=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._input_shape = input_shape
        self._n_classes = n_classes
        self._n_layers = n_layers
        self._n_nodes = n_nodes
        self._multiplier = multiplier

        C_curr = C * stem_multiplier
        self.stem = keras.Sequential()
        self.stem.add(keras.layers.Conv2D(C_curr, kernel_size=(3,3), padding='same', strides=1, use_bias=False, input_shape=input_shape))
        self.stem.add(keras.layers.BatchNormalization())
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        
        self.cells = []
        reduction_previous = False
        for i in range(n_layers):
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
                
            cell = Cell(reduction)
            reduction_previous = reduction
            self.cells.append(cell)
            C_curr_out = C_curr * n_nodes
            C_prev_prev, C_prev = C_prev, C_curr_out
 
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        # using Dense layer from tensorflow as a replacement of nn.Linear()
        self.classifier = tf.keras.layers.Dense(n_classes, activation=None)