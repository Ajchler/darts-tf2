from genotypes import Genotype
from genotypes import PRIMITIVES
import tensorflow as tf
import tensorflow.keras as keras

class Cell():
    def __init__(self):
        print("I am a cell!")

class Network():
    def __init__(self, C, input_shape, n_classes, n_layers, n_cells=4, multiplier=4, stem_multiplier=3):
        self._C = C
        self._input_shape = input_shape
        self._n_classes = n_classes
        self._n_layers = n_layers
        self._n_cells = n_cells
        self._multiplier = multiplier

        C_curr = C * stem_multiplier
        self.stem = keras.Sequential()
        self.stem.add(keras.layers.Conv2D(C_curr, kernel_size=(3,3), padding='same', strides=1, use_bias=False, input_shape=input_shape))
        self.stem.add(keras.layers.BatchNormalization())