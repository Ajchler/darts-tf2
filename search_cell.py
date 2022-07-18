import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from model_search import *
from config import Config
import data_utils

def main():
    config = Config()
    tf.random.set_seed(config.args.seed)

    # Load data
    #TODO: split the data into batches?
    (x_train, y_train), (x_test, y_test) = data_utils.load_cifar10()
    input_shape = x_train.shape[1:]
    inputs = keras.Input(input_shape)
    output = Model(inputs, True, config.args.init_channels, 10, 8)

if __name__ == "__main__":
    main()