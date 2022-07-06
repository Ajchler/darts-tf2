from distutils.command import config
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from model_search import Network
from config import Config
import data_utils

def main():
    config = Config() 
    (x_train, y_train), (x_test, y_test) = data_utils.load_cifar10()
    input_shape = x_train.shape[1:]
    tf.random.set_seed(config.args.seed)
    
    model = Network(config.args.init_channels, input_shape, n_classes=10, n_layers=8)
    for cell in model.cells:
        print(cell.name)

if __name__ == "__main__":
    main()