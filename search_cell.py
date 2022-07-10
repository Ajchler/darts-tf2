import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from model_search import Network
from config import Config
import data_utils

def main():
    config = Config() 
    tf.random.set_seed(config.args.seed)
 
    # Load data
    (x_train, y_train), (x_test, y_test) = data_utils.load_cifar10()
    input_shape = x_train.shape[1:]

    # Questionable if from_logits should be used TODO!
    criterion = keras.losses.CategoricalCrossentropy(from_logits=True)    

    # SUBMODULES ARE STORED in model.submodules
    model = Network(config.args.init_channels, criterion, input_shape, n_classes=10, n_layers=8)

if __name__ == "__main__":
    main()