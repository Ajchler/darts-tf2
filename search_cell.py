from distutils.command import config
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from config import Config
import data_utils

def main():
    config = Config() 
    (x_train, y_train), (x_test, y_test) = data_utils.load_cifar10()
    print(x_train.shape)
    tf.random.set_seed(config.args.seed)

if __name__ == "__main__":
    main()