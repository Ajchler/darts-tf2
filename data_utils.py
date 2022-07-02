import tensorflow as tf
import tensorflow.keras as keras

def load_cifar10():
    return keras.datasets.cifar10.load_data()