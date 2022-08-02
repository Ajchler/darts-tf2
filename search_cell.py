import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from model_search import *
from config import Config
from architect import Architect
import data_utils

def main():
    config = Config()
    tf.random.set_seed(config.args.seed)

    # dataset handling
    (x_train, y_train), (x_test, y_test) = data_utils.load_cifar10()
    x_train = x_train / 255
    y_train = y_train / 255
    x_test = x_test / 255
    y_test = y_test / 255
    input_shape = x_train.shape[1:]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=100).batch(config.args.batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))# validation dataset
    val_dataset = val_dataset.shuffle(buffer_size=100).batch(config.args.batch_size)

    criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.SGD(learning_rate=config.args.learning_rate, momentum=config.args.momentum)

    model = Network(config.args.init_channels, criterion, 10, 8)
    architect = Architect(model, config.args)

    for epoch in range(config.args.epochs):
        print(f"Start of epoch {epoch}")

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            x_batch_valid, y_batch_valid = next(iter(val_dataset))
            # eta needs to be changed to learning rate scheduler
            architect.step(x_batch_train, y_batch_train, x_batch_valid, y_batch_valid, eta=config.args.learning_rate, net_optimizer=optimizer, unrolled=config.args.unrolled)

            with tf.GradientTape() as tape:
                logits = model(x_batch_train) # maybe use training=True?
                loss = criterion(y_batch_train, logits)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))


            if step % 10 == 0:
                print(f'step {step}')

if __name__ == "__main__":
    main()