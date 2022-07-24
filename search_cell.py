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
    inputs = keras.Input(input_shape) # to be replaced with first value of dataset
    logits = Model(inputs, config.args.init_channels, 10, 8)
    model = keras.Model(inputs=inputs, outputs=logits)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=100).batch(config.args.batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))# validation dataset
    val_dataset = val_dataset.shuffle(buffer_size=100).batch(config.args.batch_size)

    optimizer = keras.optimizers.SGD(learning_rate=config.args.learning_rate, momentum=config.args.momentum)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_acc = keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(config.args.epochs):
        print(f"Start of epoch {epoch}")

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            print(f'Step n.{step}')
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 10 == 0:
                print(f'Training loss at step {step} is: {loss}')


if __name__ == "__main__":
    main()