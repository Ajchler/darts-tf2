import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from model_search import *
from config import Config
from architect import Architect
import data_utils

def current_lr(step, decay_steps, alpha, initial_lr):
    step = min(step + 1, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_lr * decayed

def main():
    config = Config()
    tf.random.set_seed(config.args.seed)

    # dataset handling
    (x_train, y_train), (x_test, y_test) = data_utils.load_cifar10()
    x_train = x_train / 255
    y_train = y_train / 255
    x_test = x_test / 255
    y_test = y_test / 255
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=100).batch(config.args.batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))# validation dataset
    val_dataset = val_dataset.shuffle(buffer_size=100).batch(config.args.batch_size)

    decay_steps = config.args.epochs * len(x_train) // config.args.batch_size

    lr_scheduler = keras.experimental.CosineDecay(config.args.learning_rate, decay_steps, config.args.learning_rate_min)
    criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=config.args.momentum)

    model = Network(config.args.init_channels, criterion, 10, 8)
    architect = Architect(model, config.args, criterion)

    lr_step = 0

    for epoch in range(config.args.epochs):
        print(f"Start of epoch {epoch}")

        # TODO: make thsi a train function and add tf.function decorator
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            x_batch_valid, y_batch_valid = next(iter(val_dataset))
            # eta needs to be changed to learning rate scheduler
            lr = current_lr(lr_step, decay_steps, config.args.learning_rate_min, config.args.learning_rate)
            architect.step(x_batch_train, y_batch_train, x_batch_valid, y_batch_valid, xi=lr, net_optimizer=optimizer, unrolled=config.args.unrolled)

            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True) # maybe use training=True?
                loss = criterion(y_batch_train, logits)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            lr_step += 1

            if step % 10 == 0:
                print(f'step {step}')

if __name__ == "__main__":
    main()