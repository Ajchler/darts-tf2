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

    train_acc = keras.metrics.SparseCategoricalAccuracy()
    validation_acc = keras.metrics.SparseCategoricalAccuracy()
    best_acc = 0
    best_genotype = model.genotypes()

    for epoch in range(config.args.epochs):
        print(f"Start of epoch {epoch}")

        # training
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

            train_acc.update_state(y_batch_train, logits)

            lr_step += 1

            if step % 10 == 0:
                print(f'Step: {step}')
                print(f'Number of samples seen: {(step + 1) * config.args.batch_size}')
                print(f"Loss is: {loss}\n")

        trn_acc = train_acc.result()
        train_acc.reset_states()
        print(f'Training accuracy over this epoch: {float(trn_acc)}')

        # validation
        for step, (x_batch_valid, y_batch_valid) in enumerate(val_dataset):
            logits = model(x_batch_valid, training=False)
            validation_acc.update_state(y_batch_valid, logits)

        val_acc = validation_acc.result()
        if (val_acc > best_acc or epoch == 0):
            best_acc = val_acc
            best_genotype = model.genotypes()
        validation_acc.reset_states()
        print(f"Validation accuracy: {float(val_acc)}")
        print(f"Genotype: {model.genotypes()}")
        print(f"Alphas: {model.arch_params()}")
        print(f"End of epoch {epoch}\n\n")

    print(f"Best accuracy is: {best_acc}")
    print(f"This was achieved with this genotype: {best_genotype}")
    print(f"Alphas: {model.arch_params()}")

if __name__ == "__main__":
    main()