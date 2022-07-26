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
    x_train = x_train / 255
    y_train = y_train / 255
    input_shape = x_train.shape[1:]
    criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = Network(config.args.init_channels, criterion, 10, 8)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=100).batch(config.args.batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))# validation dataset
    val_dataset = val_dataset.shuffle(buffer_size=100).batch(config.args.batch_size)

    #model = keras.Model(inputs=inputs, outputs=output, name='darts4coatnet')
    ## TODO: weight decay is missing, haven't found a way to add global weight decay yet
    #model.compile(optimizer=keras.optimizers.SGD(learning_rate=config.args.learning_rate,
    #                                             momentum=config.args.momentum),
    #              loss=keras.losses.CategoricalCrossentropy(from_logits=True))

#    optimizer = keras.optimizers.SGD(learning_rate=config.args.learning_rate, momentum=config.args.momentum)
#    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
#    train_acc = keras.metrics.SparseCategoricalAccuracy()
#
    for epoch in range(config.args.epochs):
        print(f"Start of epoch {epoch}")

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                out = model(x_batch_train)


if __name__ == "__main__":
    main()