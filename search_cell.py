import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from model_search import *
from config import Config
from architect import Architect
import data_utils
import datetime

LOG_DIR='./logs'

tf.get_logger().setLevel('INFO')

@tf.function
def validation_step(x_batch_valid, y_batch_valid):
    logits = model(x_batch_valid, training=False)
    loss = criterion(y_batch_valid, logits)
    validation_acc.update_state(y_batch_valid, logits)
    valid_loss.update_state(y_batch_valid, logits)
    return loss

@tf.function
def train_step(x_batch_train, y_batch_train):
    with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True) # maybe use training=True?
        loss = criterion(y_batch_train, logits)

    grads = tape.gradient(loss, model.trainable_weights)
    grads = [(tf.clip_by_norm(grad, clip_norm=config.args.grad_clip)) for grad in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_loss.update_state(y_batch_train, logits)
    train_acc.update_state(y_batch_train, logits)
    return loss

@tf.function
def architect_step(x_batch_train, y_batch_train, x_batch_valid, y_batch_valid):
    architect.step(x_batch_train, y_batch_train, x_batch_valid, y_batch_valid, xi=lr, net_optimizer=optimizer, unrolled=config.args.unrolled)

def current_lr(step, decay_steps, alpha, initial_lr):
    step = min(step + 1, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_lr * decayed

config = Config('search')
tf.random.set_seed(config.args.seed)

# dataset handling
(x_train, y_train), (x_test, y_test) = data_utils.load_cifar10()
x = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])
x_train = x[:len(x) // 2]
x_test = x[len(x) // 2:]
y_train = y[:len(y) // 2]
y_test = y[len(y) // 2:]
x_train = x_train / 255
y_train = y_train
x_test = x_test / 255
y_test = y_test

train_transform = tf.keras.Sequential([
    keras.layers.RandomCrop(32, 32),
    keras.layers.RandomFlip("horizontal"),
    keras.layers.Normalization(mean=[0.49139968, 0.48215827, 0.44653124], variance=[0.24703233, 0.24348505, 0.26158768])
])

valid_transform = tf.keras.Sequential([
    keras.layers.Normalization(mean=[0.49139968, 0.48215827, 0.44653124], variance=[0.24703233, 0.24348505, 0.26158768])
])

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=30000).batch(config.args.batch_size
                                                            ).map(lambda x, y: (train_transform(x, training=True), y))
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.shuffle(buffer_size=30000).batch(config.args.batch_size).map(lambda x, y: (valid_transform(x, training=True), y))

# calculate number of steps for learning rate decay
decay_steps = config.args.epochs * len(x_train) // config.args.batch_size

# Initialize learing rate scheduler, loss function and optimizer
lr_scheduler = keras.experimental.CosineDecay(config.args.learning_rate, decay_steps, config.args.learning_rate_min)
criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=config.args.momentum, clipnorm=config.args.grad_clip)

# Create a model and an architect
model = Network(config.args.init_channels, criterion, 10, config.args.layers, n_nodes=config.args.nodes, multiplier=config.args.multiplier)
architect = Architect(model, config.args, criterion)

tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)
tb_callback.set_model(model)

lr = tf.cast(config.args.learning_rate, tf.float32)
lr_step = 0

# Initialize metrics
train_loss = keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
train_acc = keras.metrics.SparseCategoricalAccuracy()
valid_loss = keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
validation_acc = keras.metrics.SparseCategoricalAccuracy()
best_acc = 0
best_genotype = model.genotypes()

# prepare log directories
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/search_arch/' + current_time + '/train'
test_log_dir = 'logs/search_arch/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


with open(f"{train_log_dir}/genotype_initial", 'w') as genotype_file:
    genotype_file.write(str(best_genotype))
with open(f"{train_log_dir}/config", 'w') as config_file:
    config_file.write(str(config.args))
print(f"Initial genotype: {best_genotype}")
print(f"Initial alphas: {model.arch_params()}")

for epoch in range(config.args.epochs):
    # training
    for step, ((x_batch_train, y_batch_train), (x_batch_valid, y_batch_valid)) in enumerate(zip(train_dataset, val_dataset)):
        # eta needs to be changed to learning rate scheduler
        lr = tf.cast(current_lr(lr_step, decay_steps, config.args.learning_rate_min, config.args.learning_rate), tf.float32)
        architect_step(x_batch_train, y_batch_train, x_batch_valid, y_batch_valid)
        loss = train_step(x_batch_train, y_batch_train)
        lr_step += 1

        if (step + 1) % 100 == 0:
            print(datetime.datetime.now())
            print(f'Epoch: {epoch + 1}')
            print(f'Step: {step + 1}')
            print(f'Number of samples seen: {(step + 1) * config.args.batch_size}')
            print(f"Loss is: {loss}\n")

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_acc.result(), step=epoch)

    trn_acc = train_acc.result()

    # validation
    for step, (x_batch_valid, y_batch_valid) in enumerate(val_dataset):
        loss = validation_step(x_batch_valid, y_batch_valid)
        if (step + 1) % 100 == 0:
            print(datetime.datetime.now())
            print(f'Epoch: {epoch + 1}')
            print(f'Validation step: {step + 1}')
            print(f"Validation loss is: {loss}\n")

    val_acc = validation_acc.result()

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', valid_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', validation_acc.result(), step=epoch)

    # end of epoch logging and updating/reseting metrics
    with open(f"{train_log_dir}/genotype_epoch_{epoch + 1}", 'w') as genotype_file:
        genotype_file.write(str(model.genotypes()))

    if (val_acc > best_acc or epoch == 0):
        best_acc = val_acc
        best_genotype = model.genotypes()

    print(f"End of epoch {epoch + 1}")
    print(f"Validation accuracy: {float(val_acc)}")
    print(f"Genotype: {model.genotypes()}")
    print(f"Alphas: {model.arch_params()}\n\n")

    train_loss.reset_states()
    valid_loss.reset_states()
    train_acc.reset_states()
    validation_acc.reset_states()

# end of architecture search
print(f"Best accuracy is: {best_acc}")
print(f"This was achieved with this genotype: {best_genotype}")
print(f"Alphas: {model.arch_params()}")
model.summary()
with open(f"{train_log_dir}/genotype_best", 'w') as genotype_file:
    genotype_file.write(str(best_genotype))