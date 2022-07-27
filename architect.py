import tensorflow as tf
import tensorflow.keras as keras

class Architect():
    def __init__(self, model, args):
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.model = model
        self.optimizer = keras.optimizers.Adam(learning_rate=args.arch_learning_rate, beta_1=0.5, beta_2=0.999)

    def step(self, input_train, target_train, input_valid, target_valid, eta, net_optimizer, unrolled):
        with tf.GradientTape(persistent=True) as gt:
            if unrolled:
                loss = 'roll'
            else:
                loss = self._backward_step(input_valid, target_valid)
        grads_reduce = gt.gradient(loss, self.model.alphas_reduce)
        grads_normal = gt.gradient(loss, self.model.alphas_normal)
        self.optimizer.apply_gradients(zip([grads_reduce, grads_normal], [self.model.alphas_reduce, self.model.alphas_normal]))

    def _backward_step(self, input_valid, target_valid):
        return self.model._loss(input_valid, target_valid)