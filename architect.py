import tensorflow as tf
import tensorflow.keras as keras

class Architect():
    def __init__(self, model, args):
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.model = model
        self.optimizer = keras.optimizers.Adam(learning_rate=args.arch_learning_rate, beta_1=0.5, beta_2=0.999)

    def step(self, x_train, y_train, x_valid, y_valid, eta, net_optimizer, unrolled):
        if unrolled:
            self._backward_step_unrolled(x_train, y_train, x_valid, y_valid, eta, net_optimizer)
        else:
            with tf.GradientTape(persistent=True) as gt:
                loss = self._backward_step(x_valid, y_valid)
            grads_reduce = gt.gradient(loss, self.model.alphas_reduce)
            grads_normal = gt.gradient(loss, self.model.alphas_normal)
        #self.optimizer.apply_gradients(zip([grads_reduce, grads_normal], [self.model.alphas_reduce, self.model.alphas_normal]))

    def _compute_unrolled_model(self, x_train, y_train, eta, net_optimizer):
        with tf.GradientTape() as gt:
            loss = self.model._loss(x_train, y_train)
        theta = tf.concat([tf.reshape(x, -1) for x in self.model.trainable_weights], -1)
        # TODO: moment calculation?
        moment = tf.zeros_like(theta)
        grads = gt.gradient(loss, self.model.trainable_weights)
        dtheta = tf.concat([tf.reshape(g, [-1]) for g in grads], -1)
        #TODO:#unrolled_model = self._construct_model_from_theta(tf.subtract(dtheta,))

    def _backward_step_unrolled(self, x_train, y_train, x_valid, y_valid, eta, net_optimizer):
        unrolled_model = self._compute_unrolled_model(x_train, y_train, eta, net_optimizer)

    def _backward_step(self, input_valid, target_valid):
        return self.model._loss(input_valid, target_valid)