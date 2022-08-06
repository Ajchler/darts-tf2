from model_search import Network
import tensorflow as tf
import tensorflow.keras as keras

class Architect():
    def __init__(self, model, args, criterion):
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.model = model
        self.v_model = Network(args.init_channels, criterion, 10, 8)
        self.v_model.set_weights(self.model.get_weights())
        self.optimizer = keras.optimizers.Adam(learning_rate=args.arch_learning_rate, beta_1=0.5, beta_2=0.999)

    def step(self, x_train, y_train, x_valid, y_valid, xi, net_optimizer, unrolled):
        if unrolled:
            self._backward_step_unrolled(x_train, y_train, x_valid, y_valid, xi, net_optimizer)
        else:
            with tf.GradientTape(persistent=True) as gt:
                loss = self._backward_step(x_valid, y_valid)
            grads_reduce = gt.gradient(loss, self.model.alphas_reduce)
            grads_normal = gt.gradient(loss, self.model.alphas_normal)
            self.optimizer.apply_gradients(zip([grads_reduce, grads_normal], [self.model.alphas_reduce, self.model.alphas_normal]))

    def _backward_step_unrolled(self, x_train, y_train, x_valid, y_valid, xi, net_optimizer):
        self._virtual_step(x_train, y_train, xi, net_optimizer)

        with tf.GradientTape() as gt:
            loss = self.v_model._loss(x_valid, y_valid)

        variables = self.v_model.trainable_weights
        variables.append(self.v_model.alphas_normal)
        variables.append(self.v_model.alphas_reduce)
        v_grads = gt.gradient(loss, variables)
        dalpha = v_grads[-2:]
        dw = v_grads[:-2]

        hess = self.calc_hessian(dw, x_train, y_train)

    def calc_hessian(self, dw, x_train, y_train):
        norm = tf.concat([tf.reshape(x, [-1]) for x in dw], 0)
        norm = tf.norm(norm)

    def _virtual_step(self, x_train, y_train, xi, net_optimizer):
        with tf.GradientTape() as gt:
            loss = self.model._loss(x_train, y_train)
        grads = gt.gradient(loss, self.model.trainable_weights)

        # TODO: momentum calculation
        m = 0
        for idx, (w, vw, g) in enumerate(zip(self.model.weights, self.v_model.weights, grads)):
            self.v_model.weights[idx] = w - xi * (m + g + self.weight_decay * w)

        for idx, (a, va) in enumerate(zip(self.model._arch_params, self.v_model._arch_params)):
            self.v_model._arch_params[idx] = a

    def _backward_step(self, input_valid, target_valid):
        return self.model._loss(input_valid, target_valid)