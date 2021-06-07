import numpy as np
import scipy.linalg as linalg

from keras import datasets, optimizers
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import Callback
from sklearn.utils import check_random_state
import tensorflow as tf


def my_init(shape, dtype=None, **kwargs):
    fan_in, _ = shape
    return tf.random.normal(shape, stddev=np.sqrt(2. / fan_in),
                            dtype=dtype, **kwargs)


def build_model(input_dim, hidden_dim, loss="mean_squared_error",
activation_function="relu",
                rf_regime=False, kernel_initializer=my_init,
                learning_rate=1):
    net = Sequential()
    net.add(Dense(hidden_dim, input_dim=input_dim,
                  activation=activation_function, use_bias=False,
                  kernel_initializer=kernel_initializer,
                  trainable=not rf_regime))
    net.add(Dense(1, use_bias=False))
    optimizer = optimizers.SGD(learning_rate=learning_rate)
    net.compile(loss=loss, optimizer=optimizer)
    return net


class Stats(object):
    def __init__(self, verbose=0):
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _compute_gradients(self, Ws, intermediate_activations):
        z = intermediate_activations
        self._log("Computing gradients w.r.t inputs...")
        grads = [Ws[-1] for _ in z[0]]
        for layer_zz, W in zip(z[::-1][1:], Ws[::-1][1:]):
            for i, (zz, grad) in enumerate(zip(layer_zz, grads)):
                W_ = W.copy()
                mask = zz <= 0
                grads[i] = grad[:, mask]@W_[mask]
        return grads

    def _compute_forward_and_backward_stats(self, model, X):
        # get weights matrices and per-sample intermediate activations
        z = []
        Ws = []
        outputs = []
        self._log("Computing weights and gradients")
        for layer in model.layers:
            if isinstance(layer, Dense):
                W = layer.get_weights()[0].T
                Ws.append(W)
                outputs.append(layer.output)
        func = Model(inputs=model.input, outputs=outputs)
        z = [zz.numpy() for zz in func(X)]

        # compute gradients of outputs w.r.t inputs
        grads = self._compute_gradients(Ws, z)

        # get extract
        return z, grads, Ws

    def _spectral(self, W, hidden):
        WW = W@W.T
        input_dim = W.shape[1]
        C = np.cov(hidden.T)
        C *= input_dim
        W_op_norm = linalg.norm(W, ord=2)
        W_F_norm = linalg.norm(W, ord="fro")
        return linalg.svdvals(C).min(), W_op_norm ** 2, W_F_norm ** 2

    def get_stats(self, model, X):
        """Computes varous quantifies mentioned in our bounds.
        """
        z, grads, Ws = self._compute_forward_and_backward_stats(model, X)
        hidden, _ = z
        W, _ = Ws
        stats = {}
        self._log("Computing singular-values of stacked grad matrices, etc.")
        # stats["grad_norm"] = [linalg.norm(grad, ord=2) for grad in grads]
        lambda_min, lambda_max, lambda_sum = self._spectral(Ws[0], hidden)
        stats["lambda_min_C"] = lambda_min
        stats["lambda_max_WWT"] = lambda_max
        stats["lambda_sum_WWT"] = lambda_sum
        stats["inv_cond"] = np.sqrt(lambda_min / lambda_max)
        # stats["grads"] = list(map(np.ravel, grads))
        return stats


@tf.function
def tf_get_grad(model, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x[None])
    return tape.gradient(y, x)
