import numpy as np
from scipy import special

from sklearn.utils import check_random_state

import models, keras_models


class _NTK(models.GLM):
    def __init__(self, first_layer_variance=None,
                 second_layer_variance=None,
                 **kwargs):
        super(_NTK, self).__init__(**kwargs)
        self.first_layer_variance = first_layer_variance
        self.second_layer_variance = second_layer_variance
        self._check_init()

    def _check_init(self):
        pass

    def init_from(self, W, coef):
        self.W_ = W
        self.coef_ = coef
        return self

    @property
    def n_total_features(self):
        raise NotImplementedError

    def sigma(self, z):
        raise NotImplementedError

    def _prefit(self, X, y):
        rng = check_random_state(self.random_state)
        W = rng.randn(self.hidden_dim, self.input_dim)
        W /= np.sqrt(self.input_dim)
        self.W_ = W
        return self


class NTK(_NTK):
    def _check_init(self):
        if max(self.first_layer_variance, self.second_layer_variance) <= 0:
            raise ValueError

    def _rf_transform(self, X):
        XW = X@self.W_.T
        Z = self.sigma(XW)
        Z *= np.sqrt(self.second_layer_variance / self.hidden_dim)
        return Z

    def _rf_grad(self, X):
        XW = X@self.W_.T
        rf_coefs = self._get_rf_coefs()
        grads = rf_coefs * self.sigma_grad(XW)@self.W_
        grads *= np.sqrt(self.second_layer_variance / self.hidden_dim)
        return grads

    @property
    def n_total_features(self):
        raise NotImplementedError

    def sigma(self, z):
        raise NotImplementedError

    def sigma_grad(self, z):
        raise NotImplementedError

    def sigma_hess(self, z):
        raise NotImplementedError

    def transform(self, X):
        all_Z = []
        XW = X@self.W_.T
        if self.first_layer_variance:
            F_prime = self.sigma_grad(XW)
            Z = F_prime[:, :, None] * X[:, None]
            Z = Z.reshape((len(X), -1))
            Z *= np.sqrt(self.first_layer_variance / self.hidden_dim)
            all_Z.append(Z)
        if self.second_layer_variance:
            Z = self._rf_transform(X)
            all_Z.append(Z)
        return np.hstack(all_Z)

    def _get_ntk_coefs(self):
        assert self.first_layer_variance > 0
        kd = self.hidden_dim * self.input_dim
        return self.coef_[:kd]

    def _get_rf_coefs(self):
        assert self.second_layer_variance > 0
        if self.first_layer_variance == 0:
            return self.coef_
        else:
            return self.coef_[-self.hidden_dim:]

    def grad(self, X):
        net_grads = np.zeros_like(X)
        if self.first_layer_variance:
            XW = X@self.W_.T
            p = self.hidden_dim * self.input_dim
            ntk_coefs = self._get_ntk_coefs()
            a = ntk_coefs.reshape((self.hidden_dim, self.input_dim))
            G = self.sigma_grad(XW)
            H = self.sigma_hess(XW)
            grads = G@a
            grads += (H@a) * X
            grads *= np.sqrt(self.first_layer_variance / self.hidden_dim)
            net_grads += grads
        if self.second_layer_variance:
            net_grads += self._rf_grad(X)
        return net_grads


class PureRF(NTK):
    def __init__(self, first_layer_variance=1, **kwargs):
        super(PureRF, self).__init__(first_layer_variance=first_layer_variance,
        second_layer_variance=0, **kwargs)


class PureNTK(NTK):
    def __init__(self, second_layer_variance=1, **kwargs):
        super(PureNTK, self).__init__(first_layer_variance=0,
        second_layer_variance=1, **kwargs)


class ReLUNTK(NTK):
    def sigma(self, z):
        return np.maximum(z, 0)

    def sigma_grad(self, z):
        g = np.zeros_like(z)
        g[z > 0] = 1
        return g

    def sigma_hess(self, z):
        return np.zeros_like(z)


class AbsoluteValueNTK(NTK):
    def sigma(self, z):
        return np.abs(z)

    def sigma_grad(self, z):
        return np.sign(z)

    def sigma_hess(self, z):
        return np.zeros_like(z)


class QuadraticNTK(NTK):
    def sigma(self, z):
        return .5 * z ** 2

    def sigma_grad(self, z):
        return z

    def sigma_hess(self, z):
        return np.ones_like(z)


class TanhNTK(NTK):
    def sigma(self, z):
        return np.tanh(z)

    def sigma_grad(self, z):
        return 1 - np.tanh(z) ** 2

    def sigma_hess(self, z):
        t = np.tanh(z)
        return 2 * t * (t ** 2 - 1)


class ErfNTK(NTK):
    def sigma(self, z):
        return special.erf(z)

    def sigma_grad(self, z):
        return 2 * np.exp(-z ** 2) / np.sqrt(np.pi)

    def sigma_hess(self, z):
        return -4 * z * np.exp(-z ** 2) / np.sqrt(np.pi)


import keras_models

class FullyTrainableModel(models.GLM):
    def _prefit(self, X, y):
        pass
        return self

    def fit(self, X, y):
        self.model = keras_models.build_model(input_dim=self.input_dim,
        hidden_dim=self.hidden_dim,
        learning_rate=self.kwargs.get("learning_rate", 1),
        activation_function=self.kwargs["activation_function"])
        self.model.fit(X, y, epochs=self.kwargs.get("n_epochs", 50))

        for layer in self.model.layers:
            if hasattr(layer, "get_weights"):
                coef = layer.get_weights()[0]
        self.coef_ = coef.ravel()
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_stats(self, X):
        statistician = keras_models.Stats()
        return statistician.get_stats(self.model, X)

    def grad(self, X):
        return [keras_models.tf_get_grad(self.model, x).numpy().ravel()
        for x in X]
