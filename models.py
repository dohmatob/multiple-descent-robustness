import itertools

import numpy as np
from scipy import linalg

from joblib import delayed, Parallel
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.utils import check_random_state

import kernels


class GLM:
    def __init__(self, input_dim, hidden_dim, ridge=0, solver="sklearn",
                 random_state=None, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ridge = ridge
        self.solver = solver
        self.random_state = random_state
        self.kwargs = kwargs

    def transform(self, X):
        raise NotImplementedError

    def _prefit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        return self

    def fit(self, X, y):
        self._prefit(X, y)
        n_samples = len(X)
        Z = self.transform(X)
        if self.ridge:
            model = Ridge(fit_intercept=False,
                          alpha=self.ridge * n_samples)
        else:
            model = LinearRegression(fit_intercept=False)
        if self.solver == "linalg":
            assert self.ridge == 0
            model.coef_ = linalg.lstsq(Z, y)[0]
            model.intercept_ = 0
        else:
            model.fit(Z, y)
        self.coef_ = model.coef_
        return self

    def predict(self, X, Z=None):
        if Z is None:
            Z = self.transform(X)
        return Z@self.coef_

    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)

    def grad(self, X):
        raise NotImplementedError

    @property
    def n_total_features(self):
        raise NotImplementedError


class InfiniteWidthNTKModel(GLM):
    def __init__(self, activation_function=None, kind=None, n_samples=None,
                 first_layer_variance=None, second_layer_variance=None,
                 **kwargs):
        super(InfiniteWidthNTKModel, self).__init__(**kwargs)
        self.activation_function = activation_function
        self.kind = kind
        self.n_samples = n_samples
        self.first_layer_variance = first_layer_variance
        self.second_layer_variance = second_layer_variance
        self._build_kernels(**self.kwargs)

    def _build_kernels(self, **kwargs):
        if self.activation_function == "relu":
            self.k0 = kernels.ZerothOrderArccosKernel(input_dim=self.input_dim)
            self.k1 = kernels.FirstOrderArccosKernel(input_dim=self.input_dim)
        elif self.activation_function == "abs":
            self.k0 = kernels.ZerothOrderArcsinKernel(input_dim=self.input_dim)
            self.k1 = kernels.FirstOrderArcsinKernel(input_dim=self.input_dim)
        elif self.activation_function == "erf":
            self.k0 = kernels.ZerothOrderErfKernel(input_dim=self.input_dim)
            self.k1 = kernels.FirstOrderErfKernel(input_dim=self.input_dim)
        else:
            if self.kind is None:
                raise NotImplementedError(self.activation_function)

    def transform(self, X):
        dot = X@self.X_train_.T
        out = np.zeros((len(X), len(self.X_train_)))
        for i, x in enumerate(X):
            dot = self.X_train_@x
            if hasattr(self, "k"):
                out[i] += self.k.kappa(dot)
            else:
                if self.first_layer_variance:
                    tmp = dot * self.k0.kappa(dot)
                    tmp *= np.sqrt(self.first_layer_variance)
                    out[i] += tmp
                if self.second_layer_variance:
                    tmp = self.k1.kappa(dot)
                    tmp *= np.sqrt(self.second_layer_variance)
                    out[i] += tmp
        return out

    def grad(self, X):
        dot = X@self.X_train_.T
        grads = np.zeros((len(X), self.input_dim))
        grads[:] = np.inf
        n_samples = len(self.X_train_)
        for i, Xx in enumerate(dot):
            if hasattr(self, "k"):
                aux = self.k.kappa_grad(Xx)
            else:
                aux = np.zeros(n_samples)
                if self.first_layer_variance:
                    tmp = self.k0.kappa(Xx) + Xx * self.k0.kappa_grad(Xx)
                    tmp *= np.sqrt(self.first_layer_variance)
                    aux += tmp
                if self.second_layer_variance:
                    tmp = self.k1.kappa_grad(Xx)
                    tmp *= np.sqrt(self.second_layer_variance)
                    aux += tmp
            grad_z = aux[:, None] * self.X_train_
            grads[i] = self.coef_@grad_z
        return np.asanyarray(grads)


class ClassicalKernelModel(InfiniteWidthNTKModel):
    def _build_kernels(self, **kwargs):
        if self.kind == "laplace":
            self.k = kernels.LaplaceKernel(input_dim=self.input_dim, **kwargs)
        elif self.kind == "gaussian":
            self.k = kernels.GaussianKernel(input_dim=self.input_dim, **kwargs)
        elif self.kind == "rough":
            self.k = kernels.RoughKernel(input_dim=self.input_dim, **kwargs)
        elif self.kind == "polynomial":
            self.k = kernels.PolynomialKernel(input_dim=self.input_dim, **kwargs)
        else:
            raise NotImplementedError(self.kind)


class FiniteWidthReLUNTKModel(GLM):
    def __init__(self, first_layer_variance=0, second_layer_variance=1,
                 **kwargs):
        super(FiniteWidthReLUNTKModel, self).__init__(**kwargs)
        self.first_layer_variance = first_layer_variance
        self.second_layer_variance = second_layer_variance
        self._check_init()

    @property
    def n_total_features(self):
        return self.hidden_dim * self.input_dim

    def _check_init(self):
        if self.first_layer_variance == 0 and self.second_layer_variance == 0:
            raise ValueError

    def _prefit(self, X, y):
        rng = check_random_state(self.random_state)
        W = rng.randn(self.hidden_dim, self.input_dim)
        W /= np.sqrt(self.input_dim)
        self.W_ = W
        return self

    def sigma(self, z):
        """Apply activation function"""
        return np.maximum(z, 0)

    def sigma_grad(self, z):
        return z > 0

    def _get_second_layer_features(self, x):
        if self.second_layer_variance:
            z = self.sigma(self.W_@x)
            z *= np.sqrt(self.second_layer_variance)
            return z

    def _get_first_layer_features(self, x):
        if self.first_layer_variance:
            assert len(x) == self.input_dim
            z = np.zeros((self.hidden_dim, self.input_dim))
            mask = self.W_@x > 0
            z[mask] = x
            z *= np.sqrt(self.first_layer_variance / self.hidden_dim)
            return z.ravel()

    def transform(self, X):
        features = []
        for x in X:
            z = []
            for layer_z in [self._get_first_layer_features(x),
                            self._get_second_layer_features(x)]:
                if layer_z is not None:
                    z.append(layer_z)
            assert len(z)
            features.append(np.concatenate(z))
        return np.asanyarray(features)

    def grad(self, X):
        grads = np.zeros((len(X), self.input_dim))
        XW = X@self.W_.T
        p = self.hidden_dim * self.input_dim
        for i in range(len(X)):
            mask = XW[i] > 0
            if self.first_layer_variance > 0:
                v = self.coef_[:p].reshape((self.hidden_dim, self.input_dim))
                grads[i] += v[mask].sum(0) * np.sqrt(
                    self.first_layer_variance / self.hidden_dim)
            if self.second_layer_variance > 0:
                if self.first_layer_variance > 0:
                    v = self.coef_[p:]
                else:
                    v = self.coef_
                grads[i] += np.sqrt(
                    self.second_layer_variance) * v[mask]@self.W_[mask]
        return grads


class FiniteWidthRandomFeaturesReLUModel(FiniteWidthReLUNTKModel):
    def __init__(self, second_layer_variance=1, **kwargs):
        super(FiniteWidthRandomFeaturesReLUModel, self).__init__(
            second_layer_variance=second_layer_variance,
            first_layer_variance=0, **kwargs)


class FiniteWidthPureReLUNTKModel(FiniteWidthReLUNTKModel):
    def __init__(self, first_layer_variance=1, **kwargs):
        super(FiniteWidthPureReLUNTKModel, self).__init__(
            first_layer_variance=first_layer_variance,
            second_layer_variance=0, **kwargs)
