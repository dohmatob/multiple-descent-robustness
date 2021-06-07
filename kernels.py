import numpy as np
from scipy import special


class DotProductKernel(object):
    """
    Abstract implementation of dot-product kernel
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim

    # def __call__(self, X, X_=None):
    #    if X_ is None:
    #        X_ = X
    #    return self.kappa(X_@X.T)

    def __call__(self, rho):
        return self.kappa(rho)

    def kappa(self, rho):
        raise NotImplementedError

    @property
    def maclaurin_coefs(self):
        """
        Returns the first 3 (or more) coefficients in the Maclaurin
        """
        raise NotImplementedError

    @property
    def curvature_coefs(self):
        a0, a1 = self.maclaurin_coefs[:2]
        return a0, a1, self.kappa(1) - a0 - a1


class IdentityKernel(DotProductKernel):
    def kappa(self, rho):
        return rho

    @property
    def maclaurin_coefs(self):
        return np.array([0, 1, 0, 0])

    def kappa_grad(self, rho):
        return 1.


class ExponentialKernel(DotProductKernel):
    def __init__(self, input_dim, c=1., beta=1.4):
        super(ExponentialKernel, self).__init__(input_dim=input_dim)
        self.c = c
        self.beta = beta

    def kappa(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return np.exp(-self.c * (2 - 2 * rho) ** (self.beta / 2))

    def kappa_grad(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        a = self.c * (2 - 2 * rho) ** (self.beta / 2 - 1)
        b = self.c * (2 - 2 * rho) ** (self.beta / 2)
        out = self.beta * a * np.exp(-b)
        return out

    @property
    def maclaurin_coefs(self):
        assert self.c == 1
        beta = self.beta
        a1 = 2 ** (beta / 2 - 1) * beta * np.exp(-2 ** (beta / 2))
        a2 = 2 ** (beta / 2 - 3) * beta * np.exp(-2 ** (beta / 2)) * (2 + beta * (2 ** (beta / 2) - 1))
        return [None, a1, a2, None]


class GaussianKernel(ExponentialKernel):
    def __init__(self, input_dim, c=1.):
        super(GaussianKernel, self).__init__(input_dim=input_dim, c=c, beta=2)


class LaplaceKernel(ExponentialKernel):
    def __init__(self, input_dim, c=1):
        super(LaplaceKernel, self).__init__(input_dim=input_dim, c=1, beta=1)


class PolynomialKernel(DotProductKernel):
    def __init__(self, input_dim, p=1, c=1):
        super(PolynomialKernel, self).__init__(input_dim=input_dim)
        self.p = p
        self.c = c

    def kappa(self, rho):
        return (self.c + rho) ** self.p

    def kappa_grad(self, rho):
        return self.p * (self.c + rho) ** (self.p - 1)

    @property
    def maclaurin_coefs(self):
        a0 = self.c ** self.p
        a1 = self.c ** (self.p - 1) * self.p
        a2 = self.c ** (self.p - 2) * self.p * (self.p - 1) / 2
        a3 = self.c ** (self.p - 3) * self.p * (self.p - 1) * (self.p - 2) / 6
        return [a0, a1, a2, a3]


class QuadraticKernel(PolynomialKernel):
    def __init__(self, input_dim, c=1):
        super(QuadraticKernel, self).__init__(input_dim=input_dim, c=c, p=2)


class ZerothOrderArccosKernel(DotProductKernel):
    def kappa(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return np.arccos(-rho) / (2 * np.pi)

    def kappa_grad(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return 1 / (2 * np.pi * np.sqrt(1 - rho ** 2))

    @property
    def maclaurin_coefs(self):
        return np.array([1 / 4, 1 / (2 * np.pi), 0])


class ZerothOrderArcsinKernel(DotProductKernel):
    def kappa(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return 2 * np.arcsin(rho) / np.pi

    def kappa_grad(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return 2 / (np.pi * np.sqrt(1 - rho ** 2))

    @property
    def maclaurin_coefs(self):
        return np.array([0, 0, 2 / np.pi])


class FirstOrderArccosKernel(DotProductKernel):
    def kappa(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return (rho * np.arccos(-rho) + np.sqrt(1 - rho ** 2)) / (2 * np.pi)

    def kappa_grad(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return np.arccos(-rho) / (2 * np.pi)

    @property
    def maclaurin_coefs(self):
        return np.array([1. / (2 * np.pi), 1 / 4, 1 / (4 * np.pi)])


class FirstOrderArcsinKernel(DotProductKernel):
    def kappa(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return 2 * (rho * np.arcsin(rho) + np.sqrt(1 - rho ** 2)) / np.pi

    def kappa_grad(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return 2 * np.arcsin(rho) / np.pi

    @property
    def maclaurin_coefs(self):
        return np.array([2 / np.pi, 0, 1 / np.pi])


class FirstOrderErfKernel(DotProductKernel):
    def kappa(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return (2 / np.pi) * np.arcsin(2 * rho / 3)

    def kappa_grad(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return 4 / (np.pi * np.sqrt(9 - 4 * rho ** 2))


class ZerothOrderErfKernel(DotProductKernel):
    def kappa(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return 4 / (np.pi * np.sqrt(9 - 4 * rho ** 2))

    def kappa_grad(self, rho):
        rho = np.minimum(np.maximum(rho, -1), 1)
        return 16 / (np.pi * np.sqrt(9 - 4 * rho ** 2) ** 3)


class FullNTKKernel(DotProductKernel):
    def __init__(self, input_dim, activation_function="relu"):
        super(FullNTKKernel, self).__init__(input_dim=input_dim)
        self.activation_function = activation_function
        self._build_kernels()

    def _build_kernels(self):
        if self.activation_function == "relu":
            self.k0 = ZerothOrderArccosKernel(input_dim=self.input_dim)
            self.k1 = FirstOrderArccosKernel(input_dim=self.input_dim)
        elif self.activation_function == "abs":
            self.k0 = ZerothOrderArcsinKernel(input_dim=self.input_dim)
            self.k1 = FirstOrderArcsinKernel(input_dim=self.input_dim)
        else:
            raise NotImplementedError(self.activation_function)

    def kappa(self, rho):
        return rho * self.k0.kappa(rho) + self.k1.kappa(rho) / self.input_dim

    def kappa_grad(self, rho):
        grad = self.kappa(rho) + rho * self.k0.kappa_grad(rho)
        grad += self.k1.kappa_grad(rho) / self.input_dim
        return grad

    @property
    def maclaurin_coefs(self):
        coefs = np.append(0, self.k0.maclaurin_coefs[:-1])
        coefs += self.k1.maclaurin_coefs / self.input_dim
        return coefs


class SphericalGradientKernel(FullNTKKernel):
    def kappa(self, rho):
        return rho * self.k0.kappa(rho) - self.k1.kappa(rho) / self.input_dim

    def kappa_grad(self, rho):
        grad = self.k0.kappa(rho) + rho * self.k0.kappa_grad(rho)
        grad -= self.k1.kappa_grad(rho) / self.input_dim
        return grad

    @property
    def maclaurin_coefs(self):
        coefs = np.append(0, self.k0.maclaurin_coefs[:-1])
        coefs -= self.k1.maclaurin_coefs / self.input_dim
        return coefs


class PoincareKernel(object):
    def __init__(self, input_dim, activation_function="tanh"):
        self.input_dim = input_dim
        self.activation_function = activation_function
        self._build()

    def _build(self):
        if self.activation_function == "tanh":
            self.sigma = np.tanh
            self.maclaurin_coefs = [0, 1, 0, -1 / 3]
        elif self.activation_function == "sin":
            self.sigma = np.sin
            self.maclaurin_coefs = [0, 1, 0, - 1 / 6]
        elif self.activation_function == "erf":
            self.sigma = special.erf
            self.maclaurin_coefs = [0, 2 / np.sqrt(np.pi), 0,
            -2 / (3 * np.sqrt(np.pi))]
        else:
            raise NotImplementedError(self.activation_function)

    def __call__(self, XW):
        Z = self.sigma(XW)
        C = np.cov(Z.T)
        C *= self.input_dim
        return C


class TwoLayerReLUNTKKernel(DotProductKernel):
    def __init__(self, p, scale=0.5, first_layer_variance=1,
                 second_layer_variance=1):
        super(SphericalGradientKernel, self).__init__()
        self.p = p
        self.scale = scale
        self.first_layer_variance = first_layer_variance
        self.second_layer_variance = second_layer_variance
        self.k0 = ZerothOrderArccosKernel()
        self.k1 = FirstOrderArccosKernel()

    def kappa(self, rho):
        out = np.zeros_like(rho)
        if self.second_layer_variance:
            out = self.second_layer_variance * self.k0.kappa(rho)
        if self.first_layer_variance:
            out += self.first_layer_variance * self.k1.kappa(rho) / self.p
        return self.scale * out

    @property
    def maclaurin_coefs(self):
        if self.second_layer_variance:
            coefs = self.second_layer_variance * self.k0.maclaurin_coefs
        if self.first_layer_variance:
            coefs += self.first_layer_variance * self.k1.maclaurin_coefs / self.p
        coefs *= self.scale
        return coefs
