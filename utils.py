import numpy as np
from scipy import linalg

from sklearn.utils import check_random_state
import models, new_models


def get_model(activation_function=None, kind=None, n_samples=None,
              infinite_width=False, first_layer_variance=0,
              second_layer_variance=0, regime=None, **kwargs):
    if first_layer_variance == second_layer_variance == 0:
        raise ValueError

    # infinite-width models
    if infinite_width:
        if activation_function is None:
            assert kind is not None
            return models.ClassicalKernelModel(
            kind=kind, n_samples=n_samples, **kwargs)
        else:
            assert kind is None
            return models.InfiniteWidthNTKModel(
            activation_function=activation_function,
            n_samples=n_samples,
            first_layer_variance=first_layer_variance,
            second_layer_variance=second_layer_variance, **kwargs)

    if regime == "trainable":
        return new_models.FullyTrainableModel(
        activation_function=activation_function, **kwargs)

    if activation_function == "relu":
        return new_models.ReLUNTK(
        first_layer_variance=first_layer_variance,
        second_layer_variance=second_layer_variance, **kwargs)
    elif activation_function == "quadratic":
        return new_models.QuadraticNTK(
        first_layer_variance=first_layer_variance,
        second_layer_variance=second_layer_variance, **kwargs)
    elif activation_function == "tanh":
        return new_models.TanhNTK(
        first_layer_variance=first_layer_variance,
        second_layer_variance=second_layer_variance, **kwargs)
    elif activation_function == "abs":
        return new_models.AbsoluteValueNTK(
        first_layer_variance=first_layer_variance,
        second_layer_variance=second_layer_variance, **kwargs)
    elif activation_function == "erf":
        return new_models.ErfNTK(
        first_layer_variance=first_layer_variance,
        second_layer_variance=second_layer_variance, **kwargs)
    else:
        raise NotImplementedError


def compute_scores(model, X, y, margin=.8):
    y_pred = model.predict(X)
    if y_pred.ndim != 1:
        assert y_pred.shape == (len(X), 1), y_pred.shape
        y_pred = y_pred.ravel()
    scores = {}
    for loss in ["0/1 loss", "squared loss", "hinge loss",
                 "truncated hinge loss"]:
        if loss == "squared loss":
            scores[loss] = np.mean((y_pred - y) ** 2)
        elif loss == "hinge loss":
            scores[loss] = np.mean(y  * y_pred <= margin)
        elif loss == "truncated hinge loss":
            if margin > 0:
                scores[loss] = np.mean(np.minimum(
                    np.maximum(1 - (1 / margin) * y * y_pred, 0), 1))
        elif loss == "0/1 loss":
            scores[loss] = np.mean(y * y_pred <= 0)

        else:
            raise NotImplementedError(loss)
    return scores


def run_exp(n_samples=None, input_dim=None, hidden_dim=None, regime="rf",
            n_samples_test=1000, activation_function=None, kind=None,
            ntk_first_layer_variance=0, ntk_second_layer_variance=1,
            solver="sklearn", ridge=None, n_models=10, random_state=None,
            debug=False, noise_sigma=1, dataset_index=0, **kwargs):
    assert hidden_dim is not None
    X, y = make_dataset(n_samples=n_samples + n_samples_test,
    # X, y = make_complex_data(n_samples=n_samples + n_samples_test,
    input_dim=input_dim,
    noise_sigma=noise_sigma,
    random_state=random_state)
    X_train, y_train = X[:n_samples], y[:n_samples]
    X_test, y_test = X[n_samples:], y[n_samples:]
    results = []
    infinite_width = np.isinf(hidden_dim)
    if infinite_width:
        n_models = 1
    if debug:
        n_datasets = 1
        n_models = 1
    for model_index in range(n_models):
        try:
            model = get_model(activation_function=activation_function, kind=kind,
            infinite_width=infinite_width,
            regime=regime,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            ridge=ridge, solver=solver,
            random_state=random_state,
            first_layer_variance=ntk_first_layer_variance,
            second_layer_variance=ntk_second_layer_variance,
            **kwargs)
        except NotImplementedError:
            return

        model.fit(X_train, y_train)

        if debug:
            return model

        # compute squared sobolev norm (w.r.t inputs) of the model
        # XXX rm loop!
        grad_norms_squared = []
        for grad in model.grad(X_test):
            assert grad.shape == (input_dim,)
            grad_norms_squared.append(grad@grad)
        sob_norm = np.sqrt(np.mean(grad_norms_squared))

        train_scores = compute_scores(model, X_train, y_train)
        test_scores = compute_scores(model, X_test, y_test, margin=0)
        eps = model.score(X_train, y_train)
        if y_test is not None:
            eps_test = model.score(X_test, y_test)
        else:
            eps_test = np.nan
        print(("%s-width %s (%s), n = %d, d = %d, k = %s, lambda = %.1e,"
               " noise_sigma = %.2f, dataset #%d, model #%d, train error "
               "= %.2f, test error = %.2f, sob_norm = %.2f") % (
            "Infinite" if np.isinf(hidden_dim) else "Finite",
            regime.upper(), activation_function, n_samples, input_dim,
            hidden_dim, ridge, noise_sigma, dataset_index + 1, model_index + 1,
            train_scores["squared loss"], test_scores["squared loss"],
            sob_norm))

        if hasattr(model, "get_stats"):
            stats = model.get_stats(X_test)
            stats.pop("grads", None)
        else:
            stats = {}
        results.append(dict(sob_norm=sob_norm, n_samples=n_samples,
                            input_dim=input_dim, hidden_dim=hidden_dim,
                            activation_function=activation_function,
                            kind=kind, noise_sigma=noise_sigma,
                            regime=regime, v_norm=linalg.norm(model.coef_),
                            dataset_index=dataset_index, ridge=ridge,
                            model_index=model_index,
                            **{"train %s" % k: v for k, v in train_scores.items()},
                            **{"test %s" % k: v for k, v in test_scores.items()},
                            **kwargs, **stats))
    return results


def make_dataset(n_samples, input_dim, random_state=None, noise_sigma=1,
                 return_w=False):
    rng = check_random_state(random_state)
    n_samples = int(n_samples)
    input_dim = int(input_dim)
    X = rng.randn(n_samples, input_dim)
    X /= linalg.norm(X, axis=1, keepdims=True)
    w = rng.randn(input_dim)
    w /= linalg.norm(w)
    y = X@w
    if noise_sigma:
        y += rng.randn(n_samples) * noise_sigma
    if return_w:
        return X, y, w
    return X, y


def make_complex_data(n_samples, input_dim, random_state=None, noise_sigma=1,
                      return_w=False):
    rng = check_random_state(random_state)

    X = rng.randn(n_samples, input_dim)
    X /= linalg.norm(X, axis=1, keepdims=True)
    W = rng.randn(10, input_dim)
    W /= linalg.norm(W, axis=1, keepdims=True)
    y = np.maximum(X@W.T, 0).sum(1)
    if noise_sigma:
        y += rng.randn(n_samples) * noise_sigma
    if return_w:
        return X, y, w
    return X, y
