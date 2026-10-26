"""CatBoost training, prediction and metrics; no raster or plotting side effects."""

from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..common.config import ModelConfig


def metrics(observed, predicted):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    if observed.size == 0 or observed.shape != predicted.shape:
        raise ValueError('Metrics require nonempty, equally sized arrays.')
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError('Metrics received nonfinite values.')
    return dict(n=int(observed.size), MAE=float(mean_absolute_error(observed, predicted)),
                RMSE=float(np.sqrt(mean_squared_error(observed, predicted))),
                R2=float(r2_score(observed, predicted)) if observed.size >= 2 else None)


def make_pool(X, names, config, y=None):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(names) or not np.isfinite(X).all():
        raise ValueError('Model inputs must be finite and match the feature names.')
    unknown = set(config.categorical_features).difference(names)
    if unknown:
        raise ValueError(f'Unknown categorical features: {sorted(unknown)}')
    cat_indices = [i for i, n in enumerate(names) if n in config.categorical_features]
    if cat_indices:
        encoded = X.astype(object)
        for i in cat_indices:
            if not np.allclose(X[:, i], np.rint(X[:, i]), rtol=0, atol=1e-6):
                raise ValueError(f'Categorical feature {names[i]} contains noninteger values.')
            encoded[:, i] = [str(int(v)) for v in X[:, i]]
        X = encoded
    return Pool(X, label=y, feature_names=list(names), cat_features=cat_indices)


def predict(model, X, names, config):
    if len(X) == 0:
        return np.empty(0, dtype=float)
    return model.predict(make_pool(X, names, config))


@dataclass
class HoldoutResult:
    model: CatBoostRegressor
    scores: dict
    train_indices: np.ndarray
    validation_indices: np.ndarray
    predictions: np.ndarray


def fit_holdout(X, y, names, config=ModelConfig()):
    """Notebook-compatible 80:20 split; the 20% set also selects the best tree count.

    These are early-stopping validation scores, not an independent test estimate.
    Outer validation in validation.py keeps its test labels out of this function.
    """
    if len(y) < 5 or len(X) != len(y) or not np.isfinite(y).all():
        raise ValueError('At least five finite training samples are required.')
    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=config.validation_fraction,
                                         random_state=config.split_seed)
    model = CatBoostRegressor(**config.catboost_params(),
                              early_stopping_rounds=config.early_stopping_rounds)
    model.fit(make_pool(X[train_idx], names, config, y[train_idx]),
              eval_set=make_pool(X[val_idx], names, config, y[val_idx]), use_best_model=True)
    predictions = predict(model, X[val_idx], names, config)
    return HoldoutResult(model, metrics(y[val_idx], predictions), train_idx, val_idx, predictions)


def refit_all(X, y, names, config, tree_count):
    """Refit all available clear-sky samples using an already selected tree count."""
    model = CatBoostRegressor(**config.catboost_params(iterations=int(tree_count)))
    model.fit(make_pool(X, names, config, y))
    return model
