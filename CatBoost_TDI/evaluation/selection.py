"""SHAP importance, cross-scene relative importance and Top-N ablation."""

import numpy as np

from ..common.config import ModelConfig, PAPER_RANKING
from ..reconstruction.model import fit_holdout, make_pool


def shap_importance(model, X, names, config=ModelConfig(), max_samples=5000):
    if max_samples < 1 or len(X) == 0:
        raise ValueError('SHAP requires a positive cap and nonempty samples.')
    if len(X) > max_samples:
        indices = np.random.default_rng(config.split_seed).choice(len(X), max_samples, replace=False)
        X = X[indices]
    # CatBoost's native exact tree SHAP avoids an additional shap dependency.
    values = model.get_feature_importance(make_pool(X, names, config), type='ShapValues')[:, :-1]
    absolute = np.abs(values).mean(axis=0)
    total = absolute.sum()
    relative = absolute / total if total else np.zeros_like(absolute)
    return dict(samples=len(X), mean_absolute=dict(zip(names, absolute.tolist())),
                relative=dict(zip(names, relative.tolist())))


def average_relative_importance(per_scene):
    """Normalize within each scene first, then average with equal scene weights."""
    if not per_scene:
        raise ValueError('At least one scene importance result is required.')
    names = tuple(per_scene[0]['relative'])
    if any(set(item['relative']) != set(names) for item in per_scene):
        raise ValueError('Every scene must use the same feature names.')
    means = np.mean([[item['relative'][n] for n in names] for item in per_scene], axis=0)
    return dict(sorted(zip(names, means.tolist()), key=lambda item: item[1], reverse=True))


def top_n_ablation(scene, *, ranking=PAPER_RANKING, counts=range(16, 4, -1),
                    config=ModelConfig(), max_samples=None):
    names = scene.feature_names
    ranking, counts = tuple(ranking), tuple(counts)
    if set(ranking) != set(names) or len(ranking) != len(names):
        raise ValueError('Ranking must include every input feature exactly once.')
    if any(n < 1 or n > len(names) for n in counts):
        raise ValueError('Top-N counts exceed the available feature count.')
    X, y, _ = scene.samples(max_samples=max_samples, seed=config.split_seed)
    results = []
    for n in counts:
        selected = ranking[:n]
        subconfig = ModelConfig(**{**config.as_dict(), 'categorical_features': tuple(
            f for f in config.categorical_features if f in selected)})
        result = fit_holdout(X[:, [names.index(f) for f in selected]], y, selected, subconfig)
        results.append(dict(top_n=n, features=selected, trees=result.model.tree_count_, **result.scores))
    return dict(common_sample_n=len(y), ranking=ranking, results=results,
                score_role='early_stopping_validation_exploratory', config=config.as_dict())


def tdi_ablation(scene, config=ModelConfig(), max_samples=None):
    """Compare the same samples and split; do not change the mask when dropping TDI."""
    if 'TDI' not in scene.feature_names:
        raise ValueError('TDI ablation requires a TDI feature.')
    without = tuple(n for n in scene.feature_names if n != 'TDI')
    X, y, _ = scene.samples(max_samples=max_samples, seed=config.split_seed)
    with_result = fit_holdout(X, y, scene.feature_names, config)
    subconfig = ModelConfig(**{**config.as_dict(), 'categorical_features': tuple(
        n for n in config.categorical_features if n != 'TDI')})
    without_result = fit_holdout(X[:, [scene.feature_names.index(n) for n in without]], y, without, subconfig)
    return dict(with_tdi=with_result.scores, without_tdi=without_result.scores,
                rmse_improvement=without_result.scores['RMSE'] - with_result.scores['RMSE'],
                common_sample_n=len(y), score_role='early_stopping_validation_exploratory')
