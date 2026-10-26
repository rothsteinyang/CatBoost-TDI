"""Sample/spatial cross-validation and real-cloud-mask simulation.

An outer test set never selects the tree count. With a precomputed TDI band,
scores are conditional on that supplied band: its source class means may have
used outer test temperatures. feature_builder enables fold-specific rebuilding.
"""

import numpy as np
from sklearn.model_selection import KFold

from ..common.config import ModelConfig
from ..reconstruction.model import fit_holdout, metrics, predict, refit_all


def _evaluation_features(scene, train_indices, test_indices, feature_builder):
    if feature_builder is None:
        return scene.features.reshape(-1, len(scene.feature_names))
    visible = np.zeros(scene.lst.size, dtype=bool)
    visible[train_indices] = True
    built = np.asarray(feature_builder(visible.reshape(scene.lst.shape)))
    if built.shape != scene.features.shape:
        raise ValueError('feature_builder must return the full (height, width, feature) array.')
    flat = built.reshape(-1, len(scene.feature_names))
    selected = np.r_[train_indices, test_indices]
    if not np.isfinite(flat[selected]).all() or (flat[selected] <= scene.invalid_threshold).any():
        raise ValueError('Rebuilt features are invalid for selected training/test samples.')
    return flat


def _fit_outer(X, y, names, config, refit):
    result = fit_holdout(X, y, names, config)
    model = refit_all(X, y, names, config, result.model.tree_count_) if refit else result.model
    return model, result.model.tree_count_


def cross_validate(scene, *, mode='sample', n_splits=10, config=ModelConfig(),
                   max_samples=None, feature_builder=None, refit=True, block_labels=None):
    _, y, indices = scene.samples(max_samples=max_samples, seed=config.split_seed)
    if n_splits < 2 or len(indices) < max(n_splits, 10):
        raise ValueError('Insufficient samples or invalid number of folds.')
    if mode == 'sample':
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=config.split_seed).split(indices)
    elif mode == 'spatial':
        if block_labels is None:
            width = scene.lst.shape[1]
            if width < n_splits:
                raise ValueError('Raster is narrower than the number of vertical strips.')
            # Equal-width vertical strips in the raster grid, not equal-sized sample batches.
            groups = np.minimum((indices % width) * n_splits // width, n_splits - 1)
            if len(np.unique(groups)) != n_splits:
                raise ValueError('Some spatial strips have no eligible samples; reduce n_splits or change the ROI.')
        else:
            if block_labels.shape != scene.lst.shape:
                raise ValueError('block_labels must match the LST grid.')
            groups = np.asarray(block_labels).ravel()[indices]
            if not np.isfinite(groups).all() or len(np.unique(groups)) != n_splits:
                raise ValueError('Every selected pixel must have one of exactly n_splits block labels.')
        splits = ((np.flatnonzero(groups != group), np.flatnonzero(groups == group))
                  for group in np.unique(groups))
    else:
        raise ValueError('mode must be sample or spatial.')
    oof = np.full(len(y), np.nan)
    folds = []
    for fold, (train, test) in enumerate(splits, start=1):
        if len(test) < 2 or len(train) < 5:
            raise ValueError(f'Fold {fold} has insufficient training or test samples.')
        flat = _evaluation_features(scene, indices[train], indices[test], feature_builder)
        model, trees = _fit_outer(flat[indices[train]], y[train], scene.feature_names, config, refit)
        oof[test] = predict(model, flat[indices[test]], scene.feature_names, config)
        folds.append(dict(fold=fold, train_n=len(train), trees=trees, **metrics(y[test], oof[test])))
    if not np.isfinite(oof).all():
        raise ValueError('Some selected samples did not receive an outer-fold prediction.')
    return dict(mode=mode, n_splits=n_splits, pooled=metrics(y, oof), folds=folds,
                fold_mean={k: float(np.mean([f[k] for f in folds])) for k in ('MAE', 'RMSE', 'R2')},
                fold_std={k: float(np.std([f[k] for f in folds])) for k in ('MAE', 'RMSE', 'R2')},
                early_stopping='inner_validation_only', refit_outer_training=refit,
                tdi_provenance='rebuilt_from_training_visibility' if feature_builder else 'supplied_precomputed_band',
                tdi_rebuild_callback_used=feature_builder is not None,
                tdi_test_label_exclusion='caller must enforce visibility mask' if feature_builder else 'not verified for supplied TDI',
                samples_capped=max_samples is not None, config=config.as_dict())


def validate_cloud_mask(scene, mask, *, config=ModelConfig(), max_samples=None,
                        feature_builder=None, refit=True):
    """0=clear, 1=simulated cloud, NaN=unknown; original missing LST is not scored."""
    mask = np.asarray(mask)
    if mask.shape != scene.lst.shape:
        raise ValueError('Cloud mask must be aligned to the LST grid.')
    finite = mask[np.isfinite(mask)]
    if not np.isin(finite, (0, 1)).all():
        raise ValueError('Cloud mask must contain only 0, 1 or NaN.')
    _, _, train_indices = scene.samples(mask == 0, max_samples, config.split_seed)
    test_indices = np.flatnonzero(scene.training_mask & (mask == 1))
    if len(train_indices) < 5 or len(test_indices) < 2:
        raise ValueError('Insufficient clear/cloud validation samples.')
    flat = _evaluation_features(scene, train_indices, test_indices, feature_builder)
    y = scene.lst.ravel()
    model, trees = _fit_outer(flat[train_indices], y[train_indices], scene.feature_names, config, refit)
    estimated = predict(model, flat[test_indices], scene.feature_names, config)
    eligible = scene.training_mask & np.isin(mask, (0, 1))
    return dict(mode='cloud_mask', cloud_scores=metrics(y[test_indices], estimated),
                train_n=len(train_indices), eligible_n=int(eligible.sum()), trees=trees,
                effective_cloud_fraction=float(len(test_indices) / eligible.sum()),
                early_stopping='clear_region_inner_validation_only', refit_clear_training=refit,
                tdi_provenance='rebuilt_from_training_visibility' if feature_builder else 'supplied_precomputed_band',
                tdi_rebuild_callback_used=feature_builder is not None,
                tdi_test_label_exclusion='caller must enforce visibility mask' if feature_builder else 'not verified for supplied TDI',
                config=config.as_dict())
