"""Compose raster preparation, training, gap-only prediction and auditable outputs."""

import json
from pathlib import Path

import numpy as np

from ..common.config import KEY_FEATURES, ModelConfig
from .model import fit_holdout, predict, refit_all
from ..common.raster import read_scene, write_raster


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def fill_gaps(scene, model, config=ModelConfig(), batch_size=100000):
    """Equation (4): keep every valid observed pixel; predict eligible gaps only.

    QA: 0=outside, 1=observation, 2=filled, 3=gap without usable features.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    observed = scene.observed
    missing = scene.domain & ~observed
    eligible = missing & scene.valid_features
    out = np.full(scene.lst.shape, np.nan, dtype=np.result_type(scene.lst.dtype, np.float32))
    out[observed] = scene.lst[observed]
    qa = np.zeros(scene.lst.shape, dtype='uint8')
    qa[observed], qa[missing] = 1, 3
    flat = scene.features.reshape(-1, len(scene.feature_names))
    indices = np.flatnonzero(eligible)
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        estimated = predict(model, flat[selected], scene.feature_names, config)
        if not np.isfinite(estimated).all():
            raise ValueError('Model produced nonfinite LST predictions.')
        out.ravel()[selected] = estimated
        qa.ravel()[selected] = 2
    if not np.array_equal(out[observed], scene.lst[observed]):
        raise AssertionError('Observed LST was modified during gap filling.')
    stats = dict(domain_pixels=int(scene.domain.sum()), observed_pixels=int(observed.sum()),
                 original_gap_pixels=int(missing.sum()), filled_pixels=int((qa == 2).sum()),
                 unfilled_pixels=int((qa == 3).sum()))
    stats['gap_fraction'] = stats['original_gap_pixels'] / stats['domain_pixels'] if stats['domain_pixels'] else 0.0
    return out, qa, stats


def inspect_scene(scene):
    observed = scene.lst[scene.observed]
    return dict(source=scene.source, shape=list(scene.lst.shape), features=list(scene.feature_names),
                crs=str(scene.profile.get('crs')), pixel_size=[abs(scene.profile['transform'].a), abs(scene.profile['transform'].e)],
                domain_pixels=int(scene.domain.sum()), observed_pixels=int(scene.observed.sum()),
                training_pixels=int(scene.training_mask.sum()),
                predictable_gaps=int((scene.domain & ~scene.observed & scene.valid_features).sum()),
                observed_range=[float(observed.min()), float(observed.max())] if len(observed) else None,
                temperature_unit='source values; no automatic Celsius/Kelvin conversion')


def reconstruct(input_path, output_dir, *, feature_names=KEY_FEATURES, config=ModelConfig(),
                roi_path=None, max_samples=None, min_samples=100, refit=False,
                overwrite=False, batch_size=100000, tdi_path=None):
    stem = Path(input_path).stem
    output_dir = Path(output_dir)
    paths = {key: output_dir / f'{stem}_{suffix}' for key, suffix in (
        ('lst', 'gapless.tif'), ('qa', 'qa.tif'), ('model', 'model.cbm'), ('report', 'report.json'))}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Outputs already exist; choose another directory or use --overwrite: {existing}')
    scene = read_scene(input_path, feature_names=feature_names, roi_path=roi_path, tdi_path=tdi_path)
    X, y, _ = scene.samples(max_samples=max_samples, seed=config.split_seed)
    if len(y) < max(5, min_samples):
        raise ValueError(f'{input_path}: only {len(y)} training samples; require {max(5, min_samples)}.')
    fit = fit_holdout(X, y, scene.feature_names, config)
    final_model = refit_all(X, y, scene.feature_names, config, fit.model.tree_count_) if refit else fit.model
    filled, qa, stats = fill_gaps(scene, final_model, config, batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_raster(paths['lst'], filled, scene.profile, dtype=str(filled.dtype),
                 tags=dict(method='TDI + CatBoost; observed values preserved', unit='same_as_source'))
    write_raster(paths['qa'], qa, scene.profile, nodata=0, dtype='uint8',
                 tags=dict(classes='0 outside; 1 observed; 2 filled; 3 unfilled'))
    final_model.save_model(str(paths['model']))
    import catboost
    import rasterio
    report = dict(source=scene.source, scene=inspect_scene(scene), features=list(scene.feature_names),
                  config=config.as_dict(), roi=str(roi_path) if roi_path else None,
                  domain_definition='ROI pixel centres' if roi_path else 'any usable requested band',
                  training_samples=len(y), holdout_validation=fit.scores,
                  score_role='early_stopping_validation_not_independent_test',
                  tdi_provenance=str(Path(tdi_path).resolve()) if tdi_path else 'supplied_precomputed_band',
                  selected_tree_count=fit.model.tree_count_,
                  refit_all_clear_samples=refit, samples_capped=max_samples is not None, filling=stats,
                  outputs={k: str(p.resolve()) for k, p in paths.items()},
                  versions=dict(numpy=np.__version__, rasterio=rasterio.__version__, catboost=catboost.__version__))
    write_json(paths['report'], report)
    return report
