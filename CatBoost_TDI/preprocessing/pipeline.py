"""Prepare a paper-compatible stack from independent source rasters, then reconstruct."""

import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import rasterio

from ..common.config import ALL_FEATURES, LEGACY_BANDS, ModelConfig
from ..common.paths import resolve_data_path
from ..common.raster import read_roi, write_raster
from ..reconstruction.pipeline import reconstruct, write_json
from .lst import prepare_lst
from .raster import (
    PAPER_RESAMPLING, align_source, generate_spectral_layers, reference_grid, source_spec,
)


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))

    def resolve(value):
        return str(resolve_data_path(value, base=path.parent))

    def visit(value, key=None):
        if isinstance(value, dict):
            return {k: visit(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v, key) for v in value]
        return resolve(value) if isinstance(value, str) and key in ('path', 'roi', 'reference', 'data_root') else value

    manifest = visit(manifest)
    if manifest.get('schema_version') != 1 or not manifest.get('scenes'):
        raise ValueError('Manifest requires schema_version=1 and at least one scene.')
    ids = [s['id'] for s in manifest['scenes']]
    if len(set(ids)) != len(ids) or any(not i or Path(i).name != i or ':' in i or '/' in i or '\\' in i for i in ids):
        raise ValueError('Scene IDs must be unique simple filenames.')
    def normalize(value):
        return {'path': resolve(value)} if isinstance(value, str) else value

    manifest['static_features'] = {n: normalize(v) for n, v in manifest.get('static_features', {}).items()}
    for scene in manifest['scenes']:
        for group in ('features', 'spectral_bands'):
            if group in scene:
                scene[group] = {n: normalize(v) for n, v in scene[group].items()}
        for name in ('lst', 'cloud_mask', 'quality', 'zenith'):
            if name in scene:
                scene[name] = normalize(scene[name])
    # Refuse unresolved feature names before any expensive raster reads.
    for scene in manifest['scenes']:
        features = {**manifest.get('static_features', {}), **scene.get('features', {})}
        generated = {'NDVI', 'NDWI', 'NDBI', 'Albedo'} if scene.get('spectral_bands') else set()
        if set(features).difference(ALL_FEATURES) or set(ALL_FEATURES).difference(features, generated):
            raise ValueError(f'{scene["id"]}: every one of the 16 predictors must be supplied or derived explicitly.')
        for spec in [scene['lst'], *features.values(), *scene.get('spectral_bands', {}).values()]:
            if not Path(source_spec(spec)['path']).is_file():
                raise FileNotFoundError(source_spec(spec)['path'])
        for mask in ('cloud_mask', 'quality', 'zenith'):
            if scene.get(mask) is not None and not Path(source_spec(scene[mask])['path']).is_file():
                raise FileNotFoundError(source_spec(scene[mask])['path'])
        if not Path(scene.get('reference', source_spec(scene['lst'])['path'])).is_file():
            raise FileNotFoundError(scene.get('reference'))
    return manifest


def _scene_paths(output, scene_id):
    return dict(stack=output / 'stacks' / f'{scene_id}.tif',
                clean_lst=output / 'preprocessing' / f'{scene_id}_lst.tif',
                lst_qa=output / 'preprocessing' / f'{scene_id}_lst_qa.tif',
                report=output / 'preprocessing' / f'{scene_id}_preprocessing.json')


def _reject_existing(paths, overwrite):
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Outputs exist: {existing}; use --overwrite or another output directory.')


@contextmanager
def _stack_writer(destination, profile):
    """Publish a stack only after every requested band was written successfully."""
    descriptor, temporary = tempfile.mkstemp(prefix='.preparing-', suffix='.tif', dir=destination.parent)
    os.close(descriptor)
    try:
        with rasterio.open(temporary, 'w', driver='GTiff', height=profile['height'], width=profile['width'],
                           crs=profile['crs'], transform=profile['transform'], count=len(LEGACY_BANDS),
                           dtype='float64', nodata=-9999., compress='deflate') as dataset:
            yield dataset
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare_scene(scene, manifest, output, overwrite=False):
    scene = copy.deepcopy(scene)
    output = Path(output)
    paths = _scene_paths(output, scene['id'])
    _reject_existing(paths.values(), overwrite)
    roi = scene.get('roi', manifest.get('roi'))
    reference = scene.get('reference', source_spec(scene['lst'])['path'])
    profile = reference_grid(reference, roi, manifest.get('crop_to_roi', True))
    domain = read_roi(roi, profile) if roi else np.ones((profile['height'], profile['width']), dtype=bool)
    options = scene.get('lst_options', {})
    lst, reasons, lst_report = prepare_lst(scene['lst'], profile, options,
                                          cloud_source=scene.get('cloud_mask'), quality_source=scene.get('quality'),
                                          zenith_source=scene.get('zenith'))
    lst[~domain] = np.nan
    reasons[~domain] = np.nan
    features = {**manifest.get('static_features', {}), **scene.get('features', {})}
    if scene.get('spectral_bands'):
        spectral_dir = output / 'derived' / scene['id']
        _reject_existing([spectral_dir / f'{n}.tif' for n in ('NDVI', 'NDWI', 'NDBI', 'Albedo')], overwrite)
        features.update(generate_spectral_layers(scene['spectral_bands'], profile, spectral_dir))
    paths['stack'].parent.mkdir(parents=True, exist_ok=True)
    feature_stats = {}
    with _stack_writer(paths['stack'], profile) as dst:
        dst.write(np.where(np.isfinite(lst), lst, -9999.), 1)
        dst.set_band_description(1, 'LST')
        complete = domain.copy()
        for band, name in enumerate(ALL_FEATURES, start=2):
            spec = source_spec(features[name])
            method = spec.get('resampling', PAPER_RESAMPLING[name])
            if name in ('LC', 'EULUC', 'LCZ') and method != 'nearest':
                raise ValueError(f'Categorical layer {name} requires nearest-neighbour resampling.')
            values, provenance = align_source(spec, profile, method)
            values[~domain] = np.nan
            valid = np.isfinite(values) & (values > -200)
            if name in ('LC', 'EULUC', 'LCZ') and not np.allclose(values[valid], np.rint(values[valid]), atol=1e-6, rtol=0):
                raise ValueError(f'{name}: categorical values are not integers.')
            dst.write(np.where(valid, values, -9999.), band)
            dst.set_band_description(band, name)
            feature_stats[name] = dict(source=provenance, valid_pixels=int(valid.sum()),
                                       range=[float(values[valid].min()), float(values[valid].max())] if valid.any() else None)
            complete &= valid
        dst.update_tags(TDG_BAND_NAMES=json.dumps(LEGACY_BANDS), temperature_unit=options['output_unit'],
                        source_scene=scene['id'], method='explicit source calibration and paper Table 2 resampling')
    write_raster(paths['clean_lst'], lst, profile, dtype='float64', tags={'unit': options['output_unit']})
    write_raster(paths['lst_qa'], reasons, profile, dtype='uint8', nodata=255,
                 tags={'codes': '0 keep; 1 source missing; 2 cloud/unknown; 3 QC; 4 zenith; 5 threshold; 6 dilated; 255 outside'})
    report = dict(scene=scene['id'], timestamp_local=scene.get('timestamp_local'), timestamp_utc=scene.get('timestamp_utc'),
                  source_stage=scene.get('provenance'), reference=reference, roi=roi, band_names=LEGACY_BANDS,
                  grid=dict(height=profile['height'], width=profile['width'], crs=str(profile['crs']),
                            transform=list(profile['transform'])), lst=lst_report, features=feature_stats,
                  domain_pixels=int(domain.sum()), observed_pixels=int(np.isfinite(lst).sum()),
                  all_predictors_complete=int(complete.sum()),
                  outputs={k: str(p.resolve()) for k, p in paths.items()})
    write_json(paths['report'], report)
    return report


def prepare_manifest(manifest_path, output_dir, *, scene_ids=None, run_reconstruction=False,
                      config=ModelConfig(), max_samples=None, refit=False, overwrite=False):
    manifest = load_manifest(manifest_path)
    selected = [s for s in manifest['scenes'] if scene_ids is None or s['id'] in scene_ids]
    if not selected or (scene_ids and set(scene_ids).difference(s['id'] for s in selected)):
        raise ValueError('Requested scene IDs are absent from the manifest.')
    output = Path(output_dir).resolve()
    outputs = [p for s in selected for p in _scene_paths(output, s['id']).values()]
    outputs.extend(output / 'derived' / s['id'] / f'{name}.tif' for s in selected if s.get('spectral_bands')
                   for name in ('NDVI', 'NDWI', 'NDBI', 'Albedo'))
    index_path = output / 'preparation_index.json'
    _reject_existing([*outputs, index_path], overwrite)
    # Guard against output/source collisions even when overwrite is explicitly enabled.
    sources = set()
    for scene in selected:
        specs = [scene['lst'], *scene.get('features', {}).values(), *manifest.get('static_features', {}).values(),
                 *scene.get('spectral_bands', {}).values(),
                 *(scene[n] for n in ('cloud_mask', 'quality', 'zenith') if scene.get(n) is not None)]
        sources.update(Path(source_spec(s)['path']).resolve() for s in specs)
        sources.add(Path(scene.get('reference', source_spec(scene['lst'])['path'])).resolve())
    if sources.intersection(outputs):
        raise ValueError('An output would overwrite an input raster.')
    if run_reconstruction:
        model_outputs = [output / 'reconstruction' / f'{s["id"]}_{suffix}' for s in selected
                         for suffix in ('gapless.tif', 'qa.tif', 'model.cbm', 'report.json')]
        if sources.intersection(model_outputs):
            raise ValueError('A reconstruction output would overwrite an input raster.')
        _reject_existing(model_outputs, overwrite)
    results = []
    for scene in selected:
        print(f'Preparing {scene["id"]}', flush=True)
        prepared = prepare_scene(scene, manifest, output, overwrite)
        item = dict(scene=scene['id'], preprocessing=prepared['outputs'])
        if run_reconstruction:
            reconstructed = reconstruct(prepared['outputs']['stack'], output / 'reconstruction',
                                         config=config, max_samples=max_samples, refit=refit, overwrite=overwrite)
            item['reconstruction'] = reconstructed['outputs']
            item['filling'] = reconstructed['filling']
        results.append(item)
    index = dict(manifest=str(Path(manifest_path).resolve()), schema_version=1,
                 reconstruction_executed=run_reconstruction, scenes=results)
    write_json(index_path, index)
    return index
