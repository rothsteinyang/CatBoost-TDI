"""Command-line entry points. All paths and expensive operations are explicit."""

import argparse
import json
from pathlib import Path
import sys

from .common.config import ALL_FEATURES, KEY_FEATURES, ModelConfig, PAPER_RANKING
from .common.paths import ASSETS_ROOT, portable_manifest_paths


def _common(parser, *, output_required=True, default_features='key'):
    parser.add_argument('--input', nargs='+', type=Path, required=True, help='17-band paper stacks (.dat or .tif)')
    parser.add_argument('--output', type=Path, required=output_required, help='Output directory')
    parser.add_argument('--roi', type=Path, help='Optional polygon ROI; pixel-centre inclusion')
    parser.add_argument('--tdi', type=Path, help='Optional external TDI raster; replaces the mapped TDI band')
    parser.add_argument('--features', choices=('key', 'all'), default=default_features)
    parser.add_argument('--max-samples', type=int, help='Optional reproducible cap for quick runs; default uses all')
    parser.add_argument('--config', type=Path, help='JSON with ModelConfig fields')
    parser.add_argument('--iterations', type=int, help='Override configured iteration cap')
    parser.add_argument('--depth', type=int, help='Override configured tree depth')
    parser.add_argument('--threads', type=int, help='Override configured CatBoost thread_count')
    parser.add_argument('--overwrite', action='store_true')


def _model_config(args):
    values = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
    for field, value in (('iterations', args.iterations), ('depth', args.depth), ('thread_count', args.threads)):
        if value is not None:
            values[field] = value
    return ModelConfig(**values)


def _preflight(paths, overwrite):
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Outputs already exist: {existing}; use --overwrite or a new directory.')


def build_parser():
    parser = argparse.ArgumentParser(prog='python -m CatBoost_TDI',
                                     description='Thermal diffusion guided ECOSTRESS LST reconstruction')
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='Inspect masks and the explicit band mapping')
    _common(inspect, output_required=False)
    reconstruct = commands.add_parser('reconstruct', help='Train and fill only missing LST pixels')
    _common(reconstruct)
    reconstruct.add_argument('--refit-all', action='store_true', help='Refit all clear samples at the selected tree count')
    validate = commands.add_parser('validate', help='Outer-fold or cloud-mask validation')
    _common(validate)
    validate.add_argument('--mode', choices=('sample', 'spatial', 'cloud'), default='sample')
    validate.add_argument('--folds', type=int, default=10)
    validate.add_argument('--cloud-mask', type=Path, help='Raster with 0 clear, 1 cloud; other valid codes rejected')
    validate.add_argument('--blocks', type=Path, help='Optional spatial fold label raster; otherwise vertical strips')
    select = commands.add_parser('select-features', help='SHAP and Top-16 through Top-5 ablation')
    _common(select, default_features='all')
    select.add_argument('--ranking', choices=('paper', 'shap'), default='paper')
    select.add_argument('--top-n', type=int, nargs='+', default=list(range(16, 4, -1)))
    select.add_argument('--shap-samples', type=int, default=5000)
    tdi = commands.add_parser('tdi', help='Rebuild equations (1)-(3); original ArcMap variogram not available')
    tdi.add_argument('--lc', type=Path, required=True, help='LC raster projected in metres')
    tdi.add_argument('--lst', type=Path, required=True)
    tdi.add_argument('--output', type=Path, required=True, help='Output directory')
    tdi.add_argument('--scale-m', type=float, required=True, help='Exponential e-fold scale, NOT practical range')
    tdi.add_argument('--partial-sill', type=float, default=1.0)
    tdi.add_argument('--nugget', type=float, default=0.0)
    tdi.add_argument('--distance-m', type=float, default=350.0)
    tdi.add_argument('--classes', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    tdi.add_argument('--visible-mask', type=Path, help='1 permits LST in class means; 0/unknown excludes it')
    tdi.add_argument('--overwrite', action='store_true')
    distance = commands.add_parser('distance', help='50 m buffer summaries and cubic diffusion distance')
    distance.add_argument('--lst', type=Path, required=True)
    distance.add_argument('--sources', type=Path, required=True, help='Representative source point/polygon file')
    distance.add_argument('--output', type=Path, required=True)
    distance.add_argument('--class-field')
    distance.add_argument('--step-m', type=float, default=50.0)
    distance.add_argument('--max-distance-m', type=float, default=1000.0)
    distance.add_argument('--cumulative', action='store_true', help='Use cumulative buffers instead of source-code rings')
    distance.add_argument('--buffer-crs', help='Optional projected metre CRS; defaults to estimated local UTM')
    distance.add_argument('--legacy-grid', action='store_true', help='Original 500-point turning-point approximation')
    distance.add_argument('--overwrite', action='store_true')
    discover = commands.add_parser('discover', help='Select scenes from a data-bundle workflow and write a preparation manifest')
    discover.add_argument('--data-root', type=Path, default=ASSETS_ROOT,
                          help='Data bundle containing workflow.json; defaults to the project assets directory')
    discover.add_argument('--output', type=Path, required=True, help='Manifest JSON path')
    discover.add_argument('--scenes', nargs='+', help='Optional exact local-time scene IDs')
    discover.add_argument('--reference-dir', type=Path, help='Optional .dat stack directory, used only for grid geometry')
    discover.add_argument('--overwrite', action='store_true')
    for name in ('prepare', 'workflow'):
        preparation = commands.add_parser(name, help='Prepare source rasters' if name == 'prepare' else 'Prepare source rasters and reconstruct LST')
        preparation.add_argument('--manifest', type=Path, required=True)
        preparation.add_argument('--output', type=Path, required=True, help='New output root; source data is read only')
        preparation.add_argument('--scenes', nargs='+', help='Optional exact manifest scene IDs')
        preparation.add_argument('--overwrite', action='store_true')
        if name == 'workflow':
            preparation.add_argument('--config', type=Path)
            preparation.add_argument('--iterations', type=int)
            preparation.add_argument('--depth', type=int)
            preparation.add_argument('--threads', type=int)
            preparation.add_argument('--max-samples', type=int)
            preparation.add_argument('--refit-all', action='store_true')
    return parser


def run(args):
    from .reconstruction.pipeline import inspect_scene, reconstruct, write_json
    from .common.raster import align_band, read_scene
    if args.command == 'discover':
        from .preprocessing.manifest import build_manifest
        _preflight([args.output], args.overwrite)
        manifest = build_manifest(args.data_root, args.scenes, args.reference_dir)
        write_json(args.output, portable_manifest_paths(manifest, args.output))
        print(f'{len(manifest["scenes"])} scenes: {args.output.resolve()}')
        return
    if args.command in ('prepare', 'workflow'):
        from .preprocessing.pipeline import prepare_manifest
        config = _model_config(args) if args.command == 'workflow' else ModelConfig()
        result = prepare_manifest(args.manifest, args.output, scene_ids=args.scenes,
                                   run_reconstruction=args.command == 'workflow', config=config,
                                   max_samples=getattr(args, 'max_samples', None),
                                   refit=getattr(args, 'refit_all', False), overwrite=args.overwrite)
        print(f'{len(result["scenes"])} scenes completed: {args.output.resolve()}')
        return
    if args.command in ('inspect', 'reconstruct', 'validate', 'select-features'):
        config = _model_config(args)
        features = KEY_FEATURES if args.features == 'key' else ALL_FEATURES
        stems = [p.stem for p in args.input]
        if len(set(stems)) != len(stems):
            raise ValueError('Input basenames must be unique in one batch.')
        if args.command == 'select-features' and args.features != 'all':
            raise ValueError('Feature selection expects --features all (16 predictors).')
        if args.command == 'validate':
            if args.mode == 'cloud' and args.cloud_mask is None:
                raise ValueError('--cloud-mask is required for cloud validation.')
            if args.blocks is not None and args.mode != 'spatial':
                raise ValueError('--blocks is only applicable to spatial validation.')
        suffix = {'inspect': 'inspection', 'validate': 'validation', 'select-features': 'selection'}.get(args.command)
        if suffix and args.output:
            _preflight([args.output / f'{stem}_{suffix}.json' for stem in stems], args.overwrite)
        if args.command == 'select-features':
            _preflight([args.output / 'global_relative_importance.json'], args.overwrite)
        if args.command == 'reconstruct':
            _preflight([args.output / f'{stem}_{ending}' for stem in stems for ending in (
                'gapless.tif', 'qa.tif', 'model.cbm', 'report.json')], args.overwrite)
        importances = []
        scenes = []
        for path in args.input:
            print(f'{args.command}: {path}', flush=True)
            if args.command == 'reconstruct':
                result = reconstruct(path, args.output, feature_names=features, config=config,
                                     roi_path=args.roi, max_samples=args.max_samples,
                                     refit=args.refit_all, overwrite=args.overwrite, tdi_path=args.tdi)
                print(json.dumps(result['filling'], ensure_ascii=False))
                print(result['outputs']['report'])
                continue
            scene = read_scene(path, feature_names=features, roi_path=args.roi, tdi_path=args.tdi)
            if args.command == 'inspect':
                result = inspect_scene(scene)
            elif args.command == 'validate':
                from .evaluation.validation import cross_validate, validate_cloud_mask
                if args.mode == 'cloud':
                    result = validate_cloud_mask(scene, align_band(args.cloud_mask, scene.profile),
                                                 config=config, max_samples=args.max_samples)
                else:
                    blocks = align_band(args.blocks, scene.profile) if args.blocks else None
                    result = cross_validate(scene, mode=args.mode, n_splits=args.folds,
                                             config=config, max_samples=args.max_samples, block_labels=blocks)
            else:
                from .reconstruction.model import fit_holdout
                from .evaluation.selection import shap_importance
                X, y, _ = scene.samples(max_samples=args.max_samples, seed=config.split_seed)
                fit = fit_holdout(X, y, scene.feature_names, config)
                importance = shap_importance(fit.model, X[fit.train_indices], scene.feature_names,
                                             config, max_samples=args.shap_samples)
                importances.append(importance)
                # Defer ablation until the global relative SHAP ranking is known.
                scenes.append((path, scene, fit.scores, importance))
                continue
            if args.output:
                target = args.output / f'{path.stem}_{suffix}.json'
                write_json(target, result)
                print(str(target.resolve()))
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == 'select-features':
            from .evaluation.selection import average_relative_importance, top_n_ablation
            average = average_relative_importance(importances)
            ranking = PAPER_RANKING if args.ranking == 'paper' else tuple(average)
            for path, scene, scores, importance in scenes:
                result = top_n_ablation(scene, ranking=ranking, counts=args.top_n,
                                       config=config, max_samples=args.max_samples)
                result.update(source=scene.source, full_model_validation=scores, shap=importance,
                              ranking_source=args.ranking)
                target = args.output / f'{path.stem}_selection.json'
                write_json(target, result)
                print(str(target.resolve()))
            write_json(args.output / 'global_relative_importance.json', average)
        return
    if args.command == 'tdi':
        from .thermal.tdi import ExponentialVariogram, build_tdi_raster
        target = args.output / f'{args.lst.stem}_tdi.tif'
        _preflight([target, target.with_suffix('.json')], args.overwrite)
        result = build_tdi_raster(args.lc, args.lst, target,
                                  ExponentialVariogram(args.scale_m, args.partial_sill, args.nugget),
                                  distance_m=args.distance_m, classes=tuple(args.classes),
                                  visible_mask_path=args.visible_mask)
        write_json(target.with_suffix('.json'), result)
        print(str(target.resolve()))
        return
    if args.command == 'distance':
        import pandas as pd
        from .thermal.distance import extract_buffer_temperatures, fit_diffusion_distance
        buffers = args.output / f'{args.sources.stem}_buffers.csv'
        fitted = args.output / f'{args.sources.stem}_distances.json'
        _preflight([buffers, fitted], args.overwrite)
        records = extract_buffer_temperatures(args.lst, args.sources, step_m=args.step_m,
                                              max_distance_m=args.max_distance_m, rings=not args.cumulative,
                                              class_field=args.class_field, buffer_crs=args.buffer_crs)
        frame = pd.DataFrame(records)
        if frame.empty:
            raise ValueError('No source-buffer summaries were produced.')
        summaries = []
        for source_id, group in frame.groupby('source_id', sort=False):
            try:
                summary = fit_diffusion_distance(group.distance_m, group.mean_lst, legacy_grid=args.legacy_grid)
            except ValueError as exc:
                summary = dict(status='insufficient_valid_buffer_data', reason=str(exc))
            summaries.append(dict(source_id=source_id, source_class=group.source_class.iloc[0], **summary))
        args.output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(buffers, index=False, encoding='utf-8-sig')
        write_json(fitted, summaries)
        print(str(fitted.resolve()))


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, FileExistsError, FileNotFoundError, TypeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    return 0
