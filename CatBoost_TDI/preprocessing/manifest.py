"""Select scenes from an explicit data-bundle workflow manifest."""

import json
from pathlib import Path

from ..common.paths import ASSETS_ROOT, resolve_data_path


def build_manifest(data_root=None, scene_ids=None, reference_dir=None):
    """Load a bundle's workflow and optionally select scenes or reference grids."""
    from .pipeline import load_manifest

    root = resolve_data_path(data_root if data_root is not None else ASSETS_ROOT)
    catalog_path = root / 'catalog.json'
    if catalog_path.is_file():
        catalog = json.loads(catalog_path.read_text(encoding='utf-8-sig'))
        workflow = root / catalog['workflow_manifest']
    else:
        workflow = root / 'workflow.json'
    manifest = load_manifest(workflow)
    if scene_ids is not None:
        known = {scene['id'] for scene in manifest['scenes']}
        missing = set(scene_ids).difference(known)
        if missing:
            raise ValueError(f'Requested scenes are absent from the data bundle: {sorted(missing)}')
        manifest['scenes'] = [scene for scene in manifest['scenes'] if scene['id'] in scene_ids]
        if not manifest['scenes']:
            raise ValueError('At least one scene must be selected.')
    if reference_dir is not None:
        reference_root = Path(reference_dir).resolve()
        for scene in manifest['scenes']:
            reference = reference_root / f'{scene["id"]}.dat'
            if not reference.is_file():
                raise FileNotFoundError(reference)
            scene['reference'] = str(reference)
        manifest['crop_to_roi'] = False
    return manifest
