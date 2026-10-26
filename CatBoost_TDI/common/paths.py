"""Project-relative data paths and portable manifest serialization."""

import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
ASSETS_ROOT = PROJECT_ROOT / 'assets'


def resolve_data_path(value, base=None):
    """Resolve relative input paths against their manifest directory."""
    path = Path(value)
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path.resolve()


def asset_path(name):
    catalog = json.loads((ASSETS_ROOT / 'catalog.json').read_text(encoding='utf-8'))
    try:
        return ASSETS_ROOT / catalog['assets'][name]
    except KeyError:
        raise KeyError(f'Asset is not included in the data bundle: {name}') from None


def portable_manifest_paths(value, manifest_path):
    """Write project paths relative to the manifest, keeping external paths explicit."""
    base = Path(manifest_path).resolve().parent

    def visit(item, key=None):
        if isinstance(item, dict):
            return {k: visit(v, k) for k, v in item.items()}
        if isinstance(item, list):
            return [visit(v, key) for v in item]
        if isinstance(item, str) and key in ('path', 'roi', 'reference', 'data_root'):
            path = Path(item)
            if path.is_absolute():
                try:
                    path.resolve().relative_to(PROJECT_ROOT)
                    return os.path.relpath(path, base).replace('\\', '/')
                except ValueError:
                    pass
        return item

    return visit(value)
