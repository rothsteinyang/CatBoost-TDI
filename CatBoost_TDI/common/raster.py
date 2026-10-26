"""Raster input, explicit feature selection, masks, alignment and GeoTIFF output."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject

from .config import INVALID_THRESHOLD, KEY_FEATURES, LEGACY_BANDS


@dataclass
class Scene:
    lst: np.ndarray
    features: np.ndarray  # (height, width, feature)
    feature_names: tuple[str, ...]
    profile: dict
    domain: np.ndarray
    source: str = ''
    invalid_threshold: float = INVALID_THRESHOLD

    def __post_init__(self):
        if self.lst.ndim != 2 or self.features.shape != (*self.lst.shape, len(self.feature_names)):
            raise ValueError('LST, features and feature names have incompatible shapes.')
        if self.domain.shape != self.lst.shape or self.domain.dtype != np.bool_:
            raise ValueError('domain must be a boolean array on the LST grid.')
        if len(set(self.feature_names)) != len(self.feature_names) or not self.feature_names:
            raise ValueError('Feature names must be nonempty and unique.')

    @property
    def observed(self):
        return self.domain & np.isfinite(self.lst) & (self.lst > self.invalid_threshold)

    @property
    def valid_features(self):
        return np.isfinite(self.features).all(axis=2) & (self.features > self.invalid_threshold).all(axis=2)

    @property
    def training_mask(self):
        return self.observed & self.valid_features

    def samples(self, mask=None, max_samples=None, seed=42):
        selected = self.training_mask.copy()
        if mask is not None:
            if mask.shape != self.lst.shape or mask.dtype != np.bool_:
                raise ValueError('Sample mask must be boolean and match the scene.')
            selected &= mask
        indices = np.flatnonzero(selected)
        if max_samples is not None:
            if max_samples < 5:
                raise ValueError('max_samples must be at least 5.')
            if len(indices) > max_samples:
                indices = np.sort(np.random.default_rng(seed).choice(indices, max_samples, replace=False))
        return self.features.reshape(-1, len(self.feature_names))[indices], self.lst.ravel()[indices], indices


def read_roi(path, profile):
    import geopandas as gpd
    frame = gpd.read_file(path)
    if frame.crs is None or profile.get('crs') is None:
        raise ValueError('Both ROI and raster must declare their CRS.')
    frame = frame.to_crs(profile['crs'])
    shapes = [g.__geo_interface__ for g in frame.geometry if g is not None and not g.is_empty]
    if not shapes:
        raise ValueError('ROI contains no geometries.')
    return geometry_mask(shapes, out_shape=(profile['height'], profile['width']),
                         transform=profile['transform'], invert=True, all_touched=False)


def read_scene(path, feature_names=KEY_FEATURES, band_names=LEGACY_BANDS,
               roi_path=None, window=None, invalid_threshold=INVALID_THRESHOLD, tdi_path=None):
    """Read explicitly mapped bands; do not infer order from ENVI descriptions.

    Some local ENVI headers have GBK descriptions, which rasterio cannot decode.
    Pixel data and georeferencing remain readable; descriptions are not accessed.
    A trailing CNN band is ignored when the first 17 bands follow LEGACY_BANDS.
    """
    feature_names, band_names = tuple(feature_names), tuple(band_names)
    if 'LST' in feature_names or len(set(band_names)) != len(band_names):
        raise ValueError('Feature bands must exclude LST; band names must be unique.')
    if not set(('LST', *feature_names)).issubset(band_names):
        raise ValueError('Requested variables are absent from the explicit band mapping.')
    indexes = [band_names.index(name) + 1 for name in ('LST', *feature_names)]
    with rasterio.open(path) as src:
        if src.count < len(band_names):
            raise ValueError(f'{path}: expected at least {len(band_names)} mapped bands; found {src.count}. '
                             'A 10-band CNN/Ta stack is not a paper TDI stack.')
        # Some ENVI bands contain the float64 minimum without a nodata tag.
        # Clean before any narrowing conversion; keep observation precision.
        data = src.read(indexes, window=window, masked=True).astype('float64').filled(np.nan)
        profile = dict(src.profile)
        if window is not None:
            profile.update(transform=src.window_transform(window), height=data.shape[1], width=data.shape[2])
    if tdi_path is not None:
        if 'TDI' not in feature_names:
            raise ValueError('A TDI override requires TDI in feature_names.')
        data[1 + feature_names.index('TDI')] = align_band(tdi_path, profile)
    data[~np.isfinite(data) | (data <= invalid_threshold)] = np.nan
    # Without a polygon, infer the footprint from any usable requested variable.
    domain = np.isfinite(data).any(axis=0)
    if roi_path is not None:
        domain = read_roi(roi_path, profile)
    return Scene(data[0], np.moveaxis(data[1:], 0, -1), feature_names, profile,
                 domain, str(Path(path).resolve()), invalid_threshold)


def align_band(path, profile, resampling=Resampling.nearest):
    """Align a single-band raster, preserving unknown/no-data as NaN."""
    out = np.full((profile['height'], profile['width']), np.nan, dtype='float64')
    with rasterio.open(path) as src:
        if src.crs is None or profile.get('crs') is None:
            raise ValueError('Raster alignment requires both CRS definitions.')
        arr = src.read(1, masked=True).astype('float64').filled(np.nan)
        arr[~np.isfinite(arr)] = np.nan
        reproject(arr, out, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=np.nan, dst_transform=profile['transform'], dst_crs=profile['crs'],
                  dst_nodata=np.nan, resampling=resampling)
    return out


def write_raster(path, data, profile, *, nodata=-9999.0, dtype='float32', tags=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.shape != (profile['height'], profile['width']):
        raise ValueError('Output shape does not match its spatial profile.')
    options = dict(driver='GTiff', count=1, height=data.shape[0], width=data.shape[1],
                   crs=profile.get('crs'), transform=profile['transform'],
                   dtype=dtype, nodata=nodata, compress='deflate')
    encoded = np.where(np.isfinite(data), data, nodata).astype(dtype)
    with rasterio.open(path, 'w', **options) as dst:
        dst.write(encoded, 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path
