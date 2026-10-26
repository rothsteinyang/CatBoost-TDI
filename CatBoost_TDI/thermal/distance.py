"""Thermal diffusion distance: buffer LST summaries and a cubic turning point.

Based on source notebook cells 66 and 71. The source uses 50 m rings; cumulative
buffers are also supported because the paper describes concentric buffers.
"""

import numpy as np

from ..common.config import INVALID_THRESHOLD


def fit_diffusion_distance(distances, temperatures, *, legacy_grid=False):
    x, y = np.asarray(distances, dtype=float), np.asarray(temperatures, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError('Distance and temperature must be matching 1D arrays.')
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(np.unique(x)) < 4 or (x < 0).any():
        raise ValueError('Cubic fitting requires four distinct nonnegative distances.')
    order = np.argsort(x)
    x, y = x[order], y[order]
    polynomial = np.poly1d(np.polyfit(x, y, 3))
    derivative = np.polyder(polynomial)
    radius = None
    if legacy_grid:
        grid = np.linspace(x.min(), x.max(), 500)
        changes = np.flatnonzero(np.diff(np.sign(derivative(grid))) != 0)
        if len(changes):
            radius = float(grid[changes[0] + 1])
    else:
        # A stationary point is valid only when the first derivative changes sign.
        roots = sorted(float(r.real) for r in np.roots(derivative)
                       if abs(r.imag) < 1e-7 and x.min() < r.real < x.max())
        for r in roots:
            eps = min(1e-3 * max(1, x.max() - x.min()), (r - x.min()) / 2, (x.max() - r) / 2)
            if derivative(r - eps) * derivative(r + eps) < 0:
                radius = r
                break
    at_zero = float(polynomial(0))
    at_radius = float(polynomial(radius)) if radius is not None else None
    return dict(distance_m=radius, temperature_at_zero=at_zero, temperature_at_distance=at_radius,
                delta_temperature=at_radius - at_zero if at_radius is not None else None,
                coefficients=polynomial.c.tolist(), legacy_grid=legacy_grid,
                status='turning_point_found' if radius is not None else 'no_turning_point_in_observed_range')


def extract_buffer_temperatures(lst_path, source_path, *, step_m=50, max_distance_m=1000,
                                rings=True, class_field=None, buffer_crs=None):
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask
    from shapely.geometry import mapping
    from shapely.ops import transform
    from pyproj import Transformer
    from .tdi import require_metre_crs

    if step_m <= 0 or max_distance_m < step_m:
        raise ValueError('Invalid buffer distances.')
    frame = gpd.read_file(source_path)
    if frame.crs is None:
        raise ValueError('Sources must declare a CRS.')
    chosen_crs = buffer_crs or frame.estimate_utm_crs()
    if chosen_crs is None:
        raise ValueError('Could not determine a metre CRS; supply buffer_crs.')
    chosen_crs = require_metre_crs(chosen_crs)
    frame = frame.to_crs(chosen_crs)
    if class_field is not None and class_field not in frame.columns:
        raise ValueError(f'Missing source class field: {class_field}')
    records = []
    with rasterio.open(lst_path) as src:
        converter = Transformer.from_crs(chosen_crs, src.crs, always_xy=True)
        for source_id, row in frame.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue
            for distance in np.arange(step_m, max_distance_m + 1e-7, step_m):
                geometry = row.geometry.buffer(float(distance))
                if rings:
                    geometry = geometry.difference(row.geometry.buffer(max(0, float(distance - step_m))))
                geometry = transform(converter.transform, geometry)
                try:
                    data, _ = mask(src, [mapping(geometry)], crop=True, filled=False, indexes=1)
                    values = data.compressed()
                    values = values[np.isfinite(values) & (values > INVALID_THRESHOLD)]
                except ValueError as exc:
                    if 'do not overlap' not in str(exc):
                        raise
                    values = np.empty(0)
                records.append(dict(source_id=str(source_id),
                                    source_class=str(row[class_field]) if class_field else '',
                                    distance_m=float(distance),
                                    mean_lst=float(values.mean()) if len(values) else np.nan,
                                    n=int(len(values)), buffer_type='ring' if rings else 'cumulative'))
    return records
