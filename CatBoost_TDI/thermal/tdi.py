"""Paper equations (1)-(3), reconstructed in Python; NOT recovered ArcMap code.

The paper specifies an exponential semivariogram but not its fitted parameters
or ArcMap grid origin. A variogram scale must therefore be provided explicitly.
Use the original TDI band when reproducing the published notebook experiments.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, transform as transform_coords
from scipy.spatial import cKDTree, distance_matrix

from ..common.config import INVALID_THRESHOLD
from ..common.raster import align_band, write_raster


def temperature_weights(land_cover, lst, classes=(1, 2, 3, 4, 5), visible_mask=None):
    """Equation (1); LST and LC must be on the same grid.

    visible_mask restricts observations used to calculate class means. Validation
    callers must exclude held-out temperatures before calculating these means.
    Missing classes raise an error instead of inventing their temperature weight.
    """
    land_cover, lst = np.asarray(land_cover), np.asarray(lst)
    if land_cover.ndim != 2 or land_cover.shape != lst.shape:
        raise ValueError('LC and LST must have the same two-dimensional shape.')
    if not classes or len(set(classes)) != len(classes):
        raise ValueError('classes must be nonempty and unique.')
    valid = np.isfinite(lst) & (lst > INVALID_THRESHOLD)
    if visible_mask is not None:
        if visible_mask.shape != lst.shape or visible_mask.dtype != np.bool_:
            raise ValueError('visible_mask must be boolean and match LST.')
        valid &= visible_mask
    means = {}
    for code in classes:
        selected = valid & (land_cover == code)
        if not selected.any():
            raise ValueError(f'LC class {code} has no visible LST observations.')
        means[int(code)] = float(lst[selected].mean(dtype=np.float64))
    low, high = min(means.values()), max(means.values())
    # Equal class means carry no hot/cold contrast; avoid division by zero.
    weights = {code: 0.0 if high == low else 2 * (value - low) / (high - low) - 1
               for code, value in means.items()}
    image = np.full(lst.shape, np.nan, dtype='float64')
    for code, value in weights.items():
        image[land_cover == code] = value
    return means, weights, image


def aggregate_weights(weight_map, transform, distance_m=350.0):
    """Equation (2): non-overlapping R-by-R square cells anchored at the LC origin.

    The LC grid must be north-up with metre units, checked by build_tdi_raster.
    Require an integer aggregation factor so the physical R is never rounded.
    Partial edge cells average their available valid pixels.
    """
    if distance_m <= 0 or transform.a <= 0 or transform.e >= 0 or transform.b or transform.d:
        raise ValueError('Aggregation requires positive R and a north-up raster.')
    factors = (distance_m / abs(transform.e), distance_m / transform.a)
    if not all(np.isclose(f, round(f), rtol=0, atol=1e-7) and f >= 1 for f in factors):
        raise ValueError('R must be an integer multiple of both LC pixel dimensions.')
    fy, fx = (int(round(f)) for f in factors)
    if weight_map.ndim != 2:
        raise ValueError('weight_map must be two-dimensional.')
    h, w = weight_map.shape
    oh, ow = (h + fy - 1) // fy, (w + fx - 1) // fx
    padded = np.full((oh * fy, ow * fx), np.nan)
    padded[:h, :w] = weight_map
    blocks = padded.reshape(oh, fy, ow, fx)
    counts = np.isfinite(blocks).sum(axis=(1, 3))
    sums = np.where(np.isfinite(blocks), blocks, 0).sum(axis=(1, 3))
    aggregated = np.divide(sums, counts, out=np.full((oh, ow), np.nan), where=counts > 0)
    return aggregated, transform * Affine.scale(fx, fy)


@dataclass(frozen=True)
class ExponentialVariogram:
    # gamma(h>0) = nugget + partial_sill * (1-exp(-h/scale_m)).
    # scale_m is the e-folding scale; the practical 95% range is ~3*scale_m.
    scale_m: float
    partial_sill: float = 1.0
    nugget: float = 0.0

    def __post_init__(self):
        values = (self.scale_m, self.partial_sill, self.nugget)
        if not np.isfinite(values).all() or self.scale_m <= 0 or self.partial_sill <= 0 or self.nugget < 0:
            raise ValueError('Variogram scale and partial sill must be positive; nugget must be nonnegative.')


def ordinary_kriging(points, values, targets, radius_m, variogram, chunk_size=4096):
    """Equation (3), using every source point within sqrt(2)*R by default upstream.

    No IDW fallback, nearest-point cap, or clipping is applied. Locations without
    a source in the radius remain NaN. Coordinates must already be in metres.
    """
    points, values, targets = (np.asarray(a, dtype=float) for a in (points, values, targets))
    if points.ndim != 2 or targets.ndim != 2 or points.shape[1] != 2 or targets.shape[1] != 2:
        raise ValueError('Point coordinates must have shape (n, 2).')
    if values.shape != (len(points),) or len(points) == 0:
        raise ValueError('One value per nonempty source point is required.')
    if not all(np.isfinite(a).all() for a in (points, values, targets)):
        raise ValueError('Kriging inputs must be finite.')
    if radius_m <= 0 or chunk_size < 1 or len(np.unique(points, axis=0)) != len(points):
        raise ValueError('Require a positive radius/chunk size and unique source coordinates.')
    tree = cKDTree(points)
    result = np.full(len(targets), np.nan)
    # A fixed, local neighbour set recurs on regular grids; reuse its inverse.
    # Bound the cache for arbitrary point sets.
    cache = {}
    for start in range(0, len(targets), chunk_size):
        batch = targets[start:start + chunk_size]
        neighbours = tree.query_ball_point(batch, radius_m)
        for j, (target, neighbours_j) in enumerate(zip(batch, neighbours)):
            if not neighbours_j:
                continue
            ids = tuple(sorted(neighbours_j))
            local = points[list(ids)]
            dist = np.linalg.norm(local - target, axis=1)
            coincident = np.flatnonzero(dist < 1e-8)
            if len(coincident):
                result[start + j] = values[ids[coincident[0]]]
                continue
            if ids not in cache:
                n = len(ids)
                cov = variogram.partial_sill * np.exp(-distance_matrix(local, local) / variogram.scale_m)
                cov.flat[::n + 1] += variogram.nugget
                system = np.ones((n + 1, n + 1))
                system[:n, :n] = cov
                system[-1, -1] = 0
                if len(cache) >= 4096:
                    cache.clear()
                cache[ids] = np.linalg.inv(system)
            rhs = np.r_[variogram.partial_sill * np.exp(-dist / variogram.scale_m), 1.0]
            weights = (cache[ids] @ rhs)[:-1]
            result[start + j] = weights @ values[list(ids)]
    return result


def require_metre_crs(crs):
    from pyproj import CRS
    crs = CRS.from_user_input(crs)
    if not crs.is_projected or not crs.axis_info or any(
            not np.isclose(a.unit_conversion_factor, 1) for a in crs.axis_info[:2]):
        raise ValueError('LC/buffer CRS must be projected in metres; reproject it first.')
    return crs


def build_tdi_raster(lc_path, lst_path, output_path, variogram, distance_m=350.0,
                     classes=(1, 2, 3, 4, 5), visible_mask_path=None, row_chunk=128):
    """Rebuild TDI on the LST grid from a high-resolution LC grid in metres.

    LST is mapped to the LC grid by nearest neighbour to calculate class means.
    A visibility raster uses 1=visible, 0=held out; unknown pixels are excluded.
    The original ArcMap grid alignment/variogram is not recoverable from PDF.
    """
    if row_chunk < 1:
        raise ValueError('row_chunk must be positive.')
    with rasterio.open(lc_path) as src:
        lc = src.read(1, masked=True).astype('float64').filled(np.nan)
        lc_profile = dict(src.profile)
        lc_crs = require_metre_crs(src.crs)
    # Apply visibility BEFORE resampling LST, so held-out labels cannot enter WTem.
    with rasterio.open(lst_path) as src:
        lst_profile = dict(src.profile)
        lst = src.read(1, masked=True).astype('float64').filled(np.nan)
    if visible_mask_path is not None:
        visible = align_band(visible_mask_path, lst_profile) == 1
        lst[~visible] = np.nan
    lst[~np.isfinite(lst) | (lst <= INVALID_THRESHOLD)] = np.nan
    from rasterio.warp import reproject
    aligned_lst = np.full(lc.shape, np.nan)
    reproject(lst, aligned_lst, src_transform=lst_profile['transform'], src_crs=lst_profile['crs'],
              src_nodata=np.nan, dst_transform=lc_profile['transform'], dst_crs=lc_crs,
              dst_nodata=np.nan, resampling=Resampling.nearest)
    means, weights, weight_map = temperature_weights(lc, aligned_lst, classes)
    aggregated, agg_transform = aggregate_weights(weight_map, lc_profile['transform'], distance_m)
    rr, cc = np.where(np.isfinite(aggregated))
    if len(rr) == 0:
        raise ValueError('No valid thermal aggregation cells.')
    xx, yy = agg_transform * (cc + 0.5, rr + 0.5)
    points = np.column_stack((xx, yy))
    out = np.full((lst_profile['height'], lst_profile['width']), np.nan, dtype='float32')
    for r0 in range(0, len(out), row_chunk):
        rows, cols = np.indices((min(row_chunk, len(out) - r0), out.shape[1]))
        tx, ty = lst_profile['transform'] * (cols.ravel() + 0.5, rows.ravel() + r0 + 0.5)
        tx, ty = transform_coords(lst_profile['crs'], lc_crs, tx, ty)
        out[r0:r0 + rows.shape[0]] = ordinary_kriging(
            points, aggregated[rr, cc], np.column_stack((tx, ty)),
            np.sqrt(2) * distance_m, variogram).reshape(rows.shape)
    write_raster(output_path, out, lst_profile,
                 tags=dict(method='Python reconstruction of paper equations 1-3', distance_m=distance_m))
    return dict(implementation='paper_equations_python_not_original_arcmap',
                class_means=means, temperature_weights=weights, distance_m=distance_m,
                search_radius_m=float(np.sqrt(2) * distance_m), variogram=asdict(variogram),
                aggregation_origin=[agg_transform.c, agg_transform.f], lc_crs=lc_crs.to_string(),
                visible_mask=str(visible_mask_path) if visible_mask_path else None,
                source_points=len(points), output=str(Path(output_path).resolve()))
