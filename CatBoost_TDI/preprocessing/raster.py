"""Bounded raster reads, explicit calibration, ROI grids and spectral features.

No files are discovered by taking the first glob result. Source paths, band
numbers, calibration and resampling are supplied by a reviewed manifest.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds


RESAMPLING = {name: getattr(Resampling, name) for name in ('nearest', 'bilinear', 'cubic', 'average')}
PAPER_RESAMPLING = dict.fromkeys(('NDVI', 'NDWI', 'NDBI', 'Albedo', 'Emissivity', 'DEM', 'TH'), 'bilinear')
PAPER_RESAMPLING.update(dict.fromkeys(('LC', 'EULUC', 'LCZ', 'BH', 'BD', 'SVF', 'TDI'), 'nearest'))
PAPER_RESAMPLING.update(AH='cubic', VNL='cubic', LST='nearest')


def source_spec(value):
    spec = {'path': str(value)} if isinstance(value, (str, Path)) else dict(value)
    allowed = {'path', 'band', 'scale', 'offset', 'use_metadata_scale', 'nodata',
               'raw_minimum', 'raw_maximum', 'resampling'}
    if not spec.get('path') or set(spec).difference(allowed):
        raise ValueError(f'Invalid raster source specification: {spec}')
    if spec.get('use_metadata_scale') and ('scale' in spec or 'offset' in spec):
        raise ValueError('Choose metadata calibration OR explicit scale/offset, not both.')
    if int(spec.get('band', 1)) < 1 or int(spec.get('band', 1)) != spec.get('band', 1):
        raise ValueError('Raster band numbers start at one.')
    if spec.get('resampling', 'nearest') not in RESAMPLING:
        raise ValueError('Unsupported resampling method.')
    return spec


def _window(src, target, padding):
    if src.transform.b or src.transform.d or src.transform.a <= 0 or src.transform.e >= 0:
        raise ValueError(f'{src.name}: rectify rotated or south-up source rasters first.')
    bounds = transform_bounds(target['crs'], src.crs,
                              *array_bounds(target['height'], target['width'], target['transform']),
                              densify_pts=21)
    requested = from_bounds(*bounds, transform=src.transform)
    c0 = max(0, int(np.floor(requested.col_off)) - padding)
    r0 = max(0, int(np.floor(requested.row_off)) - padding)
    c1 = min(src.width, int(np.ceil(requested.col_off + requested.width)) + padding)
    r1 = min(src.height, int(np.ceil(requested.row_off + requested.height)) + padding)
    if c1 <= c0 or r1 <= r0:
        raise ValueError(f'Raster does not overlap the reference grid: {src.name}')
    return Window(c0, r0, c1 - c0, r1 - r0)


def read_native(value, target=None, padding=3):
    """Read only a footprint window; mask raw fill values BEFORE calibration/warp."""
    spec = source_spec(value)
    with rasterio.open(spec['path']) as src:
        band = int(spec.get('band', 1))
        if band > src.count or src.crs is None:
            raise ValueError(f'{src.name}: band absent or CRS undefined.')
        window = _window(src, target, padding) if target else Window(0, 0, src.width, src.height)
        data = src.read(band, window=window, masked=True).astype('float64').filled(np.nan)
        if spec.get('nodata') is not None:
            data[data == spec['nodata']] = np.nan
        data[~np.isfinite(data)] = np.nan
        # Legacy reconstruction excluded <= -200, including untagged float minima.
        minimum = spec.get('raw_minimum', -200.0)
        if minimum is not None:
            data[data <= minimum] = np.nan
        if spec.get('raw_maximum') is not None:
            data[data > spec['raw_maximum']] = np.nan
        scale = src.scales[band - 1] if spec.get('use_metadata_scale') else spec.get('scale', 1.0)
        offset = src.offsets[band - 1] if spec.get('use_metadata_scale') else spec.get('offset', 0.0)
        if not np.isfinite((scale, offset)).all():
            raise ValueError('Raster calibration must be finite.')
        data = data * scale + offset
        profile = dict(height=data.shape[0], width=data.shape[1], crs=src.crs,
                       transform=src.window_transform(window))
        provenance = dict(path=str(Path(spec['path']).resolve()), band=band,
                          scale=float(scale), offset=float(offset), nodata=spec.get('nodata', src.nodata),
                          raw_minimum=minimum, window=[int(window.col_off), int(window.row_off),
                                                       int(window.width), int(window.height)],
                          size_bytes=Path(spec['path']).stat().st_size)
    return data, profile, provenance


def warp_array(data, source_profile, target_profile, method='nearest'):
    out = np.full((target_profile['height'], target_profile['width']), np.nan, dtype='float64')
    reproject(data, out, src_transform=source_profile['transform'], src_crs=source_profile['crs'],
              src_nodata=np.nan, dst_transform=target_profile['transform'], dst_crs=target_profile['crs'],
              dst_nodata=np.nan, resampling=RESAMPLING[method], num_threads=2)
    return out


def align_source(value, target_profile, method='nearest'):
    spec = source_spec(value)
    method = spec.get('resampling', method)
    data, profile, provenance = read_native(spec, target_profile)
    provenance['resampling'] = method
    return warp_array(data, profile, target_profile, method), provenance


def reference_grid(path, roi_path=None, crop=True):
    import geopandas as gpd
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError('Reference grid requires a CRS.')
        profile = dict(height=src.height, width=src.width, crs=src.crs, transform=src.transform)
        if roi_path and crop:
            roi = gpd.read_file(roi_path)
            if roi.empty or roi.crs is None:
                raise ValueError('ROI is empty or lacks a CRS.')
            bounds = roi.to_crs(src.crs).total_bounds
            request = from_bounds(*bounds, transform=src.transform)
            c0, r0 = max(0, int(np.floor(request.col_off))), max(0, int(np.floor(request.row_off)))
            c1 = min(src.width, int(np.ceil(request.col_off + request.width)))
            r1 = min(src.height, int(np.ceil(request.row_off + request.height)))
            if c0 >= c1 or r0 >= r1:
                raise ValueError('ROI and reference raster do not overlap.')
            window = Window(c0, r0, c1 - c0, r1 - r0)
            profile.update(height=r1 - r0, width=c1 - c0, transform=src.window_transform(window))
    return profile


def spectral_indices(reflectance):
    """Local ENVI .exp / notebook equations; inputs are calibrated reflectance."""
    required = ('B02', 'B03', 'B04', 'B08', 'B11')
    if set(required).difference(reflectance):
        raise ValueError(f'Spectral calculation requires {required}.')
    arrays = [np.asarray(reflectance[n], dtype='float64') for n in required]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('Spectral bands must be aligned before ratio calculation.')
    b2, b3, b4, b8, b11 = arrays

    def ratio(a, b):
        denominator = a + b
        return np.divide(a - b, denominator, out=np.full(a.shape, np.nan),
                         where=np.isfinite(a) & np.isfinite(b) & (denominator != 0))

    return dict(NDVI=ratio(b8, b4), NDWI=ratio(b3, b8), NDBI=ratio(b11, b8),
                Albedo=0.443 * b2 + 0.317 * b3 + 0.240 * b4)


def generate_spectral_layers(bands, target_profile, output_dir, block_size=256):
    """Compute on the B02 native grid in bounded tiles; do not ratio coarse averages."""
    from contextlib import ExitStack
    if block_size < 1:
        raise ValueError('block_size must be positive.')
    specs = {n: source_spec(s) for n, s in bands.items()}
    if set(('B02', 'B03', 'B04', 'B08', 'B11')).difference(specs):
        raise ValueError('Explicit B02/B03/B04/B08/B11 sources are required.')
    # Determine the local B02 grid without reading the full native raster.
    with rasterio.open(specs['B02']['path']) as src:
        window = _window(src, target_profile, 3)
        native = dict(height=int(window.height), width=int(window.width), crs=src.crs,
                      transform=src.window_transform(window))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {n: output_dir / f'{n}.tif' for n in ('NDVI', 'NDWI', 'NDBI', 'Albedo')}
    with ExitStack() as stack:
        writers = {n: stack.enter_context(rasterio.open(p, 'w', driver='GTiff', count=1,
                   dtype='float32', nodata=-9999., compress='deflate', **native)) for n, p in paths.items()}
        for r0 in range(0, native['height'], block_size):
            for c0 in range(0, native['width'], block_size):
                window = Window(c0, r0, min(block_size, native['width'] - c0),
                                min(block_size, native['height'] - r0))
                block = dict(native, height=int(window.height), width=int(window.width),
                             transform=rasterio.windows.transform(window, native['transform']))
                reflectance = {n: align_source(specs[n], block, 'bilinear')[0]
                               for n in ('B02', 'B03', 'B04', 'B08', 'B11')}
                for n, values in spectral_indices(reflectance).items():
                    writers[n].write(np.where(np.isfinite(values), values, -9999.).astype('float32'), 1, window=window)
        for writer in writers.values():
            writer.update_tags(method='local ENVI expressions', calibration='explicit source manifest')
    return {n: {'path': str(p), 'resampling': 'bilinear'} for n, p in paths.items()}
