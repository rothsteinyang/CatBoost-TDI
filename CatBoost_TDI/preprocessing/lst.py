"""LST scaling/units, supplied cloud or QC masks, thresholds and cloud buffers."""

import numpy as np
from scipy.ndimage import binary_dilation

from .raster import align_source, read_native, source_spec, warp_array


def clean_lst_array(values, options, cloud=None, quality=None, zenith=None):
    allowed = {'input_unit', 'output_unit', 'threshold', 'threshold_inclusive', 'dilation_pixels',
               'dilation_shape', 'dilate_original_nodata', 'cloud_values', 'clear_values',
               'qc_allowed_values', 'qc_reject_bits', 'max_zenith_deg', 'already_screened'}
    if set(options).difference(allowed):
        raise ValueError(f'Unknown LST options: {set(options).difference(allowed)}')
    result = np.asarray(values, dtype='float64').copy()
    source_unit, output_unit = options.get('input_unit'), options.get('output_unit')
    if source_unit not in ('C', 'K') or output_unit not in ('C', 'K'):
        raise ValueError('LST input_unit and output_unit must explicitly be C or K.')
    if source_unit != output_unit:
        result += 273.15 if output_unit == 'K' else -273.15
    initial_invalid = ~np.isfinite(result)
    reasons = np.zeros(result.shape, dtype='uint8')
    reasons[initial_invalid] = 1
    seeds = np.zeros(result.shape, dtype=bool)
    counts = dict(original_invalid=int(initial_invalid.sum()))
    for arr in (cloud, quality, zenith):
        if arr is not None and arr.shape != result.shape:
            raise ValueError('All LST masks must be on the native LST grid.')
    if cloud is not None:
        cloud_values = options.get('cloud_values', [1])
        clear_values = options.get('clear_values', [0])
        if set(cloud_values).intersection(clear_values):
            raise ValueError('Clear/cloud codes cannot overlap.')
        known = np.isin(cloud, [*cloud_values, *clear_values])
        bad = ~known | np.isin(cloud, cloud_values)
        seeds |= bad
        reasons[bad & ~initial_invalid] = 2
        counts['official_cloud_or_unknown'] = int(bad.sum())
    if quality is not None:
        if options.get('qc_allowed_values') is not None and options.get('qc_reject_bits') is not None:
            raise ValueError('Choose QC allowed values or reject bits, not both.')
        valid = np.isfinite(quality)
        if not np.all(quality[valid] == np.floor(quality[valid])) or (quality[valid] < 0).any():
            raise ValueError('QC codes must be nonnegative integers.')
        if options.get('qc_allowed_values') is not None:
            bad = ~valid | ~np.isin(quality, options['qc_allowed_values'])
        elif options.get('qc_reject_bits') is not None:
            bits = int(options['qc_reject_bits'])
            if bits < 0:
                raise ValueError('QC reject bitmask must be nonnegative.')
            bad = ~valid | ((np.where(valid, quality, 0).astype('uint64') & bits) != 0)
        else:
            raise ValueError('QC raster supplied without an explicit code/bit rule.')
        seeds |= bad
        reasons[bad & ~initial_invalid] = 3
        counts['quality_rejected'] = int(bad.sum())
    if zenith is not None:
        maximum = float(options.get('max_zenith_deg', 30.0))
        if not 0 <= maximum <= 90:
            raise ValueError('max_zenith_deg must be in [0, 90].')
        bad = ~np.isfinite(zenith) | (np.abs(zenith) > maximum)
        reasons[bad & ~initial_invalid] = 4
        result[bad] = np.nan
        counts['zenith_rejected'] = int(bad.sum())
    threshold = options.get('threshold')
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError('LST threshold must be finite and in output temperature units.')
        bad = result <= threshold if options.get('threshold_inclusive', False) else result < threshold
        seeds |= bad
        reasons[bad & ~initial_invalid] = 5
        counts['below_threshold'] = int(bad.sum())
    if options.get('dilate_original_nodata', False):
        seeds |= initial_invalid
    iterations = int(options.get('dilation_pixels', 0))
    if iterations < 0 or iterations != options.get('dilation_pixels', 0):
        raise ValueError('dilation_pixels must be a nonnegative integer.')
    shape = options.get('dilation_shape', 'cross')
    if shape not in ('cross', 'square'):
        raise ValueError('dilation_shape must be cross or square.')
    # scipy iterations=0 means dilate until stable, NOT disable dilation.
    expanded = binary_dilation(seeds, structure=np.ones((3, 3)) if shape == 'square' else None,
                               iterations=iterations) if iterations else seeds
    reasons[expanded & ~seeds & ~initial_invalid] = 6
    result[expanded | initial_invalid] = np.nan
    counts.update(dilated_additional=int((expanded & ~seeds).sum()), valid_after=int(np.isfinite(result).sum()),
                  temperature_unit=output_unit, threshold=threshold, threshold_inclusive=options.get('threshold_inclusive', False),
                  dilation_pixels=iterations, dilation_shape=shape,
                  cloud_mask_applied=cloud is not None, quality_rule_applied=quality is not None,
                  zenith_rule_applied=zenith is not None, upstream_screening_claimed=bool(options.get('already_screened', False)))
    return result, reasons, counts


def prepare_lst(source, target_profile, options, *, cloud_source=None, quality_source=None, zenith_source=None):
    padding = max(3, int(options.get('dilation_pixels', 0)) + 3)
    values, profile, provenance = read_native(source, target_profile, padding=padding)
    masks = {}
    for key, spec in (('cloud', cloud_source), ('quality', quality_source), ('zenith', zenith_source)):
        if spec is not None:
            if key in ('cloud', 'quality') and source_spec(spec).get('resampling', 'nearest') != 'nearest':
                raise ValueError('Cloud and QC rasters require nearest-neighbour resampling.')
            # Keep all unknown pixels as NaN; never initialize outside coverage to clear.
            masks[key] = align_source(spec, profile, 'nearest')[0]
    cleaned, reasons, stats = clean_lst_array(values, options, **masks)
    return (warp_array(cleaned, profile, target_profile),
            warp_array(reasons.astype(float), profile, target_profile),
            dict(source=provenance, processing=stats,
                 masks={k: str(v['path'] if isinstance(v, dict) else v) for k, v in (
                     ('cloud', cloud_source), ('quality', quality_source), ('zenith', zenith_source)) if v is not None}))
