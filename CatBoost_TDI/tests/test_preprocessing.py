"""Preprocessing checks with known physical values and a complete source-to-model run."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from affine import Affine

from CatBoost_TDI.common.config import ALL_FEATURES, LEGACY_BANDS, ModelConfig
from CatBoost_TDI.preprocessing.pipeline import load_manifest, prepare_manifest
from CatBoost_TDI.preprocessing.lst import clean_lst_array
from CatBoost_TDI.preprocessing.raster import (align_source, generate_spectral_layers,
                                                read_native, source_spec, spectral_indices)


class PreprocessingTests(unittest.TestCase):
    def test_temperature_units_cloud_unknown_qc_and_zero_dilation(self):
        values = np.full((7, 7), 303.15)
        values[0, 0], values[3, 3] = np.nan, 300.15
        cloud, qc, zenith = (np.zeros_like(values) for _ in range(3))
        cloud[6, 6], qc[1, 5], zenith[5, 1] = np.nan, 2, 31
        result, reasons, stats = clean_lst_array(values, dict(input_unit='K', output_unit='C',
            threshold=28, dilation_pixels=0, qc_reject_bits=2), cloud, qc, zenith)
        self.assertEqual(np.isfinite(result).sum(), 44)
        self.assertAlmostEqual(result[2, 2], 30)
        self.assertEqual(reasons[6, 6], 2)
        self.assertEqual(reasons[1, 5], 3)
        self.assertEqual(reasons[5, 1], 4)
        self.assertEqual(reasons[3, 3], 5)
        self.assertEqual(stats['dilated_additional'], 0)

    def test_dilation_square_and_cross_are_distinct(self):
        values = np.full((9, 9), 30.)
        values[4, 4] = 20
        base = dict(input_unit='C', output_unit='C', threshold=25, dilation_pixels=1)
        cross, _, _ = clean_lst_array(values, dict(base, dilation_shape='cross'))
        square, _, _ = clean_lst_array(values, dict(base, dilation_shape='square'))
        self.assertEqual(np.isnan(cross).sum(), 5)
        self.assertEqual(np.isnan(square).sum(), 9)

    def test_sentinel_mapping_metadata_offset_and_block_processing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transform = Affine(10, 0, 500000, 0, -10, 2500000)
            profile = dict(driver='GTiff', height=8, width=8, count=10, dtype='uint16',
                           crs='EPSG:32649', transform=transform, nodata=65535)
            source = root / 'ten_band.tif'
            with rasterio.open(source, 'w', **profile) as dst:
                cube = np.full((10, 8, 8), 1000, dtype='uint16')
                for index, value in ((0, 2000), (1, 3000), (2, 4000), (6, 7000), (8, 5000)):
                    cube[index] = value
                cube[:, 0, 0] = 65535
                dst.write(cube)
                dst.scales, dst.offsets = (0.0001,) * 10, (-0.1,) * 10
            specs = {name: dict(path=str(source), band=index, use_metadata_scale=True)
                     for name, index in (('B02', 1), ('B03', 2), ('B04', 3), ('B08', 7), ('B11', 9))}
            grid = dict(height=8, width=8, crs='EPSG:32649', transform=transform)
            products = generate_spectral_layers(specs, grid, root / 'indices', block_size=4)
            expected = dict(NDVI=1/3, NDWI=-0.5, NDBI=-0.2, Albedo=0.1797)
            for name, spec in products.items():
                with rasterio.open(spec['path']) as src:
                    a = src.read(1, masked=True)
                    self.assertTrue(a.mask[0, 0])
                    np.testing.assert_allclose(a.compressed(), expected[name], atol=2e-8)
            with self.assertRaises(ValueError):
                source_spec(dict(path=str(source), band=1.5))

    def test_nodata_is_removed_before_resampling_and_reads_are_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'large.tif'
            profile = dict(driver='GTiff', height=100, width=100, count=1, dtype='float64',
                           crs='EPSG:32649', transform=Affine(10, 0, 0, 0, -10, 1000))
            values = np.full((100, 100), 20.)
            values[40:50, 40:50] = -3.402823e38  # deliberately untagged sentinel
            with rasterio.open(source, 'w', **profile) as dst:
                dst.write(values, 1)
            grid = dict(height=10, width=10, crs='EPSG:32649', transform=Affine(20, 0, 350, 0, -20, 650))
            _, _, provenance = read_native(str(source), grid)
            self.assertLess(provenance['window'][2], 100)
            aligned, _ = align_source(str(source), grid, 'bilinear')
            np.testing.assert_allclose(aligned[np.isfinite(aligned)], 20.)

    def test_source_rasters_to_stack_to_gapless(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (16, 18)
            transform = Affine(70, 0, 500000, 0, -70, 2500000)
            profile = dict(driver='GTiff', height=shape[0], width=shape[1], crs='EPSG:32649',
                           transform=transform, dtype='float64')
            rows, cols = np.indices(shape)
            cube = np.stack([np.full(shape, 0.2 + i / 100) + cols / 100 for i in range(16)])
            for name in ('LC', 'EULUC', 'LCZ'):
                cube[ALL_FEATURES.index(name)] = (rows + cols) % 5 + 1
            feature_path = root / 'independent_features.tif'
            with rasterio.open(feature_path, 'w', count=16, **profile) as dst:
                dst.write(cube)
            lst = 303.15 + cols / 10 + rows / 20
            lst[3, 4] = 0
            lst_path = root / 'lst_kelvin.tif'
            with rasterio.open(lst_path, 'w', count=1, nodata=0, **profile) as dst:
                dst.write(lst, 1)
            before = hashlib.sha256(lst_path.read_bytes()).hexdigest()
            manifest = dict(schema_version=1, crop_to_roi=False, scenes=[dict(id='example',
                            lst='lst_kelvin.tif', lst_options=dict(input_unit='K', output_unit='C'),
                            features={name: dict(path='independent_features.tif', band=i+1)
                                      for i, name in enumerate(ALL_FEATURES)})])
            manifest['scenes'][0]['features']['NDVI'] = 'independent_features.tif'
            config_path = root / 'sources.json'
            config_path.write_text(json.dumps(manifest), encoding='utf-8')
            result = prepare_manifest(config_path, root / 'out', run_reconstruction=True,
                                       config=ModelConfig(iterations=12, depth=3, thread_count=2))
            with rasterio.open(result['scenes'][0]['preprocessing']['stack']) as src:
                self.assertEqual(src.count, 17)
                self.assertEqual(src.descriptions, LEGACY_BANDS)
                self.assertEqual(src.transform, transform)
                self.assertAlmostEqual(src.read(1)[2, 2], lst[2, 2] - 273.15)
            self.assertEqual(result['scenes'][0]['filling']['filled_pixels'], 1)
            self.assertEqual(before, hashlib.sha256(lst_path.read_bytes()).hexdigest())
            manifest['scenes'][0]['features'].pop('TDI')
            config_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_manifest(config_path)

    def test_overwrite_cannot_destroy_sources_and_failed_stack_is_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'out'
            source = output / 'stacks' / 'example.tif'
            source.parent.mkdir(parents=True)
            profile = dict(driver='GTiff', height=12, width=12, count=1, dtype='float64',
                           crs='EPSG:32649', transform=Affine(70, 0, 500000, 0, -70, 2500000))
            with rasterio.open(source, 'w', **profile) as dst:
                dst.write(np.full((12, 12), 30.), 1)
            manifest = dict(schema_version=1, crop_to_roi=False, scenes=[dict(id='example', lst=str(source),
                            lst_options=dict(input_unit='C', output_unit='C'),
                            features={name: str(source) for name in ALL_FEATURES})])
            config = root / 'manifest.json'
            config.write_text(json.dumps(manifest), encoding='utf-8')
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'overwrite an input'):
                prepare_manifest(config, output, overwrite=True)
            self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
            fractional = root / 'fractional_categories.tif'
            with rasterio.open(fractional, 'w', **profile) as dst:
                dst.write(np.full((12, 12), 1.5), 1)
            manifest['scenes'][0]['features']['LC'] = str(fractional)
            config.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'categorical values'):
                prepare_manifest(config, root / 'failed')
            self.assertFalse((root / 'failed' / 'stacks' / 'example.tif').exists())
            self.assertFalse(list((root / 'failed' / 'stacks').glob('.preparing-*')))


if __name__ == '__main__':
    unittest.main()
