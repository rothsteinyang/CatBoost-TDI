"""Numerical and geospatial regression checks for reconstruction and thermal indices."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine

from CatBoost_TDI.common.config import KEY_FEATURES, ModelConfig
from CatBoost_TDI.common.raster import Scene, align_band, read_scene
from CatBoost_TDI.evaluation.selection import (
    average_relative_importance, shap_importance, tdi_ablation,
)
from CatBoost_TDI.evaluation.validation import cross_validate, validate_cloud_mask
from CatBoost_TDI.reconstruction.model import fit_holdout, predict
from CatBoost_TDI.reconstruction.pipeline import fill_gaps
from CatBoost_TDI.thermal.distance import fit_diffusion_distance
from CatBoost_TDI.thermal.tdi import (
    ExponentialVariogram, aggregate_weights, build_tdi_raster,
    ordinary_kriging, temperature_weights,
)

FAST = ModelConfig(iterations=12, depth=3, thread_count=2)


def synthetic_scene():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(16, 18, len(KEY_FEATURES))).astype('float32')
    lst = (25 + 2 * features[:, :, 0] + 0.3 * features[:, :, -1]).astype('float32')
    profile = dict(height=16, width=18, transform=Affine(10, 0, 500000, 0, -10, 2500000), crs='EPSG:32649')
    return Scene(lst, features, KEY_FEATURES, profile, np.ones(lst.shape, dtype=bool))


class ReconstructionTests(unittest.TestCase):
    def test_native_categories_opt_in(self):
        scene = synthetic_scene()
        for name in ('EULUC', 'LCZ'):
            i = scene.feature_names.index(name)
            scene.features[:, :, i] = np.arange(scene.lst.size).reshape(scene.lst.shape) % 3 + 1
        config = ModelConfig(iterations=8, depth=3, thread_count=2, categorical_features=('EULUC', 'LCZ'))
        X, y, _ = scene.samples()
        fit = fit_holdout(X, y, scene.feature_names, config)
        self.assertTrue(np.isfinite(predict(fit.model, X[:10], scene.feature_names, config)).all())
        X = X.copy()
        X[0, scene.feature_names.index('LCZ')] = 1.5
        with self.assertRaises(ValueError):
            predict(fit.model, X[:10], scene.feature_names, config)

    def test_gap_only_prediction_and_boundary(self):
        scene = synthetic_scene()
        scene.lst[2, 3] = np.nan
        scene.lst[2, 4] = -3.402823e38
        scene.features[2, 4, 0] = np.inf
        scene.features[3, 4, 0] = np.nan
        scene.domain[1, 1] = False
        scene.lst = scene.lst.astype('float64')
        scene.lst[3, 4] = np.nextafter(25., 26.)
        original = scene.lst.copy()
        with patch('CatBoost_TDI.reconstruction.pipeline.predict', return_value=np.array([31.0])) as prediction:
            filled, qa, stats = fill_gaps(scene, object(), FAST)
        prediction.assert_called_once()
        self.assertEqual(filled[2, 3], 31)
        self.assertTrue(np.isnan(filled[2, 4]))
        self.assertTrue(np.isnan(filled[1, 1]))
        self.assertEqual(qa[3, 4], 1)
        self.assertEqual(qa[2, 4], 3)
        self.assertEqual(stats['filled_pixels'], 1)
        np.testing.assert_array_equal(filled[scene.observed], original[scene.observed])

    def test_outer_validation_and_feature_visibility(self):
        scene = synthetic_scene()
        visibility = []

        def rebuild(mask):
            visibility.append(mask.copy())
            features = scene.features.copy()
            features[:, :, -1] = float(scene.lst[mask].mean())
            return features

        result = cross_validate(scene, n_splits=3, config=FAST, feature_builder=rebuild)
        self.assertEqual(result['pooled']['n'], scene.lst.size)
        self.assertEqual(len(visibility), 3)
        np.testing.assert_array_equal(np.sum(visibility, axis=0), np.full(scene.lst.shape, 2))
        self.assertTrue(result['tdi_rebuild_callback_used'])
        spatial = cross_validate(scene, mode='spatial', n_splits=3, config=FAST)
        self.assertEqual(len(spatial['folds']), 3)
        mask = np.zeros(scene.lst.shape)
        mask[:, 9:] = 1
        mask[0, :] = np.nan
        cloud = validate_cloud_mask(scene, mask, config=FAST, feature_builder=rebuild)
        self.assertEqual(cloud['cloud_scores']['n'], 15 * 9)
        self.assertFalse(visibility[-1][0].any())
        self.assertFalse(visibility[-1][:, 9:].any())

    def test_selection_common_samples_and_relative_importance(self):
        scene = synthetic_scene()
        X, y, _ = scene.samples()
        fit = fit_holdout(X, y, scene.feature_names, FAST)
        importance = shap_importance(fit.model, X, scene.feature_names, FAST, max_samples=30)
        self.assertAlmostEqual(sum(importance['relative'].values()), 1)
        average = average_relative_importance([importance, importance])
        for name in scene.feature_names:
            self.assertAlmostEqual(average[name], importance['relative'][name])
        comparison = tdi_ablation(scene, FAST)
        self.assertEqual(comparison['with_tdi']['n'], comparison['without_tdi']['n'])

class ThermalIndexTests(unittest.TestCase):
    def test_equations_and_kriging_constraints(self):
        lc = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]])
        lst = np.where(lc == 1, 20., np.where(lc == 2, 40., 30.))
        _, weights, image = temperature_weights(lc, lst, classes=(1, 2, 3))
        self.assertEqual(weights, {1: -1, 2: 1, 3: 0})
        aggregated, transform = aggregate_weights(image, Affine(10, 0, 0, 0, -10, 40), 20)
        np.testing.assert_allclose(aggregated, [[-1, 1], [0, 0]])
        self.assertEqual(abs(transform.e), 20)
        points = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.]])
        targets = np.array([[5., 5.], [0., 0.], [100., 100.]])
        result = ordinary_kriging(points, np.full(4, 7.), targets, 20, ExponentialVariogram(10))
        np.testing.assert_allclose(result[:2], 7, atol=1e-10)
        self.assertTrue(np.isnan(result[2]))
        with self.assertRaises(ValueError):
            aggregate_weights(image, Affine.scale(11, -11), 350)
        with self.assertRaises(ValueError):
            temperature_weights(lc, lst, classes=(1, 2, 3), visible_mask=lc != 2)

    def test_cubic_turning_point_and_monotone_curve(self):
        distances = np.arange(50, 1001, 50)
        temperatures = (distances - 350.) ** 2 + 1e-5 * (distances - 350.) ** 3
        self.assertAlmostEqual(fit_diffusion_distance(distances, temperatures)['distance_m'], 350, places=6)
        legacy = fit_diffusion_distance(distances, temperatures, legacy_grid=True)
        self.assertLess(abs(legacy['distance_m'] - 350), 2)
        self.assertIsNone(fit_diffusion_distance(distances, distances ** 3)['distance_m'])

    def test_raster_tdi_alignment_and_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lc = np.repeat(np.repeat(np.array([[1, 2], [3, 4]]), 4, axis=0), 4, axis=1)
            profile = dict(driver='GTiff', height=8, width=8, count=1, dtype='float32',
                           crs='EPSG:32649', transform=Affine(10, 0, 500000, 0, -10, 2500000))
            with rasterio.open(tmp / 'lc.tif', 'w', **profile) as dst:
                dst.write(lc.astype('float32'), 1)
            profile.update(height=4, width=4, transform=profile['transform'] * Affine.scale(2))
            with rasterio.open(tmp / 'lst.tif', 'w', **profile) as dst:
                dst.write(np.repeat(np.repeat(np.array([[20., 30.], [40., 50.]]), 2, 0), 2, 1).astype('float32'), 1)
            result = build_tdi_raster(tmp / 'lc.tif', tmp / 'lst.tif', tmp / 'tdi.tif',
                                      ExponentialVariogram(30), distance_m=40, classes=(1, 2, 3, 4))
            self.assertEqual(result['source_points'], 4)
            with rasterio.open(tmp / 'tdi.tif') as src:
                self.assertEqual(src.transform, profile['transform'])
                self.assertTrue(np.isfinite(src.read(1)).all())
            self.assertEqual(align_band(tmp / 'tdi.tif', profile).shape, (4, 4))
            with rasterio.open(tmp / 'lst.tif', 'r+') as dst:
                values = dst.read(1)
                values[0, 0] = 200
                dst.write(values, 1)
            visible = np.ones((4, 4), dtype='float32')
            visible[0, 0] = 0
            with rasterio.open(tmp / 'visible.tif', 'w', **profile) as dst:
                dst.write(visible, 1)
            hidden = build_tdi_raster(tmp / 'lc.tif', tmp / 'lst.tif', tmp / 'hidden_tdi.tif',
                                      ExponentialVariogram(30), distance_m=40, classes=(1, 2, 3, 4),
                                      visible_mask_path=tmp / 'visible.tif')
            self.assertEqual(hidden['class_means'][1], 20)
            stacked = np.ones((17, 4, 4), dtype='float32')
            stacked[0], stacked[-1] = 30, -99
            profile.update(count=17)
            with rasterio.open(tmp / 'stack.tif', 'w', **profile) as dst:
                dst.write(stacked)
            overridden = read_scene(tmp / 'stack.tif', tdi_path=tmp / 'hidden_tdi.tif')
            np.testing.assert_allclose(overridden.features[:, :, -1], align_band(tmp / 'hidden_tdi.tif', profile))
            with rasterio.open(tmp / 'lst.tif', 'r+') as dst:
                values = dst.read(1)
                values[0, 0] = 200
                dst.write(values, 1)
            visible = np.ones((4, 4), dtype='float32')
            visible[0, 0] = 0
            with rasterio.open(tmp / 'visible.tif', 'w', **profile) as dst:
                dst.write(visible, 1)
            hidden = build_tdi_raster(tmp / 'lc.tif', tmp / 'lst.tif', tmp / 'hidden_tdi.tif',
                                      ExponentialVariogram(30), distance_m=40, classes=(1, 2, 3, 4),
                                      visible_mask_path=tmp / 'visible.tif')
            self.assertEqual(hidden['class_means'][1], 20)


if __name__ == '__main__':
    unittest.main()
