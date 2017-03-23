"""Test suite for PredictorStacker class
The suite is based on unittest and can be run from project root with
prompt$python -m unittest -v test.test_stacker
Coverage measures can be performed with
prompt$coverage run -m unittest -v test.test_stacker
prompt$coverage html
"""

import unittest
from stacker import PredictorStacker
import numpy as np


def metric_rmse(truth, hat):
    return np.mean(np.power(truth - hat, 2)) ** .5


def metric_mae(truth, hat):
    return np.mean(np.abs(truth - hat))


class TestPredictorStacker(unittest.TestCase):
    def test_init_stacker(self):
        stacker = PredictorStacker(metric=metric_rmse)
        self.assertEqual(stacker.metric, metric_rmse)
        self.assertEqual(stacker.predictors, None)
        self.assertEqual(stacker.target, None)
        self.assertEqual(stacker.weights, None)
        self.assertEqual(stacker.score, None)

    def test_fit_without_metric_raise_ValueError(self):
        stacker = PredictorStacker()
        self.assertRaises(ValueError, stacker.fit)

    def test_fit_without_predictors_raise_ValueError(self):
        stacker = PredictorStacker(metric=metric_rmse)
        self.assertRaises(ValueError, stacker.fit)

    def test_add_predictors_one_file(self):
        stacker = PredictorStacker()
        stacker.add_predictors(files=['noid_OOF_predictions_2.csv'])
        self.assertEqual(len(stacker.target), 188318)
        self.assertEqual(len(stacker.predictors), 188318)
        self.assertEqual(stacker.predictors.shape[1], 20)

    def test_add_predictors_two_files(self):
        stacker = PredictorStacker()
        stacker.add_predictors(files=['noid_OOF_predictions_2.csv',
                                      'noid_OOF_predictions_3.csv'])
        self.assertEqual(len(stacker.target), 188318)
        self.assertEqual(len(stacker.predictors), 188318)
        self.assertEqual(stacker.predictors.shape[1], 21)

    def test_add_predictors_file_error_raise_ValueError(self):
        stacker = PredictorStacker()
        self.assertRaises(ValueError, stacker.add_predictors, files=['does_not_exist.csv'])

    def test_add_predictors_not_a_string_raise_TypeError(self):
        stacker = PredictorStacker()
        self.assertRaises(TypeError, stacker.add_predictors, files=[1, 2])

    def test_add_predictors_target_not_in_sync_raise_ValueError(self):
        stacker = PredictorStacker()
        self.assertRaises(ValueError,
                          stacker.add_predictors,
                          files=['noid_OOF_predictions_1.csv',
                                 'noid_OOF_predictions_2.csv'])

    def test_fit_stacker_one_bag(self):
        stacker = PredictorStacker(metric=metric_mae)
        stacker.add_predictors(files=['noid_OOF_predictions_2.csv',
                                      'noid_OOF_predictions_3.csv',
                                      'noid_OOF_predictions_4.csv'])
        self.assertEqual(len(stacker.target), 188318)
        self.assertEqual(len(stacker.predictors), 188318)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit(max_iter=20, n_bags=1, max_predictors=1., max_samples=1., verbose=2)
        self.assertAlmostEqual(stacker.score, 1136.4316755211291, places=4)
        self.assertAlmostEqual(stacker.mean_score, 1137.2460719022001, places=4)

    def test_fit_stacker_one_bag_rmse(self):
        stacker = PredictorStacker(metric=metric_rmse)
        stacker.add_predictors(files=['noid_OOF_predictions_2.csv',
                                      'noid_OOF_predictions_3.csv',
                                      'noid_OOF_predictions_4.csv'])
        self.assertEqual(len(stacker.target), 188318)
        self.assertEqual(len(stacker.predictors), 188318)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit(max_iter=20, n_bags=1, max_predictors=1., max_samples=1., verbose=2)
        self.assertAlmostEqual(stacker.score, 1940.1115566433118, places=4)
        self.assertAlmostEqual(stacker.mean_score, 1942.3928913, places=4)

    def test_fit_stacker_ten_bags(self):
        stacker = PredictorStacker(metric=metric_mae)
        stacker.add_predictors(files=['noid_OOF_predictions_2.csv',
                                      'noid_OOF_predictions_3.csv',
                                      'noid_OOF_predictions_4.csv'])
        self.assertEqual(len(stacker.target), 188318)
        self.assertEqual(len(stacker.predictors), 188318)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit(max_iter=10, n_bags=20, max_predictors=.8, max_samples=.8)
        self.assertAlmostEqual(stacker.score, 1136.9059858227927, places=4)
        self.assertAlmostEqual(stacker.mean_score, 1137.2460719022001, places=4)