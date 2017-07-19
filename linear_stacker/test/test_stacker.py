"""Test suite for PredictorStacker class
The suite is based on unittest and can be run from the test folder with
prompt$python -m unittest -v test_stacker
Coverage measures can be performed with
prompt$coverage run -m unittest -v test_stacker
prompt$coverage html
"""

import unittest
from linear_stacker import LinearPredictorStacker
import numpy as np


def metric_rmse(truth, hat):
    return np.mean(np.power(truth - hat, 2)) ** .5


def metric_mae(truth, hat):
    return np.mean(np.abs(truth - hat))


class TestPredictorStacker(unittest.TestCase):
    def test_init_stacker(self):
        """Check LinearPredictorStacker's attributes initialization"""
        stacker = LinearPredictorStacker(metric=metric_rmse)
        self.assertEqual(stacker.metric, metric_rmse)
        self.assertEqual(stacker.predictors, None)
        self.assertEqual(stacker.target, None)
        self.assertEqual(stacker.weights, None)
        self.assertEqual(stacker.score, None)
        self.assertEqual(stacker.maximize, False)
        self.assertEqual(stacker.algo, 'standard')
        self.assertEqual(stacker.max_predictors, 1.0)
        self.assertEqual(stacker.max_samples, 1.0)
        self.assertEqual(stacker.n_bags, 1)
        self.assertEqual(stacker.max_iter, 10)
        self.assertEqual(stacker.step, 1)
        self.assertEqual(stacker.verbose, 0)
        self.assertEqual(stacker.verb_round, 1)
        self.assertEqual(stacker.normed_weights, True)
        self.assertEqual(stacker.eps, 1e-5)
        self.assertEqual(stacker.seed, None)

    def test_fit_without_metric_raise_ValueError(self):
        """Test exception when no metric is provided"""
        self.assertRaises(ValueError, LinearPredictorStacker)

    def test_fit_without_predictors_raise_ValueError(self):
        stacker = LinearPredictorStacker(metric=metric_rmse)
        self.assertRaises(ValueError, stacker.fit)

    def test_add_predictors_one_file(self):
        stacker = LinearPredictorStacker(metric=metric_rmse)
        stacker.add_predictors_by_filename(files=['noid_OOF_predictions_2.csv'])
        self.assertEqual(len(stacker.target), 188318)
        self.assertEqual(len(stacker.predictors), 188318)
        self.assertEqual(stacker.predictors.shape[1], 20)

    def test_add_predictors_two_files(self):
        stacker = LinearPredictorStacker(metric=metric_rmse)
        stacker.add_predictors_by_filename(files=['noid_OOF_predictions_2.csv',
                                      'noid_OOF_predictions_3.csv'])
        self.assertEqual(len(stacker.target), 188318)
        self.assertEqual(len(stacker.predictors), 188318)
        self.assertEqual(stacker.predictors.shape[1], 21)

    def test_add_predictors_file_error_raise_ValueError(self):
        stacker = LinearPredictorStacker(metric=metric_rmse)
        self.assertRaises(ValueError, stacker.add_predictors_by_filename, files=['does_not_exist.csv'])

    def test_add_predictors_not_a_string_raise_TypeError(self):
        stacker = LinearPredictorStacker(metric=metric_rmse)
        self.assertRaises(TypeError, stacker.add_predictors_by_filename, files=[1, 2])

    def test_add_predictors_target_not_in_sync_raise_ValueError(self):
        stacker = LinearPredictorStacker(metric=metric_rmse)
        self.assertRaises(ValueError,
                          stacker.add_predictors_by_filename,
                          files=['noid_OOF_predictions_1.csv',
                                 'noid_OOF_predictions_2.csv'])

    def test_fit_regression_stacker_mae_no_bagging(self):
        """Test regression stacking with metric mean absolute error, no bagging"""
        stacker = LinearPredictorStacker(metric=metric_mae,
                                         max_iter=20,
                                         n_bags=1,
                                         max_predictors=1.,
                                         max_samples=1.,
                                         verbose=2)
        stacker.add_predictors_by_filename(files=['noid_OOF_predictions_2.csv',
                                                  'noid_OOF_predictions_3.csv',
                                                  'noid_OOF_predictions_4.csv'])
        self.assertEqual(len(stacker.target), 188318)
        self.assertEqual(len(stacker.predictors), 188318)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        self.assertAlmostEqual(stacker.score, 1136.4316755211291, places=4)
        self.assertAlmostEqual(stacker.mean_score, 1137.2460719022001, places=4)

    def test_fit_regression_stacker_rmse_no_bagging(self):
        """Test regression stacking with metric root mean squared error, no bagging"""
        stacker = LinearPredictorStacker(metric=metric_rmse,
                                         max_iter=20,
                                         n_bags=1,
                                         max_predictors=1.,
                                         max_samples=1.,
                                         verbose=2)
        stacker.add_predictors_by_filename(files=['noid_OOF_predictions_2.csv',
                                                  'noid_OOF_predictions_3.csv',
                                                  'noid_OOF_predictions_4.csv'])
        self.assertEqual(len(stacker.target), 188318)
        self.assertEqual(len(stacker.predictors), 188318)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        self.assertAlmostEqual(stacker.score, 1940.1115566433118, places=4)
        self.assertAlmostEqual(stacker.mean_score, 1942.3928913, places=4)

    def test_fit_regression_stacker_mae_ten_bags(self):
        """Test regression stacking with metric mean absolute error and 20 bags"""
        stacker = LinearPredictorStacker(metric=metric_mae,
                                         max_iter=10,
                                         n_bags=20,
                                         max_predictors=.8,
                                         max_samples=.8,
                                         seed=24698537)
        stacker.add_predictors_by_filename(files=['noid_OOF_predictions_2.csv',
                                                  'noid_OOF_predictions_3.csv',
                                                  'noid_OOF_predictions_4.csv'])
        self.assertEqual(len(stacker.target), 188318)
        self.assertEqual(len(stacker.predictors), 188318)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        self.assertAlmostEqual(stacker.score, 1136.9059858227927, places=4)
        self.assertAlmostEqual(stacker.mean_score, 1137.2460719022001, places=4)

    def test_fit_swapping_regression_stacker_no_bagging(self):
        """Test regression stacking with metric mean absolute error, no bagging"""
        stacker = LinearPredictorStacker(metric=metric_mae,
                                         algo='swapping',
                                         max_iter=20,
                                         n_bags=1,
                                         max_predictors=1.,
                                         max_samples=1.,
                                         verbose=2)
        stacker.add_predictors_by_filename(files=['noid_OOF_predictions_2.csv',
                                                  'noid_OOF_predictions_3.csv',
                                                  'noid_OOF_predictions_4.csv'])
        self.assertEqual(len(stacker.target), 188318)
        self.assertEqual(len(stacker.predictors), 188318)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        self.assertAlmostEqual(stacker.score, 1128.2264332661346, places=4)
        self.assertAlmostEqual(stacker.mean_score, 1137.2460719022001, places=4)

    def test_unsupported_algorithm(self):
        """Test unsupported algorithm"""
        self.assertRaises(ValueError, LinearPredictorStacker, metric=metric_rmse, algo='unsupported')