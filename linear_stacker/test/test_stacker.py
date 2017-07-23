"""Test suite for PredictorStacker class
The suite is based on unittest and can be run from the test folder with
prompt$python -m unittest -v test_stacker
Coverage measures can be performed with
prompt$coverage run -m unittest -v test_stacker
prompt$coverage html
"""

import unittest
from linear_stacker import (
    LinearPredictorStacker,
    BinaryClassificationLinearPredictorStacker,
    MultiLabelClassificationLinearPredictorStacker,
    RegressionLinearPredictorStacker
)
import numpy as np
import pandas as pd
import os.path
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def metric_rmse(truth, hat):
    return np.mean(np.power(truth - hat, 2)) ** .5


def metric_mae(truth, hat):
    return np.mean(np.abs(truth - hat))


def get_path(file_name):
    my_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(my_path, file_name)


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
        # self.assertEqual(stacker.step, 1)
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
        stacker.add_predictors_by_filename(files=[get_path('noid_OOF_predictions_2.csv')])
        self.assertEqual(len(stacker.target), 1000)
        self.assertEqual(len(stacker.predictors), 1000)
        self.assertEqual(stacker.predictors.shape[1], 20)

    def test_add_predictors_two_files(self):
        stacker = LinearPredictorStacker(metric=metric_rmse)
        stacker.add_predictors_by_filename(files=[get_path('noid_OOF_predictions_2.csv'),
                                                  get_path('noid_OOF_predictions_3.csv')])
        self.assertEqual(len(stacker.target), 1000)
        self.assertEqual(len(stacker.predictors), 1000)
        self.assertEqual(stacker.predictors.shape[1], 21)

    def test_add_predictors_file_error_raise_ValueError(self):
        stacker = LinearPredictorStacker(metric=metric_rmse)
        self.assertRaises(ValueError, stacker.add_predictors_by_filename, files=[get_path('does_not_exist.csv')])

    def test_add_predictors_not_a_string_raise_TypeError(self):
        stacker = LinearPredictorStacker(metric=metric_rmse)
        self.assertRaises(TypeError, stacker.add_predictors_by_filename, files=[1, 2])

    def test_add_predictors_target_not_in_sync_raise_ValueError(self):
        stacker = LinearPredictorStacker(metric=metric_rmse)
        self.assertRaises(ValueError,
                          stacker.add_predictors_by_filename,
                          files=[get_path('noid_OOF_predictions_1.csv'),
                                 get_path('noid_OOF_predictions_2.csv')])

    def test_fit_regression_stacker_mae_no_bagging(self):
        """Test regression stacking with metric mean absolute error, no bagging"""
        stacker = LinearPredictorStacker(metric=metric_mae,
                                         max_iter=20,
                                         n_bags=1,
                                         max_predictors=1.,
                                         max_samples=1.,
                                         verbose=0)
        stacker.add_predictors_by_filename(files=[get_path('noid_OOF_predictions_2.csv'),
                                                  get_path('noid_OOF_predictions_3.csv'),
                                                  get_path('noid_OOF_predictions_4.csv')])
        self.assertEqual(len(stacker.target), 1000)
        self.assertEqual(len(stacker.predictors), 1000)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        self.assertAlmostEqual(stacker.score, 1176.295406, places=4)
        # Old version self.assertAlmostEqual(stacker.score, 1187.1916616561432, places=4)
        self.assertAlmostEqual(stacker.mean_score, 1188.5725272161117, places=4)

    def test_fit_regression_stacker_rmse_no_bagging(self):
        """Test regression stacking with metric root mean squared error, no bagging"""
        stacker = LinearPredictorStacker(metric=metric_rmse,
                                         max_iter=182,
                                         n_bags=1,
                                         max_predictors=1.,
                                         max_samples=1.,
                                         verbose=0)
        stacker.add_predictors_by_filename(files=[get_path('noid_OOF_predictions_2.csv'),
                                                  get_path('noid_OOF_predictions_3.csv'),
                                                  get_path('noid_OOF_predictions_4.csv')])
        self.assertEqual(len(stacker.target), 1000)
        self.assertEqual(len(stacker.predictors), 1000)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        self.assertAlmostEqual(stacker.score, 1973.3901615401289, places=4)
        # Old version self.assertAlmostEqual(stacker.score, 2030.5021340510675, places=4)
        self.assertAlmostEqual(stacker.mean_score, 2032.2110846499691, places=4)

    def test_fit_regression_stacker_rmse_no_bagging_step_decrease(self):
        """Test regression stacking with metric root mean squared error, no bagging"""
        stacker = LinearPredictorStacker(metric=metric_rmse,
                                         max_iter=250,
                                         n_bags=1,
                                         max_predictors=1.,
                                         max_samples=1.,
                                         verbose=0)
        stacker.add_predictors_by_filename(files=[get_path('noid_OOF_predictions_2.csv'),
                                                  get_path('noid_OOF_predictions_3.csv'),
                                                  get_path('noid_OOF_predictions_4.csv')])
        self.assertEqual(len(stacker.target), 1000)
        self.assertEqual(len(stacker.predictors), 1000)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        self.assertAlmostEqual(stacker.score, 1972.574584232116, places=4)
        # Old version self.assertAlmostEqual(stacker.score, 2030.5021340510675, places=4)
        self.assertAlmostEqual(stacker.mean_score, 2032.2110846499691, places=4)

    def test_swapping_fit_regression_stacker_rmse_no_bagging(self):
        """Test regression stacking with metric root mean squared error, no bagging"""
        stacker = LinearPredictorStacker(metric=metric_rmse,
                                         algo='swapping',
                                         max_iter=20,
                                         n_bags=1,
                                         max_predictors=1.,
                                         max_samples=1.,
                                         verbose=0)
        stacker.add_predictors_by_filename(files=[get_path('noid_OOF_predictions_2.csv'),
                                                  get_path('noid_OOF_predictions_3.csv'),
                                                  get_path('noid_OOF_predictions_4.csv')])
        self.assertEqual(len(stacker.target), 1000)
        self.assertEqual(len(stacker.predictors), 1000)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        self.assertAlmostEqual(stacker.score, 2008.0897782026507, places=4)
        self.assertAlmostEqual(stacker.mean_score, 2032.2110846499691, places=4)

    def test_swapping_fit_regression_stacker_rmse_10_bags(self):
        """Test regression stacking with metric root mean squared error, no bagging"""
        stacker = LinearPredictorStacker(metric=metric_rmse,
                                         algo='swapping',
                                         max_iter=20,
                                         n_bags=10,
                                         max_predictors=.8,
                                         max_samples=.8,
                                         verbose=0,
                                         seed=0)
        stacker.add_predictors_by_filename(files=[get_path('noid_OOF_predictions_2.csv'),
                                                  get_path('noid_OOF_predictions_3.csv'),
                                                  get_path('noid_OOF_predictions_4.csv')])
        self.assertEqual(len(stacker.target), 1000)
        self.assertEqual(len(stacker.predictors), 1000)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        self.assertAlmostEqual(stacker.score, 2008.6870283678847, places=4)
        self.assertAlmostEqual(stacker.mean_score, 2032.2110846499691, places=4)

    def test_fit_regression_stacker_mae_ten_bags(self):
        """Test regression stacking with metric mean absolute error and 20 bags"""
        stacker = LinearPredictorStacker(metric=metric_mae,
                                         max_iter=10,
                                         n_bags=20,
                                         max_predictors=.8,
                                         max_samples=.8,
                                         seed=24698537)
        stacker.add_predictors_by_filename(files=[get_path('noid_OOF_predictions_2.csv'),
                                                  get_path('noid_OOF_predictions_3.csv'),
                                                  get_path('noid_OOF_predictions_4.csv')])
        self.assertEqual(len(stacker.target), 1000)
        self.assertEqual(len(stacker.predictors), 1000)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        self.assertAlmostEqual(stacker.score, 1179.2406088734808, places=4)
        # Old version self.assertAlmostEqual(stacker.score, 1187.6537373418842, places=4)
        self.assertAlmostEqual(stacker.mean_score, 1188.5725272161117, places=4)

    def test_fit_swapping_regression_stacker_no_bagging_normed_weights(self):
        """Test regression stacking with metric mean absolute error, no bagging"""
        stacker = LinearPredictorStacker(metric=metric_mae,
                                         algo='swapping',
                                         normed_weights=True,
                                         max_iter=2000,
                                         n_bags=1,
                                         max_predictors=1.,
                                         max_samples=1.,
                                         verbose=2,
                                         eps=1e-3)
        stacker.add_predictors_by_filename(files=[get_path('noid_OOF_predictions_2.csv'),
                                                  get_path('noid_OOF_predictions_3.csv'),
                                                  get_path('noid_OOF_predictions_4.csv')])
        self.assertEqual(len(stacker.target), 1000)
        self.assertEqual(len(stacker.predictors), 1000)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        self.assertAlmostEqual(stacker.score, 1156.5085876533767, places=4)
        self.assertAlmostEqual(stacker.mean_score, 1188.5725272161117, places=4)

    def test_standard_fit_regression_stacker_no_bagging_free_weights(self):
        """Test regression stacking with metric mean absolute error, no bagging"""
        stacker = LinearPredictorStacker(metric=metric_mae,
                                         algo='standard',
                                         normed_weights=False,
                                         max_iter=200,
                                         n_bags=1,
                                         max_predictors=1.,
                                         max_samples=1.,
                                         verbose=2,
                                         eps=1e-3,
                                         seed=0)
        stacker.add_predictors_by_filename(files=[get_path('noid_OOF_predictions_2.csv'),
                                                  get_path('noid_OOF_predictions_3.csv'),
                                                  get_path('noid_OOF_predictions_4.csv')])
        self.assertEqual(len(stacker.target), 1000)
        self.assertEqual(len(stacker.predictors), 1000)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        print(stacker.get_weights())
        self.assertAlmostEqual(stacker.score, 1185.0745095110819, places=4)
        self.assertAlmostEqual(stacker.mean_score, 1188.5725272161117, places=4)

    def test_unsupported_algorithm(self):
        """Test unsupported algorithm"""
        self.assertRaises(ValueError, LinearPredictorStacker, metric=metric_rmse, algo='unsupported')

    def test_fit_with_predictors_and_target(self):
        data = pd.read_csv(get_path('noid_OOF_predictions_2.csv'))
        target = data.loss
        data.drop(['loss'], axis=1, inplace=True)
        stacker = LinearPredictorStacker(metric=metric_rmse,
                                         algo='swapping',
                                         max_iter=200,
                                         n_bags=1,
                                         max_predictors=1.,
                                         max_samples=1.,
                                         verbose=0)
        stacker.fit(predictors=data, target=target)
        self.assertAlmostEqual(stacker.score, 1969.3946377360355, places=4)
        self.assertAlmostEqual(stacker.mean_score, 2033.0404650725116, places=4)
        # test weights
        self.assertTupleEqual(stacker.get_weights(), (-0.75000000000000011,
                                                      0.050000000000000003,
                                                      -0.44999999999999996,
                                                      2.4499999999999993,
                                                      0.050000000000000003,
                                                      -1.2500000000000004,
                                                      0.050000000000000003,
                                                      -1.7500000000000009,
                                                      0.050000000000000003,
                                                      0.44999999999999996,
                                                      0.25,
                                                      3.099999999999997,
                                                      -0.75000000000000011,
                                                      -0.39999999999999997,
                                                      2.5499999999999989,
                                                      -2.3999999999999995,
                                                      -1.850000000000001,
                                                      0.050000000000000003,
                                                      1.5000000000000007,
                                                      0.050000000000000003))

    def test_fit_with_predictors_and_target_exception_tests(self):
        data = pd.read_csv(get_path('noid_OOF_predictions_2.csv'))
        target = data.loss
        data.drop(['loss'], axis=1, inplace=True)
        stacker = LinearPredictorStacker(metric=metric_rmse,
                                         algo='swapping',
                                         max_iter=20,
                                         n_bags=1,
                                         max_predictors=1.,
                                         max_samples=1.,
                                         verbose=0)
        # Predictors and target do not have same length
        self.assertRaises(ValueError, stacker.fit, predictors=data, target=target.head(100))
        # Predictors contain null values
        data_null = data.copy()
        data_null[data_null < 1000] = np.nan
        self.assertRaises(ValueError, stacker.fit, predictors=data_null, target=target)
        # target contains null values
        target_null = target.copy()
        target_null[target_null < 1000] = np.nan
        self.assertRaises(ValueError, stacker.fit, predictors=data, target=target_null)
        # target contains more than one columns
        self.assertRaises(ValueError, stacker.fit, predictors=data, target=data)

    def test_regression_stacking(self):
        # For a more appropriate way to use predition stacking see the examples
        # Load boston dataset
        dataset = load_boston()
        # Split in data and target
        X_full, y_full = dataset.data, dataset.target

        regressors = [
            ('ridge', Ridge(alpha=0.001, normalize=True, random_state=0)),
            ('lasso', Lasso(alpha=0.01, normalize=True, random_state=1)),
            ('xtr', ExtraTreesRegressor(n_estimators=50, max_features=.4, max_depth=10, random_state=2, n_jobs=-1)),
            ('rfr', RandomForestRegressor(n_estimators=50, max_features=.2, max_depth=10, random_state=3, n_jobs=-1)),
            ('gbr', GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=.1, random_state=4))
        ]

        # Go through regressors to get predictions
        predictors = np.empty((len(X_full), len(regressors)))
        for i_reg, (name, reg) in enumerate(regressors):
            # Fit regressor
            reg.fit(X_full, y_full)
            # predict
            predictors[:, i_reg] = reg.predict(X_full)
            # Display MSE
            print('reg %s mse %.5f' % (name, mean_squared_error(y_full, predictors[:, i_reg])))

        # Now use regression stacker
        stacker = RegressionLinearPredictorStacker(metric=mean_squared_error)
        stacker.fit(pd.DataFrame(predictors), pd.Series(y_full))
        self.assertAlmostEqual(1.2202887, mean_squared_error(y_full, stacker.predict(predictors)), places=5)

    def get_classifier_predictors(self):
        # For a more appropriate way to use predition stacking see the examples
        # Load breast cancer dataset
        dataset = load_breast_cancer()
        # Split in data and target
        X_full, y_full = dataset.data, dataset.target

        classifiers = [
            ('logit', LogisticRegression(C=0.01, random_state=1)),
            ('xtr', ExtraTreesClassifier(n_estimators=50, max_features=.4, max_depth=10, random_state=2, n_jobs=-1)),
            ('rfr', RandomForestClassifier(n_estimators=50, max_features=.2, max_depth=10, random_state=3, n_jobs=-1)),
            ('gbr', GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=.1, random_state=4))
        ]

        # Go through each classifier and get its predictions
        predictors = np.empty((len(X_full), len(classifiers)))
        for i_clf, (name, clf) in enumerate(classifiers):
            # Fit classifier
            clf.fit(X_full, y_full)
            # Get predictor
            predictors[:, i_clf] = clf.predict_proba(X_full)[:, 1]
            print("clf %s log_loss %.5f" % (name, log_loss(y_full, predictors[:, i_clf])))

        return predictors, dataset.target

    def test_classification_stacking(self):
        predictors, target = self.get_classifier_predictors()
        # Now use stacker
        stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss)
        stacker.fit(pd.DataFrame(predictors), pd.Series(target))
        self.assertAlmostEqual(0.0010017, log_loss(target, stacker.predict_proba(predictors)), places=5)
        self.assertAlmostEqual(1.0, accuracy_score(target, stacker.predict(predictors)), places=2)

    def test_several_fits_classification(self):
        predictors, target = self.get_classifier_predictors()
        stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss, max_iter=35, verbose=2,
                                                             algo='swapping', normed_weights=False)
        # first fit
        stacker.fit(pd.DataFrame(predictors), pd.Series(target))
        self.assertAlmostEqual(0.00018589380467956424, log_loss(target, stacker.predict_proba(predictors)), places=5)
        self.assertAlmostEqual(1.0, accuracy_score(target, stacker.predict(predictors)), places=2)
        # second fit should give same values
        stacker.fit(pd.DataFrame(predictors), pd.Series(target))
        self.assertAlmostEqual(0.00018589380467956424, log_loss(target, stacker.predict_proba(predictors)), places=5)
        self.assertAlmostEqual(1.0, accuracy_score(target, stacker.predict(predictors)), places=2)

        stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss, max_iter=35, verbose=2,
                                                             algo='standard', normed_weights=False)
        # first fit
        stacker.fit(pd.DataFrame(predictors), pd.Series(target))
        self.assertAlmostEqual(0.00025499752810984901, log_loss(target, stacker.predict_proba(predictors)), places=5)
        self.assertAlmostEqual(1.0, accuracy_score(target, stacker.predict(predictors)), places=2)
        # second fit should give same values
        stacker.fit(pd.DataFrame(predictors), pd.Series(target))
        self.assertAlmostEqual(0.00025499752810984901, log_loss(target, stacker.predict_proba(predictors)), places=5)
        self.assertAlmostEqual(1.0, accuracy_score(target, stacker.predict(predictors)), places=2)

    def test_regression_r2_score_maximization(self):
        stacker = LinearPredictorStacker(metric=r2_score,
                                         maximize=True,
                                         algo='standard',
                                         normed_weights=False,
                                         max_iter=200,
                                         n_bags=1,
                                         max_predictors=1.,
                                         max_samples=1.,
                                         verbose=2,
                                         eps=1e-4,
                                         seed=0)
        stacker.add_predictors_by_filename(files=[get_path('noid_OOF_predictions_2.csv'),
                                                  get_path('noid_OOF_predictions_3.csv'),
                                                  get_path('noid_OOF_predictions_4.csv')])
        self.assertEqual(len(stacker.target), 1000)
        self.assertEqual(len(stacker.predictors), 1000)
        self.assertEqual(stacker.predictors.shape[1], 22)
        stacker.fit()
        self.assertAlmostEqual(stacker.score, 0.5478882, places=4)
        self.assertAlmostEqual(stacker.mean_score, 0.537271112149033, places=4)

    def test_multilabelclassification(self):
        # Load iris dataset
        dataset = load_iris()
        n_classes = len(np.unique(dataset.target))

        classifiers = [
            ('logit', LogisticRegression(C=0.1,
                                         random_state=1)),
            ('xtr', ExtraTreesClassifier(n_estimators=50,
                                         max_features=.4,
                                         max_depth=5,
                                         random_state=2,
                                         n_jobs=-1)),
            ('rfr', RandomForestClassifier(n_estimators=50,
                                           max_features=.2,
                                           max_depth=5,
                                           random_state=3,
                                           n_jobs=-1)),
            ('gbr', GradientBoostingClassifier(n_estimators=20,
                                               max_depth=2,
                                               subsample=.6,
                                               learning_rate=.01,
                                               random_state=4))
        ]

        # Go through classifiers
        predictors_probas = np.zeros((len(dataset.data), len(classifiers) * n_classes))

        for reg_i, (name, reg) in enumerate(classifiers):
            # Fit the classifier
            reg.fit(dataset.data, dataset.target)
            # Predict labels and probas data
            predictors_probas[:, reg_i * n_classes: (reg_i + 1) * n_classes] = reg.predict_proba(dataset.data)

        print(predictors_probas[:5, :])
        # Use stacker
        stacker = MultiLabelClassificationLinearPredictorStacker(metric=log_loss, verbose=2, max_iter=100, seed=0)
        stacker.fit(predictors=predictors_probas, target=dataset.target)
        self.assertAlmostEqual(2.94e-5,
                               log_loss(dataset.target, stacker.predict_proba(predictors=predictors_probas)),
                               places=5)
        self.assertAlmostEqual(1.0, accuracy_score(dataset.target, stacker.predict(predictors=predictors_probas)))

    def test_not_fitted_error(self):
        stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss)
        self.assertRaises(ValueError, stacker.predict)

    def test_bad_type_for_predictors_error(self):
        stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss)
        self.assertRaises(ValueError, stacker.fit, predictors=pd.Series([1, 2, 3]) , target=np.array([1, 2, 3]))

    def test_bad_type_for_target_error(self):
        stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss)
        self.assertRaises(ValueError, stacker.fit, predictors=np.array([[1, 2], [2, 3]]), target=pd.DataFrame([1, 2]))
