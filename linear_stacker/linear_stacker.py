from __future__ import division
import os.path
import pandas as pd
import numpy as np


class LinearPredictorStacker(object):
    def __init__(self,
                 metric=None,
                 maximize=False,
                 algo='standard',
                 max_predictors=1.0,
                 max_samples=1.0,
                 n_bags=1,
                 max_iter=10,
                 step=1,
                 verbose=0,
                 verb_round=1,
                 normed_weights=True,
                 eps=1e-5,
                 seed=None):
        """
        Init a LinearPredictorStacker
        :param metric: metrics to be used to optimize linear mix of predictors
        :param maximize: set to True if optimizer has to maximize the metric, set to False otherwise
        :param algo: type of optimization, can be 'standard' or 'swapping'
        :param max_predictors: float default to 1.0, ratio of predictors used for each bag
        :param max_samples: float default to 1.0, ratio of samples used for each bag
        :param n_bags: number of bags, defaults to 1 (i.e. no bagging)
        :param max_iter: maximum number of iterations for the optimization process
        :param step: step size for the optimizer, defaults to 1
            .. code:: python

                candidate = prediction +/- step * i_th_predictor

        :param verbose: overall verbosity of the optimization process, defaults to 0
        :param verb_round: if verbose is more than 1, verb_round sets the number of rounds after which info is output
        :param normed_weights: set to True if predictors' weights have to sum to 1, defaults to True
        :param eps: tolerance for optimization improvement, defaults to 1e-5
        :param seed: used to seed random processes
        """

        if metric is None:
            raise ValueError('No metric has been provided')
        if algo not in ['swapping', 'standard']:
            raise ValueError('Algorithm must be either "standard" or "swapping"')

        self.metric = metric
        self.maximize = maximize
        self.algo = algo
        self.max_predictors = max_predictors
        self.max_samples = max_samples
        self.n_bags = n_bags
        self.max_iter = max_iter
        self.step = step
        self.verbose = verbose
        self.verb_round = verb_round
        self.normed_weights = normed_weights
        self.eps = eps
        self.seed = seed

        self.predictors = None
        self.target = None
        self.score = None
        self.weights = None
        self.mean_score = None
        self.fitted = None

    def add_predictors_by_filename(self, files=None):
        """
        Add a list of file names to the linear_stacker
        Files contain a minimum of 2 columns :
        - 1st column is the prediction (mandatory)
        - last column is the target (mandatory)
        If target column is not specified, target should be provided
        when fitting the linear_stacker
        :param files: list of file names containing predictions in csv format
        :return:
        """
        # Check provided files exist
        for f in files:
            if 'str' not in str(type(f)):
                raise TypeError('file ' + str(f) + ' is not a string')
            if not os.path.isfile(f):
                raise ValueError('file ' + f + ' is not a file or does not exist')

        # Get the predictors
        self._get_predictors(files)

    def _get_predictors(self, files):
        """
        Read predictors from provided files and checks length are equal and in line using target
        :param files: list of file names containing predictions in csv format
        :return: Array of predictors
        """
        for f in files:
            # Read file
            pred = pd.read_csv(f, delimiter=',')

            # Check number of columns is divided by num_class
            # if (len(pred.columns) - 1) % self.num_class != 0:
            #     raise ValueError('file ' + f + ' does not contain enough classes')

            # Check data shape : should contain all classes + target
            if len(pred.shape) < 2:
                # We don't have a target in the file
                raise ValueError('file ' + f + ' should contain at least 2 columns (prediction and target)')
            else:
                # Set the target if not assigned yet
                if self.target is None:
                    # Target is the last column
                    self.target = pred.values[:, -1]

                # Check current target is in sync with registered target
                if (np.sum(np.abs(self.target - pred.values[:, -1]))) > 1e-5:
                    raise ValueError('target in file ' + f + ' is out of sync')

                # Set predictors
                if self.predictors is None:
                    # Set predictors as all column except last
                    self.predictors = pd.DataFrame()
                    for feature in pred.columns[:-1]:
                        self.predictors[feature] = pred[feature]
                else:
                    if len(pred) != len(self.predictors):
                        raise ValueError('Unexpected length of predictors in file ' + str(f))
                    for feature in pred.columns[:-1]:
                        self.predictors[feature] = pred[feature]

    def _check_predictors_and_target(self, predictors, target):
        # Check length
        if len(predictors) != len(target):
            raise ValueError('Target and predictors have different length')
        if len(target.shape) > 1:
            raise ValueError('Target contains more than one column')
        if np.sum(predictors.isnull().sum()) > 0:
            raise ValueError('Predictors contain NaN')
        if target.isnull().sum() > 0:
            raise ValueError('Target contain NaN')
        # Set predictors and target
        self.predictors = predictors
        self.target = np.array(target)

    def fit(self, predictors=None, target=None):

        # Check predictors and target
        if (predictors is not None) and (target is not None):
            self._check_predictors_and_target(predictors, target)
        else:
            # Predictors have been set using files
            if (self.predictors is None) | (self.target is None):
                raise ValueError('predictors and target must be set before fitting')

        # Check algo
        if self.algo == 'standard':
            self._standard_fit()
        elif self.algo == 'swapping':
            self._swapping_fit()

    def _standard_fit(self):
        """
        Standard optimization process.

        At each round each predictor is either added or subtracted to to the overall prediction and overall score
        improvement is tested. The best operation is kept.
        """
        self.fitted = False

        # Compute mean score
        self.mean_score = self.metric(self.target, self.predictors.mean(axis=1).values)

        # Create samples and predictors indexes
        samp_indexes = np.arange(self.predictors.shape[0])
        pred_indexes = np.arange(self.predictors.shape[1])

        # Set a seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)

        # Init weights
        self.weights = np.zeros(self.predictors.shape[1])

        # Run bagging
        predictors = self.predictors.values
        features = self.predictors.columns

        for bag in range(self.n_bags):
            # Shuffle indexes
            np.random.shuffle(samp_indexes)
            np.random.shuffle(pred_indexes)

            # Get bagged predictors
            nb_pred = int(self.predictors.shape[1] * self.max_predictors)
            nb_samp = int(self.predictors.shape[0] * self.max_samples)
            pred_idx = pred_indexes[:nb_pred]
            samp_idx = samp_indexes[:nb_samp]

            bag_predictors = predictors[samp_idx, :][:, pred_idx]
            bag_target = self.target[samp_idx]

            # Init weights and prediction for current bag
            weights = self._init_weights(nb_pred)

            # Compute prediction
            prediction = self._compute_prediction(bag_predictors, weights)

            # Set benchmark and print it
            benchmark = self.metric(bag_target, prediction)
            if self.verbose >= 2:
                print('Benchmark : ', benchmark)

            # Try to improve on the benchmark
            improve = True
            iter_ = 0
            init_step = self.step
            while improve and (iter_ < self.max_iter):
                pred_id, score, weight_upd = self._search_best_weight_candidate(bag_predictors,
                                                                                bag_target,
                                                                                benchmark,
                                                                                weights)
                # Update benchmark and weights if things have improved
                if score < benchmark and (benchmark - score) >= self.eps:
                    # Set improvement indicator
                    improve = True

                    # Update weights
                    weights = self._update_weights(weights=weights, w_pos=pred_id, step=weight_upd)

                    # Compute prediction
                    prediction = self._compute_prediction(predictors=bag_predictors, weights=weights)

                    # Compute benchmark
                    benchmark = self.metric(bag_target, prediction)

                    # Display round improvement
                    if self.verbose >= 2 and (iter_ % self.verb_round == 0):
                        print('Round %6d benchmark for feature %20s : %13.6f'
                              % (iter_, features[pred_id], benchmark), weight_upd)
                else:
                    if self.step <= init_step / 100:
                        improve = False
                        if self.verbose >= 2:
                            print("Best round is %d" % iter_)
                    else:
                        improve = True
                        self.step /= 10
                        # print("New step : ", self.step)

                    # # No improvement found, display best round if needed
                    # if self.verbose >= 2:
                    #     print("Best round is %d" % iter_)
                    # improve = False
                iter_ += 1

            # Display current bag score
            bag_prediction = self._compute_prediction(predictors=bag_predictors, weights=weights)
            if self.verbose >= 1:
                print('Bag ' + str(bag) + ' final score : ', self.metric(bag_target, bag_prediction))

            # Update global weights with bag fit
            self.weights[pred_idx] += weights / self.n_bags

        # Display bagged prediction score
        bagged_prediction = self._compute_prediction(predictors=predictors, weights=self.weights)
        if self.verbose >= 1:
            print('Final score : ', self.metric(self.target, bagged_prediction))

        self.score = self.metric(self.target, bagged_prediction)
        # self.weights = self.weightss

        self.fitted = True

    def _search_best_weight_candidate(self, bag_predictors, bag_target, benchmark, weights):
        best_score = benchmark
        best_predictor = 0
        sign_i = None
        # Loop over all predictors and try to remove or add it
        for i in range(bag_predictors.shape[1]):
            for the_sign in [self.step, -self.step]:
                # update weights
                candidate_weights = self._update_weights(weights, i, the_sign)

                # Compute new candidate prediction
                candidate_pred = self._compute_prediction(predictors=bag_predictors,
                                                          weights=candidate_weights)

                # compute candidate score
                candidate_score = self.metric(bag_target, candidate_pred)

                # print('New score for predictor ', str(i), ':', candidate_score)
                # Check for score improvement
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_predictor = i
                    sign_i = the_sign

        return best_predictor, best_score, sign_i

    def _init_weights(self, nb_pred):
        if self.normed_weights:
            weights = np.ones(nb_pred) / nb_pred
            # Override step
            self.step = 1.0 / nb_pred
        else:
            weights = np.zeros(nb_pred)

        return weights

    def _update_weights(self, weights, w_pos, step):
        # Make sure we do not change passed weights
        weights_ = weights.copy()
        weights_[w_pos] += step
        if self.normed_weights:
            weights_ /= np.sum(weights_)

        return weights_

    def _compute_prediction(self, predictors, weights):
        """Compute prediction for predictors and weights"""
        prediction = np.zeros(len(predictors))
        if self.normed_weights:
            for i, weight in enumerate(weights):
                prediction += weight * predictors[:, i] / (np.sum(weights))
        else:
            for i, weight in enumerate(weights):
                prediction += weight * predictors[:, i]

        return prediction

    def _transform(self, data):
        """Apply fitted weights to the data"""
        if not self.fitted:
            raise (ValueError, 'Stacker is not fitted yet')

        prediction = np.zeros(len(data))
        tmp = np.array(data)
        for i, w in enumerate(self.weights):
            prediction += w * tmp[:, i]
        return prediction

    def _swapping_fit(self):
        """
        Swapping optimization process.

        At each round each pair of predictors is tested (one is removed when the other is added)
        Overall score improvement is tested. The best operation is kept.

        Note that Predictors' contributions are allowed to be negative.

        This optimization process often leads to better results compared to the standard process.
        """
        self.fitted = False

        # Compute mean score
        self.mean_score = self.metric(self.target, self.predictors.mean(axis=1).values)

        # Create samples and predictors indexes
        samp_indexes = np.arange(self.predictors.shape[0])
        pred_indexes = np.arange(self.predictors.shape[1])

        # Set a seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # Init weights
        self.weights = np.zeros(self.predictors.shape[1])

        # Run bagging
        predictors = self.predictors.values
        features = self.predictors.columns

        for bag in range(self.n_bags):
            # Shuffle indexes
            np.random.shuffle(samp_indexes)
            np.random.shuffle(pred_indexes)

            # Get ratioed predictors
            nb_pred = int(self.predictors.shape[1] * self.max_predictors)
            nb_samp = int(self.predictors.shape[0] * self.max_samples)
            pred_idx = pred_indexes[:nb_pred]
            samp_idx = samp_indexes[:nb_samp]

            bag_predictors = predictors[samp_idx, :][:, pred_idx]
            bag_target = self.target[samp_idx]

            # Init weights to equal weights
            weights = np.ones(bag_predictors.shape[1]) / bag_predictors.shape[1]

            # Compute initial prediction and benchmark
            prediction = self._compute_prediction(predictors=bag_predictors, weights=weights)

            # Set benchmark and display if needed
            benchmark = self.metric(bag_target, prediction)
            if self.verbose >= 2:
                print('Benchmark : ', benchmark)

            # Try to improve on the benchmark
            init_step = 1.0 / bag_predictors.shape[1]
            step = init_step
            improve = True
            iter_ = 0
            while improve and (iter_ < self.max_iter):
                best_score = benchmark
                best_swap = (0, 0)
                # Try to modify weights
                for the_step in [step]:
                    candidate_weights = weights.copy()
                    for i in range(len(weights)):
                        candidate_weights_i = candidate_weights.copy()
                        # Do not go under 0 or above 1
                        # if (candidate_weights_i[i] <= 0) and the_step < 0:
                        #    continue
                        # if (candidate_weights_i[i] >= 1) and the_step > 0:
                        #    continue
                        # Modify current weight and initialize counter weight
                        candidate_weights_i[i] += the_step
                        counter_step = -the_step
                        # Now find a weight to counter modify
                        for j in range(len(weights)):
                            # print(i, j)
                            candidate_weights_j = candidate_weights_i.copy()
                            if j == i:
                                continue
                            # Do not go under 0 or above 1
                            # if (candidate_weights_j[j] <= 0) and counter_step < 0:
                            #    continue
                            # if (candidate_weights_j[j] >= 1) and counter_step > 0:
                            #    continue
                            candidate_weights_j[j] += counter_step
                            # Compute candidate_weights score
                            candidate_pred = self._compute_prediction(predictors=bag_predictors,
                                                                      weights=candidate_weights_j)

                            # compute score
                            candidate_score = self.metric(bag_target, candidate_pred)
                            # Update best params
                            if candidate_score < best_score:
                                best_score = candidate_score
                                best_swap = (i, j)
                                best_step = the_step

                if best_score < benchmark:
                    improve = True
                    # Update weights with best combination
                    weights[best_swap[0]] += best_step
                    weights[best_swap[1]] -= best_step

                    # Compute prediction
                    prediction = self._compute_prediction(predictors=bag_predictors, weights=weights)

                    # Compute new benchmark and display if required
                    benchmark = self.metric(bag_target, prediction)
                    if self.verbose >= 2 and (iter_ % self.verb_round == 0):
                        print('Round %6d benchmark for feature %20s / %20s : %13.6f'
                              # % (iter_, features[best_swap[0]], features[best_swap[1]], benchmark), best_step)
                              % (iter_,
                                 features[pred_idx[best_swap[0]]],
                                 features[pred_idx[best_swap[1]]],
                                 benchmark), best_step)
                else:
                    # Decrease step size in case no improvement is found
                    if step <= init_step / 100:
                        improve = False
                        if self.verbose >= 2:
                            print("Best round is %d" % iter_)
                    else:
                        step /= 10

                # Increment iteration
                iter_ += 1

            # Print current bag score
            last_prediction = np.zeros(nb_samp)
            for i in range(len(weights)):
                last_prediction += weights[i] * bag_predictors[:, i]
            if self.verbose >= 1:
                print('Bag ' + str(bag) + ' final score : ', self.metric(bag_target, last_prediction))

            # Iteration finished for current bag, update self.weights
            self.weights[pred_idx] += weights / self.n_bags

        # All bags done, compute bagged score and display if needed
        bagged_prediction = self._compute_prediction(predictors=predictors, weights=self.weights)
        if self.verbose >= 1:
            print('Final score : ', self.metric(self.target, bagged_prediction))

        self.score = self.metric(self.target, bagged_prediction)
        self.fitted = True

    def get_weights(self):
        return tuple(self.weights)


class BinaryClassificationLinearPredictorStacker(LinearPredictorStacker):
    """
    Binary Classification Linear Stacker computes the best weights to linearly merge predictors
    against a specific metric.

    Note that Stacker raw output is probability based.
    """

    def predict_proba(self, predictors=None):
        """Apply linear stacker to predictors"""
        return self._transform(predictors)

    def predict(self, predictors=None, threshold=.5):
        """
        Apply linear stacker to predictors and then assign label against a threshold
        :param predictors: set of predictors to be merged
         number of predictors should be the same as the set used to train the stacker
        :param threshold: threshold used to assign binary label. defaults to .5
        :return: weighted sum of predictors
        """
        probas = self._transform(predictors)
        probas[probas >= threshold] = 1
        probas[probas < threshold] = 0
        return probas


class RegressionLinearPredictorStacker(LinearPredictorStacker):
    def predict(self, predictors=None):
        """
        Apply linear stacker to predictors
        :param predictors: ctors to be merged
         number of predictors should be the same as the set used to train the stacker
        :return: weighted sum of predictors
        """
        return self._transform(predictors)
