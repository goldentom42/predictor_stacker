from __future__ import division
import os.path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#np.set_printoptions(precision=2)


class LinearPredictorStacker(object):
    STANDARD = 0
    SWAPPING = 1

    def __init__(self,
                 metric=None,
                 maximize=False,
                 algorithm=STANDARD,
                 colsample=1.0,
                 subsample=1.0,
                 n_bags=1,
                 max_iter=10,
                 verbose=0,
                 verb_round=1,
                 normed_weights=True,
                 probabilities=False,
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
        :param probabilities: set to True if predictor's output must be between 0 and 1
        :param eps: tolerance for optimization improvement, defaults to 1e-5
        :param seed: used to seed random processes
        """

        if metric is None:
            raise ValueError('No metric has been provided')
        if algorithm not in [self.STANDARD, self.SWAPPING]:
            raise ValueError('Algorithm must be either "standard" or "swapping"')

        self.metric = metric
        self.maximize = maximize
        self.algo = algorithm
        self.max_predictors = colsample
        self.max_samples = subsample
        self.n_bags = n_bags
        self.max_iter = max_iter
        self.verbose = verbose
        self.verb_round = verb_round
        self.normed_weights = normed_weights
        self.probabilities = probabilities
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
        if np.sum(np.sum(np.isnan(predictors), axis=1)) > 0:
            raise ValueError('Predictors contain NaN')
        if np.sum(np.isnan(target)) > 0:
            raise ValueError('Target contain NaN')

        # check predictors type and set predictors
        if type(predictors) is pd.core.frame.DataFrame:
            self.predictors = predictors
        elif type(predictors) is np.ndarray:
            self.predictors = pd.DataFrame(predictors, columns=['p' + str(i) for i in range(predictors.shape[1])])
        else:
            raise TypeError('Predictors should be an ndarray or a pandas DataFrame object')

        # Check target type and set target
        if type(target) is pd.core.series.Series:
            self.target = np.array(target)
        elif type(target) is np.ndarray:
            self.target = target
        else:
            raise TypeError('Target should be an ndarray or a pandas Series object')

    def fit(self, predictors=None, target=None):

        # Check predictors and target
        if (predictors is not None) and (target is not None):
            self._check_predictors_and_target(predictors, target)
        else:
            # Predictors have been set using files
            if (self.predictors is None) | (self.target is None):
                raise ValueError('predictors and target must be set before fitting')

        # Check algo
        if self.algo == LinearPredictorStacker.STANDARD:
            self._standard_fit()
        elif self.algo == LinearPredictorStacker.SWAPPING:
            self._swapping_fit()

    def _standard_fit(self):
        """
        Standard optimization process.

        At each round :
        - A portion (step) of each predictor is either added or subtracted to the overall prediction
        - Overall score improvement is tested.
        - The best operation is kept.
        """

        self.fitted = False

        # Create a benchmark score with predictors mean
        self.mean_score = self.metric(self.target, self.predictors.mean(axis=1).values)

        # Create samples and predictors indexes, this is used for bagging purposes
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

            # Get bagged predictors (without replacement)
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
            init_step = 1.0 / predictors.shape[1]
            self.step = init_step
            while improve and (iter_ < self.max_iter):
                pred_id, score, weight_upd = self._search_best_weight_candidate(bag_predictors,
                                                                                bag_target,
                                                                                benchmark,
                                                                                weights)
                # Update benchmark and weights if things have improved
                if self._check_score_improvement(benchmark=benchmark, score=score):
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
                if self._check_score_improvement(benchmark=best_score, score=candidate_score):
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

        if self.probabilities:
            prediction = np.clip(prediction, 1e-10, 1 - 1e-10)

        return prediction

    def _transform(self, data):
        """Apply fitted weights to the data"""
        if not self.fitted:
            raise ValueError('Stacker is not fitted yet')

        prediction = np.zeros(len(data))
        tmp = np.array(data)
        for i, w in enumerate(self.weights):
            prediction += w * tmp[:, i]
        return prediction

    def _check_score_improvement(self, benchmark, score):
        if self.maximize:
            return (score > benchmark) and abs(benchmark - score) >= self.eps
        else:
            return (score < benchmark) and abs(benchmark - score) >= self.eps

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
                best_score, best_step, best_swap = self._search_best_pair_swap_candidate(bag_predictors,
                                                                                         bag_target,
                                                                                         benchmark,
                                                                                         step,
                                                                                         weights)

                if self._check_score_improvement(benchmark=benchmark, score=best_score):
                    improve = True
                    # Update weights with best combination
                    weights[best_swap[0]] += best_step
                    weights[best_swap[1]] -= best_step

                    # Compute prediction
                    prediction = self._compute_prediction(predictors=bag_predictors, weights=weights)


                    # Compute new benchmark and display if required
                    benchmark = self.metric(bag_target, prediction)
                    if self.verbose >= 2 and (iter_ % self.verb_round == 0):
                        print('Round %6d benchmark replacing %-20s by %-20s : %13.6f'
                              # % (iter_, features[best_swap[0]], features[best_swap[1]], benchmark), best_step)
                              % (iter_,
                                 features[pred_idx[best_swap[1]]],
                                 features[pred_idx[best_swap[0]]],
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
            last_prediction = self._compute_prediction(predictors=bag_predictors, weights=weights)
            # last_prediction = np.zeros(nb_samp)
            # for i in range(len(weights)):
            #     last_prediction += weights[i] * bag_predictors[:, i]
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

    def _search_best_pair_swap_candidate(self, bag_predictors, bag_target, benchmark, step, weights):
        best_score = benchmark
        best_swap = (0, 0)
        best_step = None
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
                    if self._check_score_improvement(benchmark=best_score, score=candidate_score):
                        best_score = candidate_score
                        best_swap = (i, j)
                        best_step = the_step
        return best_score, best_step, best_swap

    def get_weights(self):
        return tuple(self.weights)


class BinaryClassificationLinearPredictorStacker(LinearPredictorStacker):
    """
    Binary Classification Linear Stacker computes the best weights to linearly merge predictors
    against a specific metric.

    Note that Stacker raw output is probability based.
    """

    def __init__(self,
                 metric=None,
                 maximize=False,
                 algorithm=LinearPredictorStacker.STANDARD,
                 colsample=1.0,
                 subsample=1.0,
                 n_bags=1,
                 max_iter=10,
                 verbose=0,
                 verb_round=1,
                 normed_weights=True,
                 eps=1e-5,
                 seed=None):
        super(BinaryClassificationLinearPredictorStacker, self).__init__(
            metric=metric,
            maximize=maximize,
            algorithm=algorithm,
            colsample=colsample,
            subsample=subsample,
            n_bags=n_bags,
            max_iter=max_iter,
            probabilities=True,
            verbose=verbose,
            verb_round=verb_round,
            normed_weights=normed_weights,
            eps=eps,
            seed=seed
        )

    def predict_proba(self, predictors=None):
        """Apply linear stacker to predictors and make sure probabilities are in [0, 1]"""
        return np.clip(self._transform(predictors), 1e-6, 1 - 1e-6)

    def predict(self, predictors=None, threshold=.5):
        """
        Apply linear stacker to predictors and then assign label against a threshold
        :param predictors: set of predictors to be merged
         number of predictors should be the same as the set used to train the stacker
        :param threshold: threshold used to assign binary label. defaults to .5
        :return: weighted sum of predictors
        """
        # Get probabilities
        probas = self.predict_proba(predictors)
        # Apply threshold to decide label
        probas[probas >= threshold] = 1
        probas[probas < threshold] = 0
        return probas


class BinaryRankingLinearPredictorStacker(LinearPredictorStacker):
    """
    Binary Ranking Linear Stacker computes the best weights to linearly merge predictors
    against a ranking metric.
    This is suitable for AUC or Gini

    Note that Stacker raw output is probability based.
    """

    def predict_proba(self, predictors=None):
        """
        Apply linear stacker to predictors and make sure probabilities are in [0, 1]
        Since we are in a ranking problem the output of the satcker can be out of [0, 1]
        In a classification problem using logloss for example probabilities are constraint in 0, 1
        But for AUC or Gini only ranking is important so we can use MinMaxScaler
        """
        skl = MinMaxScaler(feature_range=(1e-6, 1 - 1e-6))
        return skl.fit_transform(self._transform(predictors).reshape(-1, 1))[:, 0]

    def predict(self, predictors=None, threshold=.5):
        """
        Apply linear stacker to predictors and then assign label against a threshold
        :param predictors: set of predictors to be merged
         number of predictors should be the same as the set used to train the stacker
        :param threshold: threshold used to assign binary label. defaults to .5
        :return: weighted sum of predictors
        """
        # Get probabilities
        probas = self.predict_proba(predictors)
        # Apply threshold to decide label
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


class MultiLabelClassificationLinearPredictorStacker(object):
    """
    MultiLabelClassificationLinearPredictorStacker uses 1 versus rest algorithm
    There is one BinaryClassificationLinearPredictorStacker per class label
    """
    def __init__(self,
                 metric=None,
                 maximize=False,
                 colsample=1.0,
                 subsample=1.0,
                 n_bags=1,
                 max_iter=10,
                 verbose=0,
                 verb_round=1,
                 normed_weights=True,
                 eps=1e-5,
                 seed=None):

        if metric is None:
            raise ValueError('No metric has been provided')

        self.metric = metric
        self.algo = LinearPredictorStacker.SWAPPING
        self.maximize = maximize
        self.max_predictors = colsample
        self.max_samples = subsample
        self.n_bags = n_bags
        self.max_iter = max_iter
        self.verbose = verbose
        self.verb_round = verb_round
        self.normed_weights = normed_weights
        self.eps = eps
        self.seed = seed

        # Add multi class attributes
        self.labels = None
        self.labels_weights = None
        self.label_stackers = None

    # Overload fit method to fit
    def fit(self, predictors=None, target=None):
        """
        Method to fit all one vs all stackers
        :param predictors: set of predictors to be stacked
        :param target: target labels
        :return: None
        """
        # Check predictors and target
        # if (predictors is not None) and (target is not None):
        #     self._check_predictors_and_target(predictors, target)
        # else:
        #     # Predictors have been set using files
        #     if (self.predictors is None) | (self.target is None):
        #         raise ValueError('predictors and target must be set before fitting')

        # get labels ordered
        labeled_target = target.copy()
        self.labels = sorted(np.unique(labeled_target))

        # Go through labels for one vs all binary classification
        self.labels_weights = np.zeros((predictors.shape[1], len(self.labels)))
        for i_lab, label in enumerate(self.labels):
            # Make binary target for current label
            class_target = labeled_target.copy()
            class_target[class_target != label] = -1
            class_target[class_target == label] = 1
            class_target[class_target == -1] = 0

            # Istantiate a Binary Stacker
            stacker = BinaryClassificationLinearPredictorStacker(metric=self.metric,
                                                                 maximize=self.maximize,
                                                                 algorithm=self.algo,
                                                                 colsample=self.max_predictors,
                                                                 subsample=self.max_samples,
                                                                 n_bags=self.n_bags,
                                                                 max_iter=self.max_iter,
                                                                 verbose=self.verbose,
                                                                 verb_round=self.verb_round,
                                                                 seed=self.seed)
            # Fit stacker
            stacker.fit(predictors=predictors, target=class_target)
            # Keep label's weight
            self.labels_weights[:, i_lab] = stacker.get_weights()
            # Keep stacker
            if self.label_stackers is None:
                self.label_stackers = [stacker]
            else:
                self.label_stackers.append(stacker)

    def predict_proba(self, predictors=None):
        label_probas = np.zeros((len(predictors), len(self.labels)))
        for i_lab in range(len(self.labels)):
            label_probas[:, i_lab] = self.label_stackers[i_lab].predict_proba(predictors)
            # Return probas making sure everything sum to 1
        return label_probas / np.sum(label_probas, axis=1).reshape(-1, 1)

    def predict(self, predictors=None):
        # Compute probabilities
        label_probas = self.predict_proba(predictors=predictors)

        # For each row find the max
        the_max = np.argmax(label_probas, axis=1)
        return np.array([self.labels[x] for x in the_max])
