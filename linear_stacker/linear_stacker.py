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
            if self.predictors is None:
                raise ValueError('predictors not set before fitting')
            if self.target is None:
                raise ValueError('target not provided before fitting')

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
        # Compute mean score
        self.mean_score = self.metric(self.target, self.predictors.mean(axis=1).values)

        # Create samples and predictors indexes
        samp_indexes = np.arange(self.predictors.shape[0])
        pred_indexes = np.arange(self.predictors.shape[1])

        # Set a seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)

        # Init weights
        full_weights = np.zeros(self.predictors.shape[1])

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

            # Init weights and prediction for current bag
            if self.normed_weights:
                weights = np.ones(nb_pred) * nb_pred
            else:
                weights = np.zeros(nb_pred)

            prediction = np.zeros(nb_samp)
            if self.normed_weights:
                for i, weight in enumerate(weights):
                    prediction += weight * bag_predictors[:, i] / (np.sum(weights))
            else:
                for i, weight in enumerate(weights):
                    prediction += weight * bag_predictors[:, i]

            # Set benchmark and print it
            benchmark = self.metric(bag_target, prediction)
            if self.verbose >= 2:
                print('Benchmark : ', benchmark)

            # Try to improve on the benchmark
            improve = True
            iter_ = 0
            while improve and (iter_ < self.max_iter):
                best_score = benchmark
                best_predictor = 0
                # Loop over all predictors and try to remove or add it
                for i in range(bag_predictors.shape[1]):
                    for the_sign in [self.step, -self.step]:
                        if self.normed_weights:
                            candidate = prediction * np.sum(weights) + the_sign * bag_predictors[:, i]
                            candidate /= (np.sum(weights) + the_sign)
                        else:
                            candidate = prediction + the_sign * bag_predictors[:, i]

                        candidate_score = self.metric(bag_target, candidate)
                        # print('New score for predictor ', str(i), ':', candidate_score)
                        if candidate_score < best_score:
                            best_score = candidate_score
                            best_predictor = i
                            sign_i = the_sign

                # Update benchmark if things have improved
                if best_score < benchmark and (benchmark - best_score) >= self.eps:
                    improve = True
                    # Modify prediction
                    if self.normed_weights:
                        prediction = prediction * np.sum(weights) + sign_i * bag_predictors[:, best_predictor]
                        prediction /= (np.sum(weights) + sign_i)
                    else:
                        prediction = (prediction + sign_i * bag_predictors[:, best_predictor])
                    # weights[best_predictor] += 1
                    weights[best_predictor] += sign_i
                    benchmark = self.metric(bag_target, prediction)
                    if self.verbose >= 2 and (iter_ % self.verb_round == 0):
                        print('Round %6d benchmark for feature %20s : %13.6f'
                              % (iter_, features[best_predictor], benchmark), sign_i)
                else:
                    if self.verbose >= 2:
                        print("Best round is %d" % iter_)
                    improve = False
                iter_ += 1

            # Now that weight have been found for the current bag
            # Update the bagged_prediction
            if self.normed_weights:
                weights /= np.sum(weights)

            last_prediction = np.zeros(nb_samp)
            for i in range(len(weights)):
                last_prediction += weights[i] * bag_predictors[:, i]
            if self.verbose >= 1:
                print('Bag ' + str(bag) + ' final score : ', self.metric(bag_target, last_prediction))
            full_weights[pred_idx] += weights / self.n_bags

        # Find bagged prediction
        bagged_prediction = np.zeros(self.predictors.shape[0])
        for i in range(len(full_weights)):
            bagged_prediction += full_weights[i] * predictors[:, i]
        if self.verbose >= 1:
            print('Final score : ', self.metric(self.target, bagged_prediction))

        self.score = self.metric(self.target, bagged_prediction)
        self.weights = full_weights

    def _transform(self, data):
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

        # Compute mean score
        self.mean_score = self.metric(self.target, self.predictors.mean(axis=1).values)

        # Create samples and predictors indexes
        samp_indexes = np.arange(self.predictors.shape[0])
        pred_indexes = np.arange(self.predictors.shape[1])

        # Set a seed
        np.random.seed(self.seed)

        # Init weights
        full_weights = np.zeros(self.predictors.shape[1])

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

            # Init weights
            weights = np.ones(bag_predictors.shape[1]) / bag_predictors.shape[1]

            # Compute initial prediction and benchmark
            prediction = np.zeros(nb_samp)
            for i, weight in enumerate(weights):
                prediction += weight * bag_predictors[:, i] / (np.sum(weights))

            # Set benchmark
            benchmark = self.metric(bag_target, prediction)
            if self.verbose >= 2:
                print('Benchmark : ', benchmark)

            # Try to improve on the benchmark
            init_step = 1 / bag_predictors.shape[1]
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
                            candidate_pred = np.zeros(bag_predictors.shape[0])
                            for n, weight in enumerate(candidate_weights_j):
                                candidate_pred += weight * bag_predictors[:, n]
                            candidate_score = self.metric(bag_target, candidate_pred)
                            # print(candidate_weights_j)
                            # print(candidate_score)
                            # Update best params
                            if candidate_score < best_score:
                                best_score = candidate_score
                                best_swap = (i, j)
                                best_step = the_step

                if best_score < benchmark:
                    improve = True
                    # Change weights
                    weights[best_swap[0]] += best_step
                    weights[best_swap[1]] -= best_step
                    # print("best score weight : ", weights)

                    prediction = np.zeros(bag_predictors.shape[0])
                    for n, weight in enumerate(weights):
                        prediction += weight * bag_predictors[:, n]
                    benchmark = self.metric(bag_target, prediction)
                    if self.verbose >= 2 and (iter_ % self.verb_round == 0):
                        print('Round %6d benchmark for feature %20s / %20s : %13.6f'
                              % (iter_, features[best_swap[0]], features[best_swap[1]], benchmark), best_step)
                else:
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

            # Iteration finished for current bag, update full_weights
            full_weights[pred_idx] += weights / self.n_bags

        # All bags done
        bagged_prediction = np.zeros(self.predictors.shape[0])
        for i in range(len(full_weights)):
            bagged_prediction += full_weights[i] * predictors[:, i]
        if self.verbose >= 1:
            print('Final score : ', self.metric(self.target, bagged_prediction))

        self.score = self.metric(self.target, bagged_prediction)
        self.weights = full_weights

    def get_weights(self):
        return tuple(self.weights)


class BinaryClassificationLinearPredictorStacker(LinearPredictorStacker):

    def predict_proba(self, data=None):
        return self._transform(data)

    def predict(self, data=None, threshold=.5):
        probas = self._transform(data)
        probas[probas >= threshold] = 1
        probas[probas < threshold] = 0
        return probas


class RegressionLinearPredictorStacker(LinearPredictorStacker):

    def predict(self, data=None):
        return self._transform(data)
