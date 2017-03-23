import os.path
import pandas as pd
import numpy as np


class PredictorStacker:
    def __init__(self, metric=None, num_class=1):
        self.predictors = None
        self.target = None
        self.metric = metric
        self.score = None
        self.weights = None
        self.mean_score = None
        self.num_class = num_class


    def add_predictors(self, files=[]):
        """
        Add a list of file names to the stacker
        Files contain a minimum of 2 columns :
        - 1st column is the prediction (mandatory)
        - last column is the target (mandatory)
        If target column is not specified, target should be provided
        when fitting the stacker
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
        for i, f in enumerate(files):
            # Read file
            pred = pd.read_csv(f, delimiter=',')

            # Check number of columns is divided by num_class
            if (len(pred.columns) - 1) % self.num_class != 0:
                raise ValueError('file ' + f + ' does not contain enough classes')

            # Check data shape : should contain all classes + target
            if len(pred.shape) < self.num_class + 1:
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

    def set_predictors_and_target(self, predictors, target):
        # Check length
        if len(predictors) != len(target):
            raise ValueError('Target and predictors have different length')
        if len(target.shape) > 1:
            raise ValueError('Target contains more than one column')
        if (np.sum(predictors.isnull().sum()) > 0):
            raise ValueError('Predictors contain NaN')
        if (target.isnull().sum() > 0):
            raise ValueError('Target contain NaN')
        if (predictors.shape[1] % num_class != 0):
            raise ValueError('Predictors do not contain the expected number of classes')
        # Set predictors and target
        self.predictors = predictors
        self.target = np.array(target)

    def fit(self,
            sub_data=None,
            max_predictors=1.,
            max_samples=1.,
            n_bags=1,
            max_iter=10,
            verbose=0,
            verb_round=1,
            step=1,
            normed_weights=True,
            eps=1e-5,
            seed=24698537):
        # Checks
        if self.metric is None:
            raise ValueError('a metric should be set before fitting')
        if self.predictors is None:
            raise ValueError('predictors not set before fitting')
        if self.target is None:
            raise ValueError('target not provided before fitting')

        # TODO this needs to accept several classes
        # Compute mean score
        self.mean_score = self.metric(self.target, self.predictors.mean(axis=1).values)

        # Create samples and predictors indexes
        samp_indexes = np.arange(self.predictors.shape[0])
        pred_indexes = np.arange(self.predictors.shape[1])

        # Set a seed
        np.random.seed(seed)

        # Init weights
        full_weights = np.zeros(self.predictors.shape[1])

        # Run bagging
        predictors = self.predictors.values
        features = self.predictors.columns

        for bag in range(n_bags):
            # Shuffle indexes
            np.random.shuffle(samp_indexes)
            np.random.shuffle(pred_indexes)

            # Get ratioed predictors
            nb_pred = int(self.predictors.shape[1] * max_predictors)
            nb_samp = int(self.predictors.shape[0] * max_samples)
            pred_idx = pred_indexes[:nb_pred]
            samp_idx = samp_indexes[:nb_samp]

            bag_predictors = predictors[samp_idx, :][:, pred_idx]
            bag_target = self.target[samp_idx]

            # Init weights and prediction for current bag
            if normed_weights:
                weights = np.ones(nb_pred) * nb_pred
            else:
                weights = np.zeros(nb_pred)

            prediction = np.zeros(nb_samp)
            if normed_weights:
                for i, weight in enumerate(weights):
                    prediction += weight * bag_predictors[:, i] / (np.sum(weights))
            else:
                for i, weight in enumerate(weights):
                    prediction += weight * bag_predictors[:, i]

            # Set benchmark and print it
            benchmark = self.metric(bag_target, prediction)
            if verbose >= 2:
                print('Benchmark : ', benchmark)

            # Try to improve on the benchmark
            improve = True
            iter = 0
            while (improve and (iter < max_iter)):
                best_score = benchmark
                best_predictor = 0
                # Loop over all predictors and try to remove or add it
                for i in range(bag_predictors.shape[1]):
                    for the_sign in [step, -step]:
                        if normed_weights:
                            candidate = (prediction * np.sum(weights) + the_sign * bag_predictors[:, i]) / (
                            np.sum(weights) + the_sign)
                        else:
                            candidate = prediction + the_sign * bag_predictors[:, i]

                        candidate_score = self.metric(bag_target, candidate)
                        # print('New score for predictor ', str(i), ':', candidate_score)
                        if candidate_score < best_score:
                            best_score = candidate_score
                            best_predictor = i
                            sign_i = the_sign

                # Update benchmark if things have improved
                if best_score < benchmark and (benchmark - best_score) >= eps:
                    improve = True
                    # Modify prediction
                    if normed_weights:
                        prediction = (prediction * np.sum(weights) + sign_i * bag_predictors[:, best_predictor]) / (
                        np.sum(weights) + sign_i)
                    else:
                        prediction = (prediction + sign_i * bag_predictors[:, best_predictor])
                    # weights[best_predictor] += 1
                    weights[best_predictor] += sign_i
                    benchmark = self.metric(bag_target, prediction)
                    if verbose >= 2 and (iter % verb_round == 0):
                        print('Round %6d benchmark for feature %20s : %13.6f' % (
                        iter, features[best_predictor], benchmark), sign_i)
                else:
                    if verbose >= 2:
                        print("Best round is %d" % iter)
                    improve = False
                iter += 1

            # Now that weight have been found for the current bag
            # Update the bagged_prediction
            if normed_weights:
                weights = weights / np.sum(weights)

            last_prediction = np.zeros(nb_samp)
            for i in range(len(weights)):
                last_prediction += weights[i] * bag_predictors[:, i]
            if verbose >= 1:
                print('Bag ' + str(bag) + ' final score : ', self.metric(bag_target, last_prediction))
            full_weights[pred_idx] += weights / n_bags

        # Find bagged prediction
        bagged_prediction = np.zeros(self.predictors.shape[0])
        for i in range(len(full_weights)):
            bagged_prediction += full_weights[i] * predictors[:, i]
        if verbose >= 1:
            print('Final score : ', self.metric(self.target, bagged_prediction))

        diff = last_prediction - bagged_prediction[samp_idx]

        diff = bag_target - self.target[samp_idx]

        self.score = self.metric(self.target, bagged_prediction)
        self.weights = full_weights

    def transform(self, data):
        prediction = np.zeros(len(data))
        tmp = np.array(data)
        for i, w in enumerate(self.weights):
            prediction += w * tmp[:, i]
        return prediction

    def fit_swapping(self,
                     sub_data=None,
                     max_predictors=1.,
                     max_samples=1.,
                     n_bags=1,
                     max_iter=10,
                     verbose=0,
                     verb_round=1,
                     eps=1e-5,
                     seed=24698537):

        # Checks
        if self.metric is None:
            raise ValueError('a metric should be set before fitting')
        if self.predictors is None:
            raise ValueError('predictors not set before fitting')
        if self.target is None:
            raise ValueError('target not provided before fitting')

        # Compute mean score
        self.mean_score = self.metric(self.target, self.predictors.mean(axis=1).values)

        # Create samples and predictors indexes
        samp_indexes = np.arange(self.predictors.shape[0])
        pred_indexes = np.arange(self.predictors.shape[1])

        # Set a seed
        np.random.seed(seed)

        # Init weights
        full_weights = np.zeros(self.predictors.shape[1])

        # Run bagging
        predictors = self.predictors.values
        features = self.predictors.columns

        for bag in range(n_bags):
            # Shuffle indexes
            np.random.shuffle(samp_indexes)
            np.random.shuffle(pred_indexes)

            # Get ratioed predictors
            nb_pred = int(self.predictors.shape[1] * max_predictors)
            nb_samp = int(self.predictors.shape[0] * max_samples)
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
            if verbose >= 2:
                print('Benchmark : ', benchmark)

            # Try to improve on the benchmark
            init_step = 1 / bag_predictors.shape[1]
            step = init_step
            improve = True
            iter = 0
            while (improve and (iter < max_iter)):
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
                    if verbose >= 2 and (iter % verb_round == 0):
                        print('Round %6d benchmark for feature %20s / %20s : %13.6f'
                              % (iter, features[best_swap[0]], features[best_swap[1]], benchmark), best_step)
                else:
                    if step == init_step / 100:
                        improve = False
                        if verbose >= 2:
                            print("Best round is %d" % iter)
                    elif step == init_step / 10:
                        step = init_step / 10
                    else:
                        step = init_step / 100
                # Increment iteration
                iter += 1

            # Print current bag score
            last_prediction = np.zeros(nb_samp)
            for i in range(len(weights)):
                last_prediction += weights[i] * bag_predictors[:, i]
            if verbose >= 1:
                print('Bag ' + str(bag) + ' final score : ', self.metric(bag_target, last_prediction))
            print(weights)
            # Iteration finished for current bag, update full_weights
            full_weights[pred_idx] += weights / n_bags

        # All bags done
        bagged_prediction = np.zeros(self.predictors.shape[0])
        for i in range(len(full_weights)):
            bagged_prediction += full_weights[i] * predictors[:, i]
        if verbose >= 1:
            print('Final score : ', self.metric(self.target, bagged_prediction))

        diff = last_prediction - bagged_prediction[samp_idx]

        diff = bag_target - self.target[samp_idx]

        self.score = self.metric(self.target, bagged_prediction)
        self.weights = full_weights
