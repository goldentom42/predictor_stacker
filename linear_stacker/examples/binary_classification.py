from sklearn.datasets import load_breast_cancer
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, f1_score
import pandas as pd
import numpy as np
from linear_stacker import BinaryClassificationLinearPredictorStacker
try:
    from sklearn.model_selection import KFold
except ImportError:
    from sklearn.cross_validation import KFold


def sigmoid(x):
    return 1 / (1 + np.exp(- x))


def get_folds(data):
    """returns correct folding generator for different versions of sklearn"""
    if sklearn_version.split('.')[1] == '18':
        # Module model_selection is in the distribution
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        return kf.split(data)
    else:
        # Module model_selection is not in the distribution
        kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=1)
        return kf


def main():
    pd.options.display.max_rows = 600

    # Load breast cancer dataset
    dataset = load_breast_cancer()
    # Split in data and target

    classifiers = [
        ('ridge', RidgeClassifier(alpha=0.0001, normalize=True, random_state=0)),
        ('logit', LogisticRegression(C=0.01, random_state=1)),
        ('xtr', ExtraTreesClassifier(n_estimators=50, max_features=.4, max_depth=10, random_state=2, n_jobs=-1)),
        ('rfr', RandomForestClassifier(n_estimators=50, max_features=.2, max_depth=10, random_state=3, n_jobs=-1)),
        ('gbr', GradientBoostingClassifier(n_estimators=100, max_depth=2,learning_rate=.1,random_state=4))
    ]

    # Go through classifiers
    oof_labels = np.zeros((len(dataset.data), len(classifiers)))
    oof_probas = np.zeros((len(dataset.data), len(classifiers)))

    for reg_i, (name, reg) in enumerate(classifiers):
        # compute out of fold (OOF) predictions
        for trn_idx, val_idx in get_folds(dataset.data):
            # Split data in training and validation sets
            trn_X, trn_Y = dataset.data[trn_idx], dataset.target[trn_idx]
            val_X = dataset.data[val_idx]
            # Fit the classifier
            reg.fit(trn_X, trn_Y)
            # Predict OOF data
            if hasattr(reg, 'predict_proba'):
                oof_probas[val_idx, reg_i] = reg.predict_proba(val_X)[:, 1]
            else:
                oof_probas[val_idx, reg_i] = sigmoid(reg.predict(val_X))
            oof_labels[val_idx, reg_i] = reg.predict(val_X)

        # Display OOF score
        print("Accuracy for classifier %6s : %.5f" % (name, accuracy_score(dataset.target, oof_labels[:, reg_i])))
        print("Log_loss for classifier %6s : %.5f" % (name, log_loss(dataset.target, oof_probas[:, reg_i])))
        print("Roc_auc  for classifier %6s : %.5f" % (name, roc_auc_score(dataset.target, oof_probas[:, reg_i])))

    # Stacking using labels
    print('Stacking using labels \n'
          '=====================')
    print("\tLog loss Benchmark using labels' average : %.5f" % (log_loss(dataset.target, np.mean(oof_labels, axis=1))))

    stackers = [
        # Linear Stacker with labels, normed weights
        ('Standard Linear Stacker (normed weights)',
         BinaryClassificationLinearPredictorStacker(metric=log_loss,
                                                    algo='standard',
                                                    max_iter=1000,
                                                    verbose=0,
                                                    normed_weights=True,
                                                    # step=0.01
                                                    )),
        # Linear Stacker with labels, no weight constraint
        ('Standard Linear Stacker (no constraint)',
         BinaryClassificationLinearPredictorStacker(metric=log_loss,
                                                    algo='standard',
                                                    max_iter=1000,
                                                    verbose=0,
                                                    normed_weights=False,
                                                    # step=0.01
                                                    )),
        # Linear Stacker with labels normed weights swapping algo
        ('Swapping Linear Stacker (normed weights)',
         BinaryClassificationLinearPredictorStacker(metric=log_loss,
                                                    algo='swapping',
                                                    max_iter=1000,
                                                    verbose=0,
                                                    normed_weights=True,
                                                    # step=0.01
                                                    )),
        # Linear Stacker with labels no weights constraints swapping algo
        ('Swapping Linear Stacker (no constraint)',
         BinaryClassificationLinearPredictorStacker(metric=log_loss,
                                                    algo='swapping',
                                                    max_iter=1000,
                                                    verbose=0,
                                                    normed_weights=False,
                                                    # step=0.01
                                                    ))
    ]

    for description, stacker in stackers:
        # Fit stacker
        stacker.fit(pd.DataFrame(oof_labels, columns=[name for (name, _) in classifiers]),
                    pd.Series(dataset.target, name='target'))
        # display results
        print("\tAccuracy  %s: %.5f"
              % (description, accuracy_score(dataset.target, stacker.predict(oof_labels))))
        print("\tF1_score  %s: %.5f"
              % (description, f1_score(dataset.target, stacker.predict(oof_labels))))
        print("\tLog loss  %s: %.5f"
              % (description, log_loss(dataset.target, stacker.predict_proba(oof_labels))))
        print("\tAUC score %s: %.5f"
              % (description, roc_auc_score(dataset.target, stacker.predict_proba(oof_labels))))

    # Stacking using labels
    print('Stacking using probabilities \n'
          '============================')
    print("\tLog loss Benchmark using probas' average : %.5f" % (log_loss(dataset.target, np.mean(oof_probas, axis=1))))

    for description, stacker in stackers:
        # Fit stacker
        stacker.fit(pd.DataFrame(oof_probas, columns=[name for (name, _) in classifiers]),
                    pd.Series(dataset.target, name='target'))
        # display results
        print("\tAccuracy  %s: %.5f"
              % (description, accuracy_score(dataset.target, stacker.predict(oof_probas))))
        print("\tF1_score  %s: %.5f"
              % (description, f1_score(dataset.target, stacker.predict(oof_probas))))
        print("\tLog loss  %s: %.5f"
              % (description, log_loss(dataset.target, stacker.predict_proba(oof_probas))))
        print("\tAUC score %s: %.5f"
              % (description, roc_auc_score(dataset.target, stacker.predict_proba(oof_probas))))

if __name__ == '__main__':
    main()
