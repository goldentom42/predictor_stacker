from sklearn import datasets
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score, f1_score
import pandas as pd
import numpy as np
from linear_stacker import MultiLabelClassificationLinearPredictorStacker
try:
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    from sklearn.cross_validation import StratifiedKFold


def get_folds(data, target):
    """returns correct folding generator for different versions of sklearn"""
    if sklearn_version.split('.')[1] == '18':
        # Module model_selection is in the distribution
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        return kf.split(data, target)
    else:
        # Module model_selection is not in the distribution
        kf = StratifiedKFold(y=target, n_folds=5, shuffle=True, random_state=1)
        return kf


def main():
    pd.options.display.max_rows = 600

    # Load iris dataset
    dataset = datasets.load_iris()
    n_classes = len(np.unique(dataset.target))

    classifiers = [
        ('logit', LogisticRegression(C=0.1,
                                     random_state=1)),
        ('xtr', ExtraTreesClassifier(n_estimators=50,
                                     max_features=.4,
                                     max_depth=10,
                                     random_state=2,
                                     n_jobs=-1)),
        ('rfr', RandomForestClassifier(n_estimators=50,
                                       max_features=.2,
                                       max_depth=10,
                                       random_state=3,
                                       n_jobs=-1)),
        ('gbr', GradientBoostingClassifier(n_estimators=20,
                                           max_depth=3,
                                           subsample=.6,
                                           learning_rate=.01,
                                           random_state=4))
    ]

    # Go through classifiers
    oof_labels = np.zeros((len(dataset.data), len(classifiers)))
    oof_probas = np.zeros((len(dataset.data), len(classifiers) * n_classes))

    for reg_i, (name, reg) in enumerate(classifiers):
        # compute out of fold (OOF) predictions
        for trn_idx, val_idx in get_folds(dataset.data, dataset.target):
            # Split data in training and validation sets
            trn_X, trn_Y = dataset.data[trn_idx], dataset.target[trn_idx]
            val_X = dataset.data[val_idx]
            # Fit the classifier
            reg.fit(trn_X, trn_Y)
            # Predict OOF data
            oof_probas[val_idx, reg_i * n_classes: (reg_i + 1) * n_classes] = reg.predict_proba(val_X)
            oof_labels[val_idx, reg_i] = reg.predict(val_X)

        # Display OOF score
        print("Accuracy for classifier %6s : %.5f"
              % (name, accuracy_score(dataset.target, oof_labels[:, reg_i])))
        print("F1_score for classifier %6s : %.5f"
              % (name, f1_score(dataset.target, oof_labels[:, reg_i], average='micro')))
        print("Log_loss for classifier %6s : %.5f"
              % (name, log_loss(dataset.target, oof_probas[:, reg_i * n_classes: (reg_i + 1) * n_classes])))

    # Use stacker
    stacker = MultiLabelClassificationLinearPredictorStacker(metric=log_loss, verbose=0, max_iter=100)
    stacker.fit(predictors=oof_probas, target=dataset.target)
    print("Resulted log_loss : ", log_loss(dataset.target, stacker.predict_proba(oof_probas)))
    print("Resulted F1-score : ", f1_score(dataset.target, stacker.predict(oof_probas), average='macro'))
    print("Resulted accuracy : ", accuracy_score(dataset.target, stacker.predict(oof_probas)))

if __name__ == '__main__':
    main()