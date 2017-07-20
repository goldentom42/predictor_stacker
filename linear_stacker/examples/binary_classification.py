from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
import pandas as pd
import numpy as np
from linear_stacker import BinaryClassificationLinearPredictorStacker

pd.options.display.max_rows = 600


def sigmoid(x):
    return 1 / (1 + np.exp(- x))

# Load breast cancer dataset
dataset = load_breast_cancer()
# Split in data and target
X_full, y_full = dataset.data, dataset.target

classifiers = [
    ('ridge', RidgeClassifier(alpha=0.0001, normalize=True, random_state=0)),
    ('logit', LogisticRegression(C=0.01, random_state=1)),
    ('xtr', ExtraTreesClassifier(n_estimators=50, max_features=.4, max_depth=10, random_state=2, n_jobs=-1)),
    ('rfr', RandomForestClassifier(n_estimators=50, max_features=.2, max_depth=10, random_state=3, n_jobs=-1)),
    ('gbr', GradientBoostingClassifier(n_estimators=100, max_depth=2,learning_rate=.1,random_state=4))
]

kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Go through classifiers
oof_labels = np.zeros((len(X_full), len(classifiers)))
oof_probas = np.zeros((len(X_full), len(classifiers)))

for reg_i, (name, reg) in enumerate(classifiers):
    # compute out of fold (OOF) predictions
    for trn_idx, val_idx in kf.split(X_full):
        # Split data in training and validation sets
        trn_X, trn_Y = X_full[trn_idx], y_full[trn_idx]
        val_X, val_Y = X_full[val_idx], y_full[val_idx]
        # Fit the classifier
        reg.fit(trn_X, trn_Y)
        # Predict OOF data
        if hasattr(reg, 'predict_proba'):
            oof_probas[val_idx, reg_i] = reg.predict_proba(val_X)[:, 1]
        else:
            oof_probas[val_idx, reg_i] = sigmoid(reg.predict(val_X))
        oof_labels[val_idx, reg_i] = reg.predict(val_X)

    # Display OOF score
    print("Accuracy for classifier %6s : %.5f" % (name, accuracy_score(y_full, oof_labels[:, reg_i])))
    print("Log_loss for classifier %6s : %.5f" % (name, log_loss(y_full, oof_probas[:, reg_i])))


# Stacking using labels
print('Stacking using labels \n'
      '=====================')
print("\tLog loss Benchmark using labels' average : %.5f" % (log_loss(y_full, np.mean(oof_labels, axis=1))))

# Linear Stacker with labels, normed weights
stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss,
                                                     algo='standard',
                                                     max_iter=1000,
                                                     verbose=0,
                                                     normed_weights=True,
                                                     step=0.01)

stacker.fit(pd.DataFrame(oof_labels, columns=[name for (name, _) in classifiers]),
            pd.Series(y_full, name='target'))

print("\tLog loss Standard Linear Stacker (normed weights): %.5f"
      % (log_loss(y_full, stacker.predict_proba(oof_labels))))

# Linear Stacker with labels, no weight constraint
stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss,
                                                     algo='standard',
                                                     max_iter=1000,
                                                     verbose=0,
                                                     normed_weights=False,
                                                     step=0.01)

stacker.fit(pd.DataFrame(oof_labels, columns=[name for (name, _) in classifiers]),
            pd.Series(y_full, name='target'))

print("\tLog loss Standard Linear Stacker (no constraint): %.5f"
      % (log_loss(y_full, stacker.predict_proba(oof_labels))))

# Linear Stacker with labels normed weights swapping algo
stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss,
                                                     algo='swapping',
                                                     max_iter=1000,
                                                     verbose=0,
                                                     normed_weights=True,
                                                     step=0.01)

stacker.fit(pd.DataFrame(oof_labels, columns=[name for (name, _) in classifiers]),
            pd.Series(y_full, name='target'))

print("\tLog loss Swapping Linear Stacker (normed weights): %.5f"
      % (log_loss(y_full, stacker.predict_proba(oof_labels))))

# Linear Stacker with labels no weights constraints swapping algo
stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss,
                                                     algo='swapping',
                                                     max_iter=1000,
                                                     verbose=0,
                                                     normed_weights=False,
                                                     step=0.01)

stacker.fit(pd.DataFrame(oof_labels, columns=[name for (name, _) in classifiers]),
            pd.Series(y_full, name='target'))

print("\tLog loss Swapping Linear Stacker (no constraint): %.5f"
      % (log_loss(y_full, stacker.predict_proba(oof_labels))))

# Stacking using labels
print('Stacking using probabilities \n'
      '============================')
print("\tLog loss Benchmark using probas' average : %.5f" % (log_loss(y_full, np.mean(oof_probas, axis=1))))

# Linear Stacker with labels, normed weights
stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss,
                                                     algo='standard',
                                                     max_iter=1000,
                                                     verbose=0,
                                                     normed_weights=True,
                                                     step=0.01)

stacker.fit(pd.DataFrame(oof_probas, columns=[name for (name, _) in classifiers]),
            pd.Series(y_full, name='target'))

print("\tLog loss Standard Linear Stacker (normed weights): %.5f"
      % (log_loss(y_full, stacker.predict_proba(oof_probas))))

# Linear Stacker with labels, no weight constraint
stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss,
                                                     algo='standard',
                                                     max_iter=1000,
                                                     verbose=0,
                                                     normed_weights=False,
                                                     step=0.01)

stacker.fit(pd.DataFrame(oof_probas, columns=[name for (name, _) in classifiers]),
            pd.Series(y_full, name='target'))

print("\tLog loss Standard Linear Stacker (no constraint): %.5f"
      % (log_loss(y_full, stacker.predict_proba(oof_probas))))

# Linear Stacker with labels normed weights swapping algo
stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss,
                                                     algo='swapping',
                                                     max_iter=1000,
                                                     verbose=0,
                                                     normed_weights=True,
                                                     step=0.01)

stacker.fit(pd.DataFrame(oof_probas, columns=[name for (name, _) in classifiers]),
            pd.Series(y_full, name='target'))

print("\tLog loss Swapping Linear Stacker (normed weights): %.5f"
      % (log_loss(y_full, stacker.predict_proba(oof_probas))))

# Linear Stacker with labels no weights constraints swapping algo
stacker = BinaryClassificationLinearPredictorStacker(metric=log_loss,
                                                     algo='swapping',
                                                     max_iter=1000,
                                                     verbose=0,
                                                     normed_weights=False,
                                                     step=0.01)

stacker.fit(pd.DataFrame(oof_probas, columns=[name for (name, _) in classifiers]),
            pd.Series(y_full, name='target'))

print("\tLog loss Swapping Linear Stacker (no constraint): %.5f"
      % (log_loss(y_full, stacker.predict_proba(oof_probas))))
