from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from linear_stacker import RegressionLinearPredictorStacker
try:
    from sklearn.model_selection import KFold
except ImportError:
    from sklearn.cross_validation import KFold


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

    # Load boston dataset
    dataset = load_boston()
    # Split in data and target
    X_full, y_full = dataset.data, dataset.target

    regressors = [
        ('ridge', Ridge(alpha=0.001, normalize=True, random_state=0)),
        ('lasso', Lasso(alpha=0.01, normalize=True, random_state=1)),
        ('xtr', ExtraTreesRegressor(n_estimators=50, max_features=.4, max_depth=10, random_state=2, n_jobs=-1)),
        ('rfr', RandomForestRegressor(n_estimators=50, max_features=.2, max_depth=10, random_state=3, n_jobs=-1)),
        ('gbr', GradientBoostingRegressor(n_estimators=100, max_depth=2,learning_rate=.1,random_state=4))
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    # Go through classifiers
    oof_preds = np.zeros((len(X_full), len(regressors)))
    for reg_i, (name, reg) in enumerate(regressors):
        # compute out of fold (OOF) predictions
        for trn_idx, val_idx in get_folds(X_full):
            # Split data in training and validation sets
            trn_X, trn_Y = X_full[trn_idx], y_full[trn_idx]
            val_X, val_Y = X_full[val_idx], y_full[val_idx]
            # Fit the regressor
            reg.fit(trn_X, trn_Y)
            # Predict OOF data
            oof_preds[val_idx, reg_i] = reg.predict(val_X)
        # Display OOF score
        print("MSE for regressor %6s : %.5f" % (name, mean_squared_error(y_full, oof_preds[:, reg_i])))


    # First test using standard algorithm
    stacker = RegressionLinearPredictorStacker(metric=mean_squared_error,
                                               algo='standard',
                                               max_iter=100,
                                               verbose=0,
                                               normed_weights=True)

    stacker.fit(pd.DataFrame(oof_preds, columns=[name for (name, _) in regressors]),
                pd.Series(y_full, name='target'))

    print("Standard stacker score with normed weights : %.5f" % (mean_squared_error(y_full, stacker.predict(oof_preds))))

    # Second test using non normed weights
    stacker = RegressionLinearPredictorStacker(metric=mean_squared_error,
                                               algo='standard',
                                               max_iter=100,
                                               verbose=0,
                                               normed_weights=False,
                                               step=.05)

    stacker.fit(pd.DataFrame(oof_preds, columns=[name for (name, _) in regressors]),
                pd.Series(y_full, name='target'))

    print("Standard stacker score without normed weights : %.5f" % (mean_squared_error(y_full, stacker.predict(oof_preds))))

    stacker = RegressionLinearPredictorStacker(metric=mean_squared_error,
                                               algo='swapping',
                                               max_iter=100,
                                               verbose=0,
                                               normed_weights=True,
                                               step=1)

    stacker.fit(pd.DataFrame(oof_preds, columns=[name for (name, _) in regressors]),
                pd.Series(y_full, name='target'))

    print("Swapping stacker score without normed weights : %.5f" % (mean_squared_error(y_full, stacker.predict(oof_preds))))

if __name__ == '__main__':
    main()