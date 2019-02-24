"""Evaluation

* :func:`.permutation_importance`
* :func:`.permutation_importance_cv`

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold



def permutation_importance(X, y, estimator, metric):
    """Compute permutation-based feature importance on validation data.
    
    Parameters
    ----------
    X : pandas DataFrame of size (Nsamples, Nfeatures)
        Features
    y : pandas Series or numpy ndarray of size (Nsamples,)
        Target
    estimator : sklearn estimator
        A fit model to use for prediction.
    metric : callable
        Metric to use for computing feature importance.  Larger should be 
        better.  If a callable, should take two arguments ``y_true`` and 
        ``y_pred`` and return a loss, where larger values indicate better
        performance.  If a string, possible values are:

        * ``'r2'`` - coefficient of variation (for regressors)
        * ``'mse'`` - mean squared error (for regressors)
        * ``'mae'`` - mean absolute error (for regressors)
        * ``'accuracy'`` or ``'acc'`` - accuracy (for classifiers)
        * ``'auc'`` - area under the ROC curve (for classifiers)

    Returns
    -------
    pandas DataFrame
        Importance scores of each feature.  Of size (1,Nfeatures)
    """

    # Check inputs
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas DataFrame')
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError('y must be a pandas Series or numpy vector')
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same number of samples')
    if not isinstance(estimator, BaseEstimator):
        raise TypeError('estimator must be an sklearn estimator')

    # Determine metric to use
    if metric == 'r2':
        metric_func = lambda t, p: r2_score(t, p)
    elif metric == 'mse':
        metric_func = lambda t, p: -mean_squared_error(t, p)
    elif metric == 'mae':
        metric_func = lambda t, p: -mean_absolute_error(t, p)
    elif metric == 'accuracy' or metric == 'acc':
        metric_func = lambda t, p: accuracy_score(t, p)
    elif metric == 'auc':
        metric_func = lambda t, p: roc_auc_score(t, p)
    elif hasattr(metric, '__call__'):
        metric_func = metric
    else:
        raise ValueError('metric must be a metric string or a callable')

    # Baseline performance
    base_score = metric_func(y, estimator.predict(X))

    # Permute each column and compute drop in metric
    Xv = X.values
    yv = y.values
    importances = pd.DataFrame(np.zeros((1,X.shape[1])), columns=X.columns)
    for iC in range(X.shape[1]):
        tC = np.copy(Xv[:,iC])
        np.random.shuffle(Xv[:,iC])
        shuff_score = metric_func(yv, estimator.predict(Xv))
        importances.loc[0,X.columns[iC]] = base_score - shuff_score
        Xv[:,iC] = tC

    # Return df with the feature importances
    return importances



def permutation_importance_cv(X, y, estimator, metric, 
                              n_splits=3, shuffle=True):
    """Compute cross-validated permutation-based feature importance.
    
    Parameters
    ----------
    X : pandas DataFrame
        Features
    y : pandas Series or numpy ndarray
        Target
    estimator : sklearn estimator
        Model to use for prediction.  For example, a pipeline object.
    metric : callable
        Metric to use for computing feature importance.  Larger should be 
        better.  If a callable, should take two arguments ``y_true`` and 
        ``y_pred`` and return a loss, where larger values indicate better
        performance.  If a string, possible values are:

        * ``'r2'`` - coefficient of variation (for regressors)
        * ``'mse'`` - mean squared error (for regressors)
        * ``'mae'`` - mean absolute error (for regressors)
        * ``'accuracy'`` - accuracy (for classifiers)
        * ``'auc'`` - area under the ROC curve (for classifiers)

    n_splits : int
        Number of cross-validation splits.  Default = 2.
    shuffle : bool
        Whether to shuffle when splitting into CV folds.  Default = True.

    Returns
    -------
    pandas DataFrame
        Importance scores of each feature for each cross-validation fold.
        Of size (n_splits, Nfeatures)
    """

    # Check inputs
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas DataFrame')
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError('y must be a pandas Series or numpy vector')
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same number of samples')
    if not isinstance(estimator, BaseEstimator):
        raise TypeError('estimator must be an sklearn estimator')
    if not isinstance(n_splits, int):
        raise TypeError('n_splits must be an integer')
    if n_splits < 1:
        raise ValueError('n_splits must be 1 or greater')
    if not isinstance(shuffle, bool):
        raise TypeError('shuffle must be True or False')

    # Compute feature importances for each fold
    importances = pd.DataFrame(np.zeros((n_splits,X.shape[1])),
                               columns=X.columns)
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    iF = 0
    for train_ix, test_ix in kf.split(X):
        t_est = clone(estimator)
        t_est.fit(X.iloc[train_ix,:], y.iloc[train_ix])
        t_imp = permutation_importance(X.iloc[test_ix,:], y.iloc[test_ix], 
                                       t_est, metric)
        importances.loc[iF,:] = t_imp.loc[0,:]
        iF += 1

    # Return df with the feature importances for each fold
    return importances



def plot_permutation_importance(importances):
    """Plot importance score of each feature.

    Parameters
    ----------
    importances : pandas DataFrame
        Importance scores for each feature.  Should be of shape 
        (Nfolds,Nfeatures).
    """
    df = pd.melt(importances, var_name='Feature', value_name='Importance')
    dfg = (df.groupby(['Feature'])['Importance']
           .aggregate(np.median)
           .reset_index()
           .sort_values('Importance', ascending=False))
    sns.barplot(x='Importance', y='Feature', data=df, order=dfg['Feature'])
