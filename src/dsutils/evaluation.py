"""Evaluation

* :func:`.permutation_importance`
* :func:`.permutation_importance_cv`
* :func:`.plot_permutation_importance`

"""



import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline



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
    importances = pd.DataFrame(np.zeros((1,X.shape[1])), columns=X.columns)
    for iC in X.columns:
        tC = X[iC].copy()
        X[iC] = X[iC].sample(frac=1, replace=True).values
        shuff_score = metric_func(y, estimator.predict(X))
        importances.loc[0,iC] = base_score - shuff_score
        X[iC] = tC

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
        t_imp = permutation_importance(X.iloc[test_ix,:].copy(),
                                       y.iloc[test_ix].copy(),
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



def cross_val_metric(model, X, y, cv=3, 
                     metric=mean_squared_error, 
                     train_subset=None, test_subset=None, 
                     shuffle=True, display=None):
    """Compute a cross-validated metric for a model.
    
    Parameters
    ----------
    model : sklearn estimator or callable
        Model to use for prediction.  Either an sklearn estimator (e.g. a 
        Pipeline), or a function which takes 3 arguments: 
        (X_train, y_train, X_test), and returns y_pred.  X_train and X_test
        should be pandas DataFrames, and y_train and y_pred should be 
        pandas Series.
    X : pandas DataFrame
        Features.
    y : pandas Series
        Target variable.
    cv : int
        Number of cross-validation folds
    metric : sklearn.metrics.Metric
        Metric to evaluate.
    train_subset : pandas Series (boolean)
        Subset of the data to train on. 
        Must be same size as y, with same index as X and y.
    test_subset : pandas Series (boolean)
        Subset of the data to test on.  
        Must be same size as y, with same index as X and y.
    shuffle : bool
        Whether to shuffle the data
    display : None or str
        Whether to print the cross-validated metric.
        If None, doesn't print.
    
    Returns
    -------
    list
        List of metrics for each test fold (length cv)
    """
    
    # Check types
    # TODO
    
    # Use all samples if not specified
    if train_subset is None:
        train_subset = y.copy()
        train_subset[:] = True
    if test_subset is None:
        test_subset = y.copy()
        test_subset[:] = True
    
    # Perform the cross-fold evaluation
    metrics = []
    TRix = y.copy()
    TEix = y.copy()
    kf = KFold(n_splits=cv, shuffle=shuffle)
    for train_ix, test_ix in kf.split(X):
        
        # Indexes for samples in training fold and in train_subset
        TRix[:] = False
        TRix.iloc[train_ix] = True
        TRix = TRix & train_subset
        
        # Indexes for samples in test fold and in test_subset
        TEix[:] = False
        TEix.iloc[test_ix] = True
        TEix = TEix & test_subset
        
        # Predict using a function
        if callable(model):
            preds = model(X.loc[TRix,:], y[TRix], X.loc[TEix,:])
        else:
            model.fit(X.loc[TRix,:], y[TRix])
            preds = model.predict(X.loc[TEix,:])
        
        # Store metric for this fold
        metrics.append(metric(y[TEix], preds))

    # Print the metric
    if display is not None:
        print('Cross-validated %s: %0.3f +/- %0.3f'
              % (display, metrics.mean(), metrics.std()))
        
    # Return a list of metrics for each fold
    return metrics
