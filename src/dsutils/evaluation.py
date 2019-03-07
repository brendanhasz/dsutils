"""Evaluation

* :func:`.permutation_importance`
* :func:`.permutation_importance_cv`

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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic




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



def optimize_cv(X, y, model, bounds,
                n_splits=3,
                max_time=None,
                max_evals=20,
                optimizer='bayesian',
                n_jobs=1,
                metric='mse'):
    """Optimize model parameters using cross-fold validation.

    Parameters
    ----------
    X : pandas DataFrame
        Independent variable values (features)
    y : pandas Series
        Dependent variable values (target)
    model : sklearn Pipeline
        Predictive model to optimize
    bounds : dict
        Parameter bounds.  A dict where the keys are the parameter names, and
        the values are the parameter bounds and type (as a tuple).  Each
        parameter name should be the Pipeline step string, then a double
        underscore, then the parameter name (see example below). Each 
        tuple should be (lower_bound, upper_bound, type), where type is int or
        float.
    n_splits : int
        Number of cross-validation folds.
    max_time : None or float
        Give up after this many seconds
    max_evals : int
        Max number of cross-validation evaluations to perform.
    optimizer : str
        Which method to use for optimization.  'random' to make random 
        parameter choices and take the best.  'bayesian' to use Gaussian
        process optimization.
    n_jobs : int
        Number of parallel jobs to run (for cross-validation).    
    metric : str or sklearn scorer
        What metric to use for evaluation.  One of:

        * 'r2' - coefficient of determination (maximize)
        * 'mse' - mean squared error (minimize)
        * 'mae' - mean absolute error (minimize)
        * 'accuracy' or 'acc' - accuracy (maximize)
        * 'auc' - area under the ROC curve (maximize)

    Returns
    -------
    opt_params : dict
        Optimal parameters.  Dict of the same format as bounds, except instead
        of tuples, the values contain the optimal parameter values.
    params : dict
        All the parameters which were evaluated.  Dict just like opt_params,
        except the values are vectors of all parameters which were evaluated.
    scores : ndarray
        Vector of scores for each evaluation.  Same size as each value in
        params

    Example
    -------

    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    model = Pipeline([
        ('pca', PCA(n_components=5)),
        ('regressor', Ridge(alpha=1.0))
    ])

    bounds = {
        'pca__n_components': [1, 100, int],
        'regressor__alpha': [0, 10, float]
    }

    opt_params, _, _ = optimize_cv(X, y, model, bounds)

    opt_params['pca']['n_components'] #optimal # components
    opt_params['regressor']['alpha'] #optimal alpha value
    """

    # Check inputs
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas DataFrame')
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError('y must be a pandas Series')
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same number of samples')
    if not isinstance(model, Pipeline):
        raise TypeError('model must be an sklearn Pipeline')
    if not isinstance(bounds, dict):
        raise TypeError('bounds must be a dict')
    if not isinstance(n_splits, int):
        raise TypeError('n_splits must be an integer')
    if n_splits < 1:
        raise ValueError('n_splits must be one or greater')
    if max_time is not None and not isinstance(max_time, float):
        raise TypeError('max_time must be None or a float')
    if max_time is not None and max_time < 0:
        raise ValueError('max_time must be positive')
    if not isinstance(max_evals, int):
        raise TypeError('max_evals must be an int')
    if max_evals < 1:
        raise ValueError('max_evals must be positive')
    if not isinstance(optimizer, str):
        raise TypeError('optimizer must be a string')
    if optimizer not in ['random', 'bayesian']:
        raise ValueError('optimizer must be \'random\' or \'bayesian\'')
    if not isinstance(n_jobs, int):
        raise TypeError('n_jobs must be an int')
    if n_jobs < 1:
        raise ValueError('n_jobs must be positive')

    # Create scorer
    if metric == 'r2':
        scorer = make_scorer(r2_score)
    elif metric == 'mse':
        scorer = make_scorer(mean_squared_error)
    elif metric == 'mae':
        scorer = make_scorer(mean_absolute_error)
    elif metric == 'accuracy' or metric == 'acc':
        scorer = make_scorer(accuracy_score)
    elif metric == 'auc':
        scorer = make_scorer(roc_auc_score)
    elif hasattr(metric, '__call__'):
        scorer = metric
    else:
        raise ValueError('metric must be a metric string or a callable')

    # Flip the score depending on the metric, such that lower is better
    if metric == 'mse' or metric == 'mae':
        flip = 1
    else:
        flip = -1

    # Create Gaussian process model
    if optimizer == 'bayesian':
        gp = GaussianProcessRegressor(kernel=RationalQuadratic(),
                                      n_restarts_optimizer=5,
                                      normalize_y=True)

    # Initialize arrays to store evaluated parameters
    scores = []
    params = dict()
    new_params = dict()
    for param in bounds:
        params[param] = []

    # Search for optimal parameters
    start_time = time.time()
    for i in range(max_evals):

        # Give up if we've spent too much time
        if max_time is not None and time.time()-start_time > max_time:
            break

        # Randomly choose next parameter values to try
        if optimizer == 'random':
            for param, bound in bounds.items():
                new_params[param] = np.random.uniform(bound[0], bound[1])

        # Bayesian optimizer
        else:
            pass
            # TODO

        # Convert integer params to integer
        for param, bound in bounds.items():
            if bound[2] is int:
                new_params[param] = round(new_params[param])

        # Modify model to use new parameters, and store values
        for param_name, val in new_params.items():
            step = param_name.split('__')[0]
            param = param_name.split('__')[1]
            model.named_steps[step].set_params(**{param: val})
            params[param_name].append(val)

        # Compute and store cross-validated metric
        t_score = cross_val_score(model, X, y, cv=n_splits, 
                                  scoring=scorer, n_jobs=n_jobs)
        scores.append(t_score.mean())

    # Get best evaluated parameters if using random optimization
    if optimizer == 'random':
        opt_params = dict()
        ix = scores.index(min(flip*e for e in scores)) #index of best score
        for param, vals in params.items():
            opt_params[param] = vals[ix]

    # Use Gaussian process to find max value if using Bayesian optimization
    else:
        pass
        # TODO

    # Return optimal parameters, all evaluated parameters, and scores
    return opt_params, params, scores

