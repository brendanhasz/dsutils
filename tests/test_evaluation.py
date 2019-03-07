"""Tests evaluation

"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression, make_classification
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import median_absolute_error, jaccard_similarity_score
from sklearn.decomposition import PCA

from dsutils.evaluation import permutation_importance
from dsutils.evaluation import permutation_importance_cv
from dsutils.evaluation import plot_permutation_importance
from dsutils.evaluation import optimize_cv



def test_permutation_importance_regression(plot):
    """Tests evaluation.permutation_importance w/ regression problems"""

    # Make dummy regression dataset
    N = 1000
    X, y = make_regression(n_samples=N,
                           n_features=5,
                           n_informative=2,
                           n_targets=1,
                           random_state=12345,
                           shuffle=False)
    df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e'])
    df['y'] = y
    X_train = df.loc[:N/2-1, ['a', 'b', 'c', 'd', 'e']].copy()
    y_train = df.loc[:N/2-1, 'y'].copy()
    X_test = df.loc[N/2:, ['a', 'b', 'c', 'd', 'e']].copy()
    y_test = df.loc[N/2:, 'y'].copy()

    # Create a regression pipeline
    reg_pipe = Pipeline([
        ('regressor', LinearRegression()),
    ])

    # Fit on first half of samples
    reg_pipe.fit(X_train, y_train)

    # Permutation importance w/ R^2
    imp_df = permutation_importance(X_test, y_test, reg_pipe, 'r2')

    if plot:
        plot_permutation_importance(imp_df)
        plt.title('a and b should be most important')
        plt.show()

    # Permutation importance w/ mean squared error
    imp_df = permutation_importance(X_test, y_test, reg_pipe, 'mse')

    # Permutation importance w/ mean absolute error
    imp_df = permutation_importance(X_test, y_test, reg_pipe, 'mae')

    # Permutation importance w/ custom metric function (median absolute error)
    imp_df = permutation_importance(X_test, y_test, reg_pipe,
                                    lambda t, p: -median_absolute_error(t, p))



def test_permutation_importance_classification():
    """Tests evaluation.permutation_importance w/ classification problems"""

    # Make dummy classification dataset
    N = 1000
    X, y = make_classification(n_samples=N,
                               n_features=5,
                               n_informative=2,
                               n_redundant=0,
                               n_clusters_per_class=1,
                               random_state=12345,
                               shuffle=False)
    df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e'])
    df['y'] = y
    X_train = df.loc[:N/2-1, ['a', 'b', 'c', 'd', 'e']].copy()
    y_train = df.loc[:N/2-1, 'y'].copy()
    X_test = df.loc[N/2:, ['a', 'b', 'c', 'd', 'e']].copy()
    y_test = df.loc[N/2:, 'y'].copy()

    # Create a classification pipeline
    class_pipe = Pipeline([
        ('classifier', LogisticRegression(solver='lbfgs')),
    ])

    # Fit on first half of samples
    class_pipe.fit(X_train, y_train)

    # Permutation importance w/ accuracy
    imp_df = permutation_importance(X_test, y_test, class_pipe, 'accuracy')

    # Permutation importance w/ area under the ROC curve
    imp_df = permutation_importance(X_test, y_test, class_pipe, 'auc')

    # Permutation importance w/ custom metric function (jaccard similarity)
    imp_df = permutation_importance(X_test, y_test, class_pipe,
                                    lambda t,p: jaccard_similarity_score(t,p))



def test_permutation_importance_cv():
    """Tests evaluation.permutation_importance_cv"""

    # Make dummy regression dataset
    N = 1000
    X, y = make_regression(n_samples=N,
                           n_features=5,
                           n_informative=4,
                           n_targets=1,
                           random_state=2,
                           shuffle=False)
    df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e'])
    df['y'] = y
    X = df.loc[:, ['a', 'b', 'c', 'd', 'e']].copy()
    y = df.loc[:, 'y'].copy()

    # Create a regression pipeline
    reg_pipe = Pipeline([
        ('regressor', LinearRegression()),
    ])

    # Compute cross-validated feature importance
    imp_df = permutation_importance_cv(X, y, reg_pipe, 'mse')



def test_plot_permutation_importance(plot):
    """Tests evaluation.plot_permutation_importance"""

    # Make dummy regression dataset
    N = 1000
    X, y = make_regression(n_samples=N,
                           n_features=5,
                           n_informative=4,
                           n_targets=1,
                           random_state=2,
                           shuffle=False)
    df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e'])
    df['y'] = y
    X = df.loc[:, ['a', 'b', 'c', 'd', 'e']].copy()
    y = df.loc[:, 'y'].copy()

    # Create a regression pipeline
    reg_pipe = Pipeline([
        ('regressor', LinearRegression()),
    ])

    # Compute cross-validated feature importance
    imp_df = permutation_importance_cv(X, y, reg_pipe, 'mse')

    plot_permutation_importance(imp_df)
    if plot:
        plt.show()



def test_optimize_cv_random_1(plot):
    """Tests evaluation.optimize_cv w/ regression problems"""

    # Make dummy regression dataset
    N = 1000
    X, y = make_regression(n_samples=N,
                           n_features=5,
                           n_informative=2,
                           n_targets=1,
                           random_state=12345,
                           shuffle=False,
                           noise=1.0)
    df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e'])
    df['y'] = y
    X_train = df.loc[:N/2-1, ['a', 'b', 'c', 'd', 'e']].copy()
    y_train = df.loc[:N/2-1, 'y'].copy()
    X_test = df.loc[N/2:, ['a', 'b', 'c', 'd', 'e']].copy()
    y_test = df.loc[N/2:, 'y'].copy()

    # Create a regression pipeline
    model = Pipeline([
        ('regressor', Ridge(alpha=1.0)),
    ])

    # Define bounds
    bounds = {
        'regressor__alpha': [0.0, 2.0, float]
    }

    # Find best parameters
    opt_params, params, scores = optimize_cv(X_train, y_train, model, bounds,
                                             n_splits=3,
                                             max_time=None,
                                             max_evals=10,
                                             optimizer='random',
                                             metric='mse')

    # Plot parameters and their scores
    if plot:
        plt.plot(params['regressor__alpha'], scores, '.')
        plt.xlabel('regressor__alpha')
        plt.ylabel('mse')
        plt.show()

    # Check n_jobs arg works
    opt_params, params, scores = optimize_cv(X_train, y_train, model, bounds,
                                             n_splits=3,
                                             max_time=None,
                                             max_evals=10,
                                             optimizer='random',
                                             metric='mse',
                                             n_jobs=2)

    # Check timeout works
    start_time = time.time()
    opt_params, params, scores = optimize_cv(X_train, y_train, model, bounds,
                                             n_splits=3,
                                             max_time=0.2,
                                             max_evals=int(1e12),
                                             optimizer='random',
                                             metric='mse')
    assert time.time()-start_time < 1.0





def test_optimize_cv_random_2(plot):
    """Tests evaluation.optimize_cv w/ 2 parameters"""

    # Make dummy regression dataset
    N = 1000
    X, y = make_regression(n_samples=N,
                           n_features=5,
                           n_informative=2,
                           n_targets=1,
                           random_state=12345,
                           shuffle=False,
                           noise=1.0)
    df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e'])
    df['y'] = y
    X_train = df.loc[:N/2-1, ['a', 'b', 'c', 'd', 'e']].copy()
    y_train = df.loc[:N/2-1, 'y'].copy()
    X_test = df.loc[N/2:, ['a', 'b', 'c', 'd', 'e']].copy()
    y_test = df.loc[N/2:, 'y'].copy()

    # Create a regression pipeline
    model = Pipeline([
        ('regressor', ElasticNet(alpha=1.0, l1_ratio=0.5)),
    ])

    # Define bounds
    bounds = {
        'regressor__alpha': [1e-4, 2.0, float],
        'regressor__l1_ratio': [0.0, 1.0, float]
    }

    # Find best parameters
    opt_params, params, scores = optimize_cv(X_train, y_train, model, bounds,
                                             n_splits=3,
                                             max_time=None,
                                             max_evals=10,
                                             optimizer='random',
                                             metric='mse')

    # Plot parameters and their scores
    if plot:
        plt.scatter(params['regressor__alpha'],
                    params['regressor__l1_ratio'],
                    c=scores, alpha=0.7)
        plt.xlabel('regressor__alpha')
        plt.ylabel('regressor__l1_ratio')
        plt.show()



def test_optimize_cv_random_int(plot):
    """Tests evaluation.optimize_cv w/ int parameter types"""

    # Make dummy regression dataset
    N = 1000
    X, y = make_regression(n_samples=N,
                           n_features=5,
                           n_informative=2,
                           n_targets=1,
                           random_state=12345,
                           shuffle=False,
                           noise=1.0)
    df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e'])
    df['y'] = y
    X_train = df.loc[:N/2-1, ['a', 'b', 'c', 'd', 'e']].copy()
    y_train = df.loc[:N/2-1, 'y'].copy()
    X_test = df.loc[N/2:, ['a', 'b', 'c', 'd', 'e']].copy()
    y_test = df.loc[N/2:, 'y'].copy()

    # Create a regression pipeline
    model = Pipeline([
        ('pca', PCA(n_components=5)),
        ('regressor', Ridge(alpha=1.0)),
    ])

    # Define bounds
    bounds = {
        'pca__n_components': [1, 5, int],
        'regressor__alpha': [0.0, 4.0, float]
    }

    # Find best parameters
    opt_params, params, scores = optimize_cv(X_train, y_train, model, bounds,
                                             n_splits=3,
                                             max_time=None,
                                             max_evals=10,
                                             optimizer='random',
                                             metric='mse')

    # Plot parameters and their scores
    if plot:
        plt.scatter(params['pca__n_components'],
                    params['regressor__alpha'],
                    c=scores, alpha=0.7)
        plt.xlabel('pca__n_components')
        plt.ylabel('regressor__alpha')
        plt.show()
