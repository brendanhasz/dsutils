"""Tests optimization

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

from dsutils.optimization import GaussianProcessOptimizer
from dsutils.optimization import optimize_params_cv


def test_GaussianProcessOptimizer_1d(plot):
    """Tests optimization.GaussianProcessOptimizer"""

    # Create the optimizer object
    gpo = GaussianProcessOptimizer(lb=[0], ub=[10], parameters='x')

    # Add points
    xx = np.linspace(2, 8, 20)
    yy = np.sin(xx*2)+0.5*np.random.randn(20)
    for iP in range(len(xx)):
        gpo.add_point([xx[iP]], [yy[iP]])

    # Show the loss surface
    gpo.plot_surface('x')
    if plot:
        plt.show()

    # Draw a random point
    new_point = gpo.random_point()
    assert isinstance(new_point, list)
    assert len(new_point) == 1


"""
def test_optimize_cv_random_1(plot):
    #Tests evaluation.optimize_cv w/ regression problems

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
                                             n_random=10,
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
                                             n_random=10,
                                             metric='mse',
                                             n_jobs=2)

    # Check timeout works
    start_time = time.time()
    opt_params, params, scores = optimize_cv(X_train, y_train, model, bounds,
                                             n_splits=3,
                                             max_time=0.2,
                                             max_evals=int(1e12),
                                             n_random=int(1e12),
                                             metric='mse')
    assert time.time()-start_time < 1.0





def test_optimize_cv_random_2(plot):
    #Tests evaluation.optimize_cv w/ 2 parameters

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
                                             n_random=10,
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
    #Tests evaluation.optimize_cv w/ int parameter types

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
                                             n_random=10,
                                             metric='mse')

    # Plot parameters and their scores
    if plot:
        plt.scatter(params['pca__n_components'],
                    params['regressor__alpha'],
                    c=scores, alpha=0.7)
        plt.xlabel('pca__n_components')
        plt.ylabel('regressor__alpha')
        plt.show()
"""