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
from dsutils.optimization import optimize_params



def test_GaussianProcessOptimizer_1d(plot):
    """Tests optimization.GaussianProcessOptimizer"""

    # Create the optimizer object
    gpo = GaussianProcessOptimizer(lb=[0], ub=[10], param_names='x')

    # Add points
    xx = np.linspace(1, 9, 20)
    yy = np.sin(xx*2)+0.3*np.random.randn(20) + 10
    for iP in range(len(xx)):
        gpo.add_point([xx[iP]], [yy[iP]])

    # Draw a random point
    new_point = gpo.random_point()
    assert isinstance(new_point, list)
    assert len(new_point) == 1
    assert isinstance(new_point[0], float)

    # Get the suggested next point to sample
    new_point = gpo.next_point()
    assert isinstance(new_point, list)
    assert len(new_point) == 1
    assert isinstance(new_point[0], float)

    # Get the expected best point
    best_point = gpo.best_point()
    assert isinstance(best_point, list)
    assert len(best_point) == 1
    assert isinstance(best_point[0], float)

    # Best point so far
    best_point = gpo.best_point(expected=False)
    assert isinstance(best_point, list)
    assert len(best_point) == 1
    assert isinstance(best_point[0], float)

    # Best point as a dict
    best_point = gpo.best_point(get_dict=True)
    assert isinstance(best_point, dict)
    assert len(best_point) == 1
    assert isinstance(best_point['x'], float)

    # Get x
    x = gpo.get_x()
    assert isinstance(x, list)
    assert len(x) == 20

    # Get x as a dict
    x = gpo.get_x(get_dict=True)
    assert isinstance(x, dict)
    assert len(x) == 1
    assert len(x['x']) == 20

    # Get y
    y = gpo.get_y()
    assert isinstance(y, list)
    assert len(y) == 20

    # Show the loss surface
    gpo.plot_surface('x')
    if plot:
        plt.show()

    # Should also work w/o specifying what parameters to plot
    gpo.plot_surface()
    if plot:
        plt.show()

    # Show the expected improvement surface
    gpo.plot_ei_surface()
    if plot:
        plt.show()

    # Plot both surfaces
    gpo.plot_surfaces()
    if plot:
        plt.show()



def test_GaussianProcessOptimizer_int(plot):
    """Tests optimization.GaussianProcessOptimizer"""

    # Create the optimizer object
    gpo = GaussianProcessOptimizer(lb=[0], ub=[20], 
                                   dtype=[int],
                                   param_names='x')

    # Add points
    xx = np.arange(20)
    yy = np.sin(xx)+0.5*np.random.randn(20) + 10
    for iP in range(len(xx)):
        gpo.add_point([int(xx[iP])], [yy[iP]])

    # Show the loss surface
    gpo.plot_surface('x')
    if plot:
        plt.show()

    # Draw a random point
    new_point = gpo.random_point()
    assert isinstance(new_point, list)
    assert len(new_point) == 1
    assert isinstance(new_point[0], int)



# TODO: test_GaussianProcessOptimizer_2d



def test_optimize_params_random_1d(plot):
    #Tests evaluation.optimize_params w/ regression problems

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
    df['y'] = (y-y.mean())/y.std()
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
    opt_params, gpo = \
        optimize_params(X_train, y_train, model, bounds,
                        n_splits=3,
                        max_time=None,
                        max_evals=10,
                        n_random=10,
                        metric='mse')

    # Ensure opt_params is correct
    assert isinstance(opt_params, dict)
    assert 'regressor__alpha' in opt_params

    # Plot parameters and their scores
    if plot:
        gpo.plot_surface()
        plt.xlabel('regressor__alpha')
        plt.ylabel('mse')
        plt.show()

    # Check n_jobs arg works
    opt_params, gpo = \
        optimize_params(X_train, y_train, model, bounds,
                        n_splits=3,
                        max_time=None,
                        max_evals=10,
                        n_random=10,
                        metric='mse',
                        n_jobs=2)

    # Check timeout works
    start_time = time.time()
    opt_params, gpo = \
        optimize_params(X_train, y_train, model, bounds,
                        n_splits=3,
                        max_time=0.2,
                        max_evals=int(1e12),
                        n_random=int(1e12),
                        metric='mse')
    assert time.time()-start_time < 3.0



def test_optimize_params_random_2d(plot):
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
    df['y'] = (y-y.mean())/y.std()
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
    opt_params, gpo = optimize_params(X_train, y_train, model, bounds,
                                      n_splits=3,
                                      max_time=None,
                                      max_evals=10,
                                      n_random=10,
                                      metric='mse')

    # Plot parameters and their scores
    if plot:
        gpo.plot_surface(x_dim='regressor__alpha',
                         y_dim='regressor__l1_ratio')
        plt.xlabel('regressor__alpha')
        plt.ylabel('regressor__l1_ratio')
        plt.title('2D random optimization')
        plt.show()



def test_optimize_params_random_int(plot):
    #Tests evaluation.optimize_params w/ int parameter types

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
    df['y'] = (y-y.mean())/y.std()
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
    }

    # Find best parameters
    opt_params, gpo = optimize_params(X_train, y_train, model, bounds,
                                      n_splits=3,
                                      max_time=None,
                                      max_evals=10,
                                      n_random=10,
                                      metric='mse')

    # Plot parameters and their scores
    if plot:
        gpo.plot_surface()
        plt.xlabel('pca__n_components')
        plt.ylabel('mse')
        plt.title('1D Int random optimization')
        plt.show()



def test_optimize_params_1d(plot):
    #Tests evaluation.optimize_params w/ nonrandom optimization

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
    df['y'] = (y-y.mean())/y.std()
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
    opt_params, gpo = \
        optimize_params(X_train, y_train, model, bounds,
                        n_splits=3,
                        max_time=None,
                        max_evals=10,
                        n_random=5,
                        metric='mse')

    # Plot parameters and their scores
    if plot:
        gpo.plot_surface()
        plt.xlabel('regressor__alpha')
        plt.ylabel('mse')
        plt.title('GP optimization')
        plt.show()



def test_optimize_params_n_grid(plot):
    #Tests evaluation.optimize_params w/ n_grid argument

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
    df['y'] = (y-y.mean())/y.std()
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
    opt_params, gpo = \
        optimize_params(X_train, y_train, model, bounds,
                        n_splits=3,
                        max_time=None,
                        max_evals=5,
                        n_grid=5,
                        n_random=0,
                        metric='mse')

    # Plot parameters and their scores
    if plot:
        gpo.plot_surface()
        plt.xlabel('regressor__alpha')
        plt.ylabel('mse')
        plt.title('n_grid')
        plt.show()
