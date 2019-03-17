"""Tests ensembling

"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA

from dsutils.ensembling import EnsembleRegressor
from dsutils.ensembling import StackedRegressor


def test_EnsembleRegressor():
    """Tests ensembling.EnsembleRegressor"""

    # Make dummy data
    N = 100
    D = 3
    X = pd.DataFrame(data=np.random.randn(N,D))
    y = pd.Series(index=X.index)
    w = np.random.randn(D)
    y = np.sum(X*w.T, axis=1)

    # Create an ensemble regressor
    base_learners = [
        ('linear_regression', LinearRegression()),
        ('knn_regression', KNeighborsRegressor())
    ]
    ensemble_model = EnsembleRegressor(base_learners)

    # Predict y values
    y_pred = ensemble_model.fit_predict(X, y)


def test_StackedRegressor():
    """Tests ensembling.StackedRegressor"""

    # Make dummy data
    N = 100
    D = 3
    X = pd.DataFrame(data=np.random.randn(N,D))
    y = pd.Series(index=X.index)
    w = np.random.randn(D)
    y = np.sum(X*w.T, axis=1)

    # Create a stacked regressor
    base_learners = [
        ('linear_regression', LinearRegression()),
        ('knn_regression', KNeighborsRegressor())
    ]
    meta_learner = Ridge()
    stacked_model = StackedRegressor(base_learners, 
                                      meta_learner=meta_learner)

    # Predict y values
    y_pred = stacked_model.fit_predict(X, y)


def test_EnsembleRegressor_preprocessing():
    """Tests ensembling.EnsembleRegressor's preprocessing arg"""

    # Make dummy data
    N = 100
    D = 3
    X = pd.DataFrame(data=np.random.randn(N,D))
    y = pd.Series(index=X.index)
    w = np.random.randn(D)
    y = np.sum(X*w.T, axis=1)

    # Create a stacked regressor
    preprocessing = PCA(n_components=2)
    base_learners = [
        ('linear_regression', LinearRegression()),
        ('knn_regression', KNeighborsRegressor())
    ]
    stacked_model = EnsembleRegressor(base_learners, 
                                     preprocessing=preprocessing)

    # Predict y values
    y_pred = stacked_model.fit_predict(X, y)


def test_StackedRegressor_preprocessing():
    """Tests ensembling.StackedRegressor's preprocessing arg"""

    # Make dummy data
    N = 100
    D = 3
    X = pd.DataFrame(data=np.random.randn(N,D))
    y = pd.Series(index=X.index)
    w = np.random.randn(D)
    y = np.sum(X*w.T, axis=1)

    # Create a stacked regressor
    preprocessing = PCA(n_components=2)
    base_learners = [
        ('linear_regression', LinearRegression()),
        ('knn_regression', KNeighborsRegressor())
    ]
    meta_learner = Ridge()
    stacked_model = StackedRegressor(base_learners, 
                                     meta_learner=meta_learner,
                                     preprocessing=preprocessing)

    # Predict y values
    y_pred = stacked_model.fit_predict(X, y)
