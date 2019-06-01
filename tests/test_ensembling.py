"""Tests ensembling

"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA

from dsutils.ensembling import EnsembleRegressor
from dsutils.ensembling import StackedRegressor
from dsutils.ensembling import BaggedRegressor
from dsutils.encoding import TargetEncoderCV

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



def test_BaggedRegressor():
    """Tests ensembling.BaggedRegressor"""

    # Make dummy data
    N = 100
    D = 3
    X = pd.DataFrame(data=np.random.randn(N,D))
    y = pd.Series(index=X.index)
    w = np.random.randn(D)
    y = np.sum(X*w.T, axis=1)

    # Create an ensemble regressor
    model = LinearRegression()
    bagged_model = BaggedRegressor(model)

    # Predict y values
    y_pred = bagged_model.fit_predict(X, y)
    assert isinstance(y_pred, pd.Series)

    # Test with numpy array as input
    X = np.random.randn(N,D)
    w = np.random.randn(D)
    y = np.sum(X*w.T, axis=1)
    model = LinearRegression()
    bagged_model = BaggedRegressor(model)
    y_pred = bagged_model.fit_predict(X, y)
    assert isinstance(y_pred, np.ndarray)



def test_BaggedRegressor_options():
    """Tests ensembling.BaggedRegressor"""

    # Make dummy data
    N = 100
    D = 3
    X = pd.DataFrame(data=np.random.randn(N,D))
    y = pd.Series(index=X.index)
    w = np.random.randn(D)
    y = np.sum(X*w.T, axis=1)

    # Base model
    model = LinearRegression()

    # Create an ensemble regressor w/ n_estimators option
    bagged_model = BaggedRegressor(model, n_estimators=5)
    y_pred = bagged_model.fit_predict(X, y)

    # Create an ensemble regressor w/ max_samples option
    bagged_model = BaggedRegressor(model, max_samples=0.5)
    y_pred = bagged_model.fit_predict(X, y)

    # Create an ensemble regressor w/ max_features option
    bagged_model = BaggedRegressor(model, max_features=0.5)
    y_pred = bagged_model.fit_predict(X, y)

    # Create an ensemble regressor w/ replace option
    bagged_model = BaggedRegressor(model, replace=False)
    y_pred = bagged_model.fit_predict(X, y)

    # Create an ensemble regressor w/ replace_features option
    bagged_model = BaggedRegressor(model, replace_features=True)
    y_pred = bagged_model.fit_predict(X, y)



def test_BaggedRegressor_TargetEncoding():
    """Tests ensembling.BaggedRegressor works w/ TargetEncoding"""

    # Make dummy data
    N = 100
    D = 3
    X = pd.DataFrame(data=np.random.randn(N,D))
    y = pd.Series(index=X.index)
    w = np.random.randn(D)
    y = np.sum(X*w.T, axis=1)

    # Add categorical col
    cats = ['a', 'b']
    X['cat_col'] = [cats[i%len(cats)] for i in range(N)]

    # Create a bagged regressor
    model = Pipeline([
        ('target_encoder', TargetEncoderCV(cols=['cat_col'])),
        ('regressor', LinearRegression())
    ])
    bagged_model = BaggedRegressor(model)

    # Predict y values
    y_pred = bagged_model.fit_predict(X, y)
