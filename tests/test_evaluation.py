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
from dsutils.evaluation import top_k_permutation_importances
from dsutils.evaluation import metric_cv



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



def test_top_k_permutation_importances():
    """Tests evaluation.top_k_permutation_importances"""

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

    # Get top features
    top_k = top_k_permutation_importances(imp_df, k=2)

    # Ensure top are correct
    assert len(top_k) == 2
    assert 'a' in top_k
    assert 'b' in top_k



def test_metric_cv():
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
    imp_df = metric_cv(reg_pipe, X, y, metric=median_absolute_error)
    