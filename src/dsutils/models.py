"""Models

* :class:`.InterpolatingPredictor`
* :class:`.SvdRegressor`

"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

import surprise


class InterpolatingPredictor(BaseEstimator):
    
    def __init__(self, loss_func=mean_squared_error):
        """Linear model which predicts the best interpolation between features

        References
        ----------
            Breiman, L. (1996). Stacked regressions. Machine learning, 24(1),
            49-64.
        """
        
        self._weights = None
        self.loss_func = loss_func


    def fit(self, X, y):
        """
        TODO
        """

        # Function to optimize
        def func(x):
            return self.loss_func(y, (x.reshape(1, -1)*X).sum(axis=1))

        # Number of dimensions of input
        Nd = X.shape[1]

        # Initial parameter guess
        x0 = np.ones(Nd)/float(Nd)
        
        # Constrain weights to sum to 1
        constraints = ({'type': 'eq',
                        'fun': lambda x: 1.0 - sum(x)})

        # Constain weights to be positive
        bounds = tuple((0.0, 1.0) for _ in range(Nd))

        # Solve for weights which optimize error
        res = minimize(func, x0, method='SLSQP',
                       bounds=bounds, constraints=constraints)
        self._weights = res.x
            
        # Return the fit object
        return self


    def predict(self, X):
        """
        TODO
        """

        # Check if model has been fit
        if self._weights is None:
            raise RuntimeError('model has not been fit')
        
        # Return the predictions
        return (self._weights.reshape(1, -1)*X).sum(axis=1)


    def fit_predict(self, X, y):
        """
        TOOD
        """
        return self.fit(X, y).predict(X)



class SvdRegressor(BaseEstimator, RegressorMixin):
    """Uses suprise's SVD to predict user scores of items
    
    TODO: docs
    
    Parameters
    ----------
    user_col : int or str
        Column of X to use as user IDs.
        Default is 0.
    item_col : int or str
        Column of X to use as item IDs.
        Default is 1.
    score_range : tuple of int or float
        Range of the scores in y.  First element should be lower 
        bound and second element should be upper bound.
        Default is to use the minimum and maximum y values.
    n_factors : int
        Number of factors to use (dimensionality of the user and item embeddings).
        Default = 100
    n_epochs : int
        Number of epochs to train for.
        Default = 20
    **kwargs
        Additional kwargs are passed to 
        surprise.prediction_algorithms.matrix_factorization.SVD.
    
    """
    
    def __init__(self, user_col=0, item_col=1, score_range=(None, None), **kwargs):
        """Create the regressor"""
        
        # Check inputs
        if not isinstance(user_col, (int, str)):
            raise TypeError('user_col must be a str or an int')
        if not isinstance(item_col, (int, str)):
            raise TypeError('item_col must be a str or an int')
        if not isinstance(score_range, tuple):
            raise TypeError('score_range must be a tuple')
        if len(score_range) != 2:
            raise ValueError('score_range must be a tuple with 2 elements')
        if not all(isinstance(e, (int, float)) for e in score_range):
            raise TypeError('score_range must contain ints or floats')
        if score_range[0] > score_range[1]:
            raise ValueError('First element of score_range must be less than second element')
            
        # Store parameters
        self._user_col = user_col
        self._item_col = item_col
        self._score_range = score_range
        self._kwargs = kwargs
        self._model = None


    def fit(self, X, y):
        """Fit the SVD model to data
        
        Parameters
        ----------
        X : pandas DataFrame
            Table containing user IDs and item IDs.
        y : pandas Series
            Scores.

        Returns
        -------
        self
            The fit estimator.
        """

        # Create a pandas DataFrame in suprise format
        X_train = X[[self._user_col, self._item_col]]
        X_train['score'] = y
        
        # Compute the score range
        if self._score_range[0] is None:
            self._score_range[0] = y.min()
        if self._score_range[1] is None:
            self._score_range[1] = y.max()
        
        # Create a suprise Dataset from the dataframe
        reader = surprise.Reader(rating_scale=self._score_range)
        dataset = (surprise.dataset.Dataset
                   .load_from_df(X_train, reader))
        dataset = dataset.build_full_trainset()

        # Fit the model
        self._model = surprise.SVD(**self._kwargs)
        self._model.fit(dataset)
        return self
    

    def predict(self, X, y=None):
        """Predict scores given user and item IDs

        Parameters
        ----------
        X : pandas DataFrame
            Table containing user IDs and item IDs.

        Returns
        -------
        y_pred : pandas Series
            Predicted scores.

        """

        # Check if model has been fit
        if self._model is None:
            raise RuntimeError('model has not been fit')
        
        # Create a pandas DataFrame in suprise format
        X_test = X[[self._user_col, self._item_col]]
        X_test['score'] = np.nan
        
        # Create a suprise Testset from the dataframe
        reader = surprise.Reader(rating_scale=self._score_range)
        testset = (surprise.dataset.Dataset
                   .load_from_df(X_test, reader))
        testset = testset.build_full_trainset().build_testset()
        
        # Use suprise model to predict scores
        preds = self._model.test(testset) #returns a list of "Prediction" objs...
        preds = [pred[3] for pred in preds]
        return pd.Series(data=np.array(preds),
                         index=X.index)
        
        
    def fit_predict(self, X, y):
        """Fit and predict scores.

        Parameters
        ----------
        X : pandas DataFrame
            Table containing user IDs and item IDs.
        y : pandas Series
            Scores.

        Returns
        -------
        y_pred : pandas Series
            Predicted scores.

        """
        return self.fit(X, y).predict(X)
