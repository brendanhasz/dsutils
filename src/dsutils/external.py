"""Classes which depend on external packages

* :class:`.SurpriseRegressor`

"""



import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

import surprise



class SurpriseRegressor(BaseEstimator, RegressorMixin):
    """Uses a suprise model to predict user scores of items
    
    A wrapper sklearn interface for suprise models 
    (https://github.com/NicolasHug/Surprise).
    
    Parameters
    ----------
    model : surprise model
        The surprise model to use (e.g. BaselineOnly, KNNBaseline, or SVD)
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
    **kwargs
        Additional kwargs are passed to the surprise model constructor.
        
    Examples
    --------
    
    To create a sklearn regressor which uses the surprise's
    implementation of SVD:
    
        model = SurpriseRegressor(model=surprise.SVD,
                                  user_col='user_id',
                                  item_col='item_id')
                                  
    Pass additional kwargs to pass to the suprise model.
    For example, to set the number of factors and the
    number of epochs for training the SVD model,
    
        model = SurpriseRegressor(model=surprise.SVD,
                                  user_col='user_id',
                                  item_col='item_id',
                                  n_factors=50,
                                  n_epochs=1000)
                                  
    Or to set the number of neighbors for the KNN model:
    
        model = SurpriseRegressor(model=surprise.KNNBaseline,
                                  user_col='user_id',
                                  item_col='item_id',
                                  k=50)
                                  
    """
    
    def __init__(self, model=surprise.SVD, user_col=0, item_col=1, score_range=(None, None), **kwargs):
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
        self._surprise_model = model
        self._model = None


    def fit(self, X, y):
        """Fit the surprise model to data.
        
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
        self._model = self._surprise_model(**self._kwargs)
        self._model.fit(dataset)
        return self
    

    def predict(self, X, y=None):
        """Predict the scores using the surprise model.
        
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
        """Fit the surprise model and predict the scores.
        
        Parameters
        ----------
        X : pandas DataFrame
            Table containing user IDs and item IDs.
        y : pandas Series
            True scores.
            
        Returns
        -------
        y_pred : pandas Series
            Predicted scores.
        """
        return self.fit(X, y).predict(X)
