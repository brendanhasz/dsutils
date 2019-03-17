"""Ensembling

* :func:`.EnsembleRegressor`
* :func:`.StackedRegressor`

"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

from dsutils.models import InterpolatingPredictor


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, base_learners, preprocessing=None):
        """Estimator which takes the mean prediction of a set of base learners

        Parameters
        ----------
        base_learners : list of sklearn Estimators
            List of base estimators to use.
        preprocessing : sklearn Estimator
            Preprocessing pipline to apply to the data before using models
            to predict.  This saves time for heavy preprocessing workloads
            because the preprocessing does not have to be repeated for each
            estimator.
        """
        
        # Check inputs
        if not isinstance(base_learners, list):
            raise TypeError('base_learners must be a list of estimators')
        if (preprocessing is not None and
                not isinstance(preprocessing, BaseEstimator)):
            raise TypeError('preprocessing must be an sklearn estimator')

        # Store learners as dict
        self.base_learners = dict()
        for i, learner in enumerate(base_learners):
            if (isinstance(learner, tuple) and
                    len(learner)==2 and 
                    isinstance(learner[0], str) and 
                    isinstance(learner[1], BaseEstimator)):
                self.base_learners[learner[0]] = learner[1]
            elif isinstance(learner, BaseEstimator):
                self.base_learners[str(i)] = learner
            else:
                raise TypeError('each element of base_learners must be an '
                                'sklearn estimator or a (str, sklearn '
                                'estimator) tuple')

        # Store parameters
        self.preprocessing = preprocessing


    def fit(self, X, y):
        """Fit the ensemble of base learners

        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas Series
            Target variable

        Returns
        -------
        self
            The fit estimator
        """

        # Preprocess the data
        if self.preprocessing is None:
            Xp = X
        else:
            self.preprocessing = self.preprocessing.fit(X, y)
            Xp = self.preprocessing.transform(X)
        
        # Fit each base learner to the data
        for _, learner in self.base_learners.items():
            learner = learner.fit(Xp, y)
            
        # Return the fit object
        return self
                
        
    def predict(self, X):
        """Predict using the average of the base learners

        Parameters
        ----------
        X : pandas DataFrame
            Features

        Returns
        -------
        y_pred : pandas Series
            Predicted target variable
        """

        # Preprocess the data
        if self.preprocessing is None:
            Xp = X
        else:
            Xp = self.preprocessing.transform(X)

        # Compute predictions for each base learner
        preds = pd.DataFrame(index=X.index)
        for name, learner in self.base_learners.items():
            preds[name] = learner.predict(Xp)
        
        # Return the average predictions
        return preds.mean(axis=1)


    def fit_predict(self, X, y):
        """
        TOOD
        """
        return self.fit(X, y).predict(X)



class StackedRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, base_learners, meta_learner=InterpolatingPredictor(),
                 n_splits=3, shuffle=True, preprocessing=None):
        """Uses a meta-estimator to predict from base estimators predictions

        Parameters
        ----------
        base_learners : list of sklearn Estimators
            List of base estimators to use.       
        preprocessing : sklearn Estimator
            Preprocessing pipline to apply to the data before using models
            to predict.  This saves time for heavy preprocessing workloads
            because the preprocessing does not have to be repeated for each
            estimator.
        """
        
        # Check inputs
        if not isinstance(base_learners, list):
            raise TypeError('base_learners must be a list of estimators')
        if not isinstance(meta_learner, BaseEstimator):
            raise TypeError('meta_learner must be an sklearn estimator')
        if not isinstance(n_splits, int):
            raise TypeError('n_splits must be an int')
        if n_splits < 1:
            raise ValueError('n_splits must be positive')
        if not isinstance(shuffle, bool):
            raise TypeError('shuffle must be True or False')
        if (preprocessing is not None and
                not isinstance(preprocessing, BaseEstimator)):
            raise TypeError('preprocessing must be an sklearn estimator')

        # Store learners as dict
        self.base_learners = dict()
        for i, learner in enumerate(base_learners):
            if (isinstance(learner, tuple) and
                    len(learner)==2 and 
                    isinstance(learner[0], str) and 
                    isinstance(learner[1], BaseEstimator)):
                self.base_learners[learner[0]] = learner[1]
            elif isinstance(learner, BaseEstimator):
                self.base_learners[str(i)] = learner
            else:
                raise TypeError('each element of base_learners must be an '
                                'sklearn estimator or a (str, sklearn '
                                'estimator) tuple')

        # Store parameters
        self.meta_learner = meta_learner
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.preprocessing = preprocessing
        
        
    def fit(self, X, y):
        """Fit the ensemble of base learners and the meta-estimator

        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas Series
            Target variable

        Returns
        -------
        self
            The fit estimator
        """

        # Preprocess the data
        if self.preprocessing is None:
            Xp = X
        else:
            self.preprocessing = self.preprocessing.fit(X, y)
            Xp = self.preprocessing.transform(X)
        
        # Use base learners to cross-val predict
        preds = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
        for name, learner in self.base_learners.items():
            preds[name] = cross_val_predict(learner, Xp, y, cv=kf)
            
        # Fit base learners to all samples
        for _, learner in self.base_learners.items():
            learner = learner.fit(Xp, y)
            
        # Fit meta learner on base learners' predictions
        self.meta_learner = self.meta_learner.fit(preds, y)

        # Return fit object
        return self
    
                
    def predict(self, X, y=None):
        """Predict using the meta-estimator

        Parameters
        ----------
        X : pandas DataFrame
            Features

        Returns
        -------
        y_pred : pandas Series
            Predicted target variable
        """

        # Preprocess the data
        if self.preprocessing is None:
            Xp = X
        else:
            Xp = self.preprocessing.transform(X)
        
        # Use base learners to predict
        preds = pd.DataFrame(index=X.index)
        for name, learner in self.base_learners.items():
            preds[name] = learner.predict(Xp)
            
        # Use meta learner to predict based on base learners' predictions
        y_pred = self.meta_learner.predict(preds)
        
        # Return meta-learner's predictions
        return y_pred


    def fit_predict(self, X, y):
        """
        TOOD
        """
        return self.fit(X, y).predict(X)
