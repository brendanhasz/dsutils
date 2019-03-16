"""Ensembling

* :func:`.permutation_importance`
* :func:`.permutation_importance_cv`
* :func:`.plot_permutation_importance`

"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold



class EnsembleRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, base_learners):
        """
        TODO
        """
        
        # Check inputs
        # TODO
        
        # Save dict of base learners
        self.base_learners = dict()
        if isinstance(base_learners[0], tuple):
            for learner in base_learners:
                self.base_learners[learner[0]] = learner[1]
        else:
            for i, learner in enumerate(base_learners):
                self.base_learners[str(i)] = learner

        
    def fit(self, X, y):
        """
        TODO
        """
        
        # Fit each base learner to the data
        for _, learner in self.base_learners.items():
            learner = learner.fit(X, y)
            
        # Return the fit object
        return self
                
        
    def predict(self, X):
        """
        TODO
        """

        # Compute predictions for each base learner
        preds = pd.DataFrame(index=X.index)
        for name, learner in self.base_learners.items():
            preds[name] = learner.predict(X)
        
        # Return the average predictions
        return preds.mean(axis=1)


    def fit_predict(self, X, y):
        """
        TOOD
        """
        return self.fit(X, y).predict(X)



class StackedRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, base_learners, meta_learner=LogisticRegression,
                 n_splits=3, shuffle=True):
        """
        TODO
        """
        
        # Check inputs
        # TODO
        # basee_learners shoudl be list

        # Save dict of base learners
        self.base_learners = dict()
        if isinstance(base_learners[0], tuple):
            for learner in base_learners:
                self.base_learners[learner[0]] = learner[1]
        else:
            for i, learner in enumerate(base_learners):
                self.base_learners[str(i)] = learner

        # Set parameters
        self.meta_learner = meta_learner
        self.n_splits = n_splits
        self.shuffle = shuffle
        
        
    def fit(self, X, y):
        """
        TODO
        """
        
        # Use base learners to cross-val predict
        preds = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
        for name, learner in self.base_learners.items():
            preds[name] = cross_val_predict(learner, X, y, cv=kf)
            
        # Fit base learners to all samples
        for _, learner in self.base_learners.items():
            learner = learner.fit(X, y)
            
        # Fit meta learner on base learners' predictions
        # TODO: oh, have to have it predict the *weights*, not the values
        self.meta_learner = self.meta_learner.fit(preds, y)
        
        # Return fit object
        return self
    
                
    def predict(self, X, y=None):
        """
        TODO
        """
        
        # Use base learners to predict
        preds = pd.DataFrame(index=X.index)
        for name, learner in self.base_learners.items():
            preds[name] = learner.predict(X)
            
        # Use meta learner to predict based on base learners' predictions
        # TODO: oh, have to have it predict the *weights*, not the values
        y_pred = self.meta_learner.predict(preds)
        
        # Return meta-learner's predictions
        return y_pred


    def fit_predict(self, X, y):
        """
        TOOD
        """
        return self.fit(X, y).predict(X)
