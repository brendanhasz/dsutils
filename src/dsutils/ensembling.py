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



class EnsembleRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, base_learners):
        """
        TODO
        """
        
        # Check inputs
        # TODO
        
        # Set parameters
        self.base_learners = base_learners

        
    def fit(self, X, y):
        """
        TODO
        """
        
        # Fit each base learner to the data
        for base_learner in self.base_learners:
            base_learner = base_learner.fit(X, y)
            
        # Return the fit object
        return self
                
        
    def predict(self, X):
        """
        TODO
        """

        # Compute predictions for each base learner
        preds = pd.DataFrame(index=X.index)
        for i, base_learner in enumerate(self.base_learners):
            preds[str(i)] = base_learner.predict(X)
        
        # Return the average predictions
        return y_pred.mean(axis=1)



class StackedRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, base_learners, meta_learner=LogisticRegression,
                 n_splits=3, shuffle=True):
        """
        TODO
        """
        
        # Check inputs
        # TODO
        
        # Set parameters
        self.base_learners = base_learners
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
        for i, learner in enumerate(self.base_learners):
            preds[str(i)] = cross_val_predict(learner, X, y, cv=kf)
            
        # Fit base learners to all samples
        for learner in self.base_learners:
            learner = learner.fit(X, y)
            
        # Fit meta learner on base learners' predictions
        self.meta_learner = self.meta_learner.fit(preds, y)
        
        # Return fit object
        return self
    
                
    def predict(self, X, y=None):
        """
        TODO
        """
        
        # Use base learners to predict
        preds = pd.DataFrame(index=X.index)
        for i, learner in enumerate(self.base_learners):
            preds[str(i)] = learner.predict(X)
            
        # Use meta learner to predict based on base learners' predictions
        y_pred = self.meta_learner.predict(preds)
        
        # Return meta-learner's predictions
        return y_pred