"""Ensembling

* :func:`.EnsembleRegressor`
* :func:`.StackedRegressor`

"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.exceptions import NotFittedError

#from dsutils.models import InterpolatingPredictor


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
                not (hasattr(preprocessing, 'fit') and 
                     hasattr(preprocessing, 'transform'))):
            raise TypeError('preprocessing must be an sklearn transformer')

        # Store learners as dict
        self.base_learners = dict()
        for i, learner in enumerate(base_learners):
            if (isinstance(learner, tuple) and
                    len(learner)==2 and 
                    isinstance(learner[0], str) and 
                    isinstance(learner[1], BaseEstimator)):
                self.base_learners[learner[0]] = learner[1]
            elif hasattr(learner, 'fit') and hasattr(learner, 'predict'):
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
        """Fit the ensemble and then predict on features in X

        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas Series
            Target variable

        Returns
        -------
        y_pred : pandas Series
            Predicted target variable
        """
        return self.fit(X, y).predict(X)



class StackedRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, base_learners, meta_learner=BayesianRidge(),
                 n_splits=3, shuffle=True, preprocessing=None, n_jobs=-1):
        """Uses a meta-estimator to predict from base estimators predictions

        Parameters
        ----------
        base_learners : list of sklearn Estimators
            List of base estimators to use.     
        meta_learner : sklearn Estimator
            Meta estimator to use.
        n_splits : int
            Number of cross-validation splits
        shuffle : bool
            Whether to shuffle the data
        preprocessing : sklearn Estimator
            Preprocessing pipline to apply to the data before using models
            to predict.  This saves time for heavy preprocessing workloads
            because the preprocessing does not have to be repeated for each
            estimator.
        n_jobs : int
            Number of parallel jobs to run. Default is to use as many 
            threads as there are processors.
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
                not (hasattr(preprocessing, 'fit') and 
                     hasattr(preprocessing, 'transform'))):
            raise TypeError('preprocessing must be an sklearn transformer')
        if not isinstance(n_jobs, int):
            raise TypeError('n_jobs must be an int')
        if n_jobs is not None and (n_jobs < -1 or n_jobs == 0):
            raise ValueError('n_jobs must be None or >0 or -1')

        # Store learners as dict
        self.base_learners = dict()
        for i, learner in enumerate(base_learners):
            if (isinstance(learner, tuple) and
                    len(learner)==2 and 
                    isinstance(learner[0], str) and 
                    isinstance(learner[1], BaseEstimator)):
                self.base_learners[learner[0]] = learner[1]
            elif hasattr(learner, 'fit') and hasattr(learner, 'predict'):
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
        self.n_jobs = n_jobs
        
        
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
            preds[name] = cross_val_predict(learner, Xp, y, 
                                            cv=kf, n_jobs=self.n_jobs)
            
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
        """Fit the ensemble and then predict on features in X

        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas Series
            Target variable
            
        Returns
        -------
        y_pred : pandas Series
            Predicted target variable
        """
        return self.fit(X, y).predict(X)



class BaggedRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, base_learner, 
                 preprocessing=None, 
                 n_estimators=10, 
                 max_samples=1.0, 
                 max_features=1.0,
                 replace=True,
                 replace_features=False):
        """Estimator which applies the same learner to samples of the data

        Parameters
        ----------
        base_learner : sklearn Estimator
            Base estimator to use.
        preprocessing : sklearn Estimator
            Preprocessing pipline to apply to the data before using model
            to predict.  This saves time for heavy preprocessing workloads
            because the preprocessing does not have to be repeated for each
            estimator.
        n_estimators : int
            Number of instances of the base learner to train
        max_samples : float between 0 and 1
            Proportion of the samples to use for training each instance
        max_features : float between 0 and 1
            Proportion of the fetaures to use for training each instance
        replace : bool
            Whether to sample samples with replacement
        replace_features : bool
            Whether to sample the features with replacement
        """
        
        # Check inputs
        if not isinstance(base_learner, BaseEstimator):
            raise TypeError('base_learner must be an sklearn estimator')
        if preprocessing is not None and not isinstance(preprocessing, 
                                                        BaseEstimator):
            raise TypeError('preprocessing must be an sklearn estimator')
        if not isinstance(n_estimators, int):
            raise TypeError('n_estimators must be an int')
        if n_estimators < 1: 
            raise ValueError('n_estimators must be positive')
        if not isinstance(max_samples, float):
            raise TypeError('max_samples must be a float')
        if max_samples < 0 or max_samples > 1: 
            raise ValueError('max_samples must be between 0 and 1')
        if not isinstance(max_features, float):
            raise TypeError('max_features must be a float')
        if max_features < 0 or max_features > 1: 
            raise ValueError('max_features must be between 0 and 1')
        if not isinstance(replace, bool):
            raise TypeError('replace must be True or False')
        if not isinstance(replace_features, bool):
            raise TypeError('replace_features must be True or False')
            
        # Store parameters
        self.base_learner = base_learner
        self.preprocessing = preprocessing
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.replace = replace
        self.replace_features = replace_features
        self.fit_learners = None
        self.features_ix = None        


    def fit(self, X, y):
        """Fit the base learners on samples of the data

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
        Ns = X.shape[0] #number of samples
        Nf = X.shape[1] #number of features
        self.fit_learners = []
        self.features_ix = []        
        for i in range(self.n_estimators):
            s_ix = np.random.choice(Ns, size=int(Ns*self.max_samples),
                                    replace=self.replace)
            f_ix = np.random.choice(Nf, size=int(Nf*self.max_features),
                                    replace=self.replace_features)
            if isinstance(Xp, pd.DataFrame):
                Xs = Xp.iloc[s_ix, f_ix]
            else:
                Xs = Xp[s_ix, :][:, f_ix]
            if isinstance(y, pd.Series):
                ys = y.iloc[s_ix]
            else:
                ys = y[s_ix]
            self.fit_learners.append(clone(self.base_learner).fit(Xs, ys))
            self.features_ix.append(f_ix)
            
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

        # Ensure model has been fit
        if self.fit_learners is None:
            raise NotFittedError('Model has not been fit')

        # Preprocess the data
        if self.preprocessing is None:
            Xp = X
        else:
            Xp = self.preprocessing.transform(X)

        # Compute predictions for each base learner
        if isinstance(X, pd.DataFrame):
            preds = pd.DataFrame(index=X.index)
        else:
            preds = pd.DataFrame(index=np.arange(X.shape[0]))
        for i, learner in enumerate(self.fit_learners):
            if isinstance(Xp, pd.DataFrame):
                Xs = Xp.iloc[:, self.features_ix[i]]
            else:
                Xs = Xp[:, self.features_ix[i]]
            preds[str(i)] = learner.predict(Xs)
        
        # Return the average predictions
        if isinstance(X, pd.DataFrame):
            return preds.mean(axis=1)
        else:
            return preds.mean(axis=1).values


    def fit_predict(self, X, y):
        """Fit the base learners and then predict on features in X

        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas Series
            Target variable

        Returns
        -------
        y_pred : pandas Series
            Predicted target variable
        """
        return self.fit(X, y).predict(X)