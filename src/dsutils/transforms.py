"""Transforms

* :func:`.quantile_transform`

"""


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin



def quantile_transform(v, res=101):
    """Quantile-transform a vector to lie between 0 and 1"""
    x = np.linspace(0, 100, res)
    prcs = np.nanpercentile(v, x)
    return np.interp(v, prcs, x/100.0)



class Scaler(BaseEstimator, TransformerMixin):
    """Z-scores each column (and outputs a DataFrame)
    
    Parameters
    ----------
    cols : list of str
        Columns to scale.  Default is to scale all columns

    """
    
    def __init__(self, cols=None):

        # Check types
        if cols is not None and not isinstance(cols, (str, list)):
            raise TypeError('cols must be a str or list of str')

        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols

        
    def fit(self, X, y):
        """Fit the scaler to X and y.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to scale
        y : pandas Series of shape (n_samples,)
            Dependent variable values.
            
        Returns
        -------
        StandardScaler
            Returns self, the fit object.
        """
        
        # Scale all columns by default
        if self.cols is None:
            self.cols = X.columns.tolist()

        # Compute the mean and std of each column
        self.means = dict()
        self.stds = dict()
        for col in self.cols:
            self.means[col] = X[col].mean()
            self.stds[col] = X[col].std()
                        
        # Return fit object
        return self

        
    def transform(self, X, y=None):
        """Perform the scaling.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to scale
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with scaled columns
        """

        # Scale each column
        Xo = X.copy()
        for col in self.cols:
            Xo[col] = (X[col]-self.means[col])/self.stds[col]

        # Return dataframe with scaled values
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to scale
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with scaled columns
        """
        return self.fit(X, y).transform(X, y)



class Imputer(BaseEstimator, TransformerMixin):
    """Imputes missing vlaues (and outputs a DataFrame)
    
    Parameters
    ----------
    cols : list of str
        Columns to impute.  Default is to impute all columns
    method : str 
        Method to use for imputation.  'mean' or 'median'.
        Default = 'median'
    """
    
    def __init__(self, cols=None, method='median'):

        # Check types
        if cols is not None and not isinstance(cols, (str, list)):
            raise TypeError('cols must be a str or list of str')
        if not isinstance(method, str):
            raise TypeError('method must be a str')
        if method not in ['mean', 'median']:
            raise ValueError('method must be \'median\' or \'mean\'')

        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.method = method

        
    def fit(self, X, y):
        """Fit the imputer to X and y.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to impute
        y : pandas Series of shape (n_samples,)
            Dependent variable values.
            
        Returns
        -------
        StandardScaler
            Returns self, the fit object.
        """
        
        # Scale all columns by default
        if self.cols is None:
            self.cols = X.columns.tolist()

        # Compute the value to use for imputation
        self.val = dict()
        for col in self.cols:
            if self.method == 'mean':
                self.val[col] = X[col].mean()
            else:
                self.val[col] = X[col].median()

        # Return fit object
        return self

        
    def transform(self, X, y=None):
        """Perform the imputation.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to impute
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with imputed values
        """

        # Scale each column
        Xo = X.copy()
        for col in self.cols:
            Xo.loc[Xo[col].isnull(), col] = self.val[col]

        # Return dataframe with imputed values
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to impute
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with imputed values
        """
        return self.fit(X, y).transform(X, y)
