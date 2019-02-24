"""Encoding

* :func:`.q_mut_info`
* :class:`.TargetEncoder`
* :class:`.TargetEncoderCV`

"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


def null_encode(df, cols=None):
    """Add indicator columns for nulls.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe from which to one-hot encode columns
    cols : list of str
        Columns in df for which to add null indicator columns

    Returns
    -------
        Nothing, modifies df in-place.
    """

    # Do for all columns if not specified
    if cols is None:
        cols = df.columns

    # Make list if not
    if isinstance(cols, str):
        cols = [cols]

    # Check columns are in df
    for col in cols:
        if col not in df:
            raise ValueError('Column \''+col+'\' not in DataFrame')

    # Add null indicator column for each column
    for col in cols:
        if df[col].isnull().sum() > 0:
            df[col+'_isnull'] = df[col].isnull().astype('uint8')



def one_hot_encode(df, cols=None, reduce_df=False):
    """One-hot encode columns.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe from which to one-hot encode columns
    cols : list of str
        Columns in df to one-hot encode
    reduce_df : bool
        Whether to add N-1 one-hot columns for a column with N categories. 
        E.g. for a column with categories A, B, and C:
        When reduce_df is True, A=[1, 0], B=[0, 1], and C=[0, 0]
        When reduce_df is False, A=[1, 0, 0], B=[0, 1, 0], and C=[0, 0, 1]

    Returns
    -------
        Nothing, modifies df in-place.
    """

    # Do for all "object" columns if not specified
    if cols is None:
        cols = [col for col in df if str(df[col].dtype)=='object']

    # Make list if not
    if isinstance(cols, str):
        cols = [cols]

    # Check columns are in df
    for col in cols:
        if col not in df:
            raise ValueError('Column \''+col+'\' not in DataFrame')

    # One-hot encode each column
    for col in cols:
        uniques = df[col].unique()
        if len(uniques) < 2:
            print('Warning: column '+col+' has <2 unique values, removing it')
        elif len(uniques) == 2: #only 2 unique categories, just add binary col
            new_col = col+'_'+str(uniques[0])
            df[new_col] = (df[col] == uniques[0]).astype('uint8')
        else:
            for u_val in uniques:
                new_col = col+'_'+str(u_val)
                df[new_col] = (df[col] == u_val).astype('uint8')
            if reduce_df:
                del df[new_col]

    # Delete original columns from dataframe
    for col in cols:
        del df[col]



class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder.
    
    Replaces categorical column(s) with the mean target value for
    each category.


    """
    
    def __init__(self, cols=None):
        """Target encoder
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.  Default is to target-encode all 
            categorical columns in the DataFrame.
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        
        
    def fit(self, X, y):
        """Fit target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X if str(X[col].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Encode each element of each column
        self.maps = dict() #dict to store map for each column
        for col in self.cols:
            tmap = dict()
            uniques = X[col].unique()
            for unique in uniques:
                tmap[unique] = y[X[col]==unique].mean()
            self.maps[col] = tmap
            
        return self

        
    def transform(self, X, y=None):
        """Perform the target encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, tmap in self.maps.items():
            vals = np.full(X.shape[0], np.nan)
            for val, mean_target in tmap.items():
                vals[X[col]==val] = mean_target
            Xo[col] = vals
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class TargetEncoderCV(TargetEncoder):
    """Cross-validated target encoder.

    Note
    ----
    You must call transform() on the same dataset to which you fit().
    """
    
    def __init__(self, n_splits=3, shuffle=True, cols=None):
        """Cross-validated target encoding for categorical features.
        
        Parameters
        ----------
        n_splits : int
            Number of cross-validation splits. Default = 3.
        shuffle : bool
            Whether to shuffle the data when splitting into folds.
        cols : list of str
            Columns to target encode.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.cols = cols
        

    def fit(self, X, y):
        """Fit target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        self._target_encoder = TargetEncoder(cols=self.cols)
        self._target_encoder.fit(X, y)
        return self

    
    def transform(self, X, y=None):
        """Perform the target encoding transformation.

        Uses cross-validated target encoding for the training fold, and uses
        normal target encoding for the test fold.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """

        # Use target encoding from fit() if this is test data
        if y is None:
            return self._target_encoder.transform(X)

        # Compute means for each fold
        self._train_ix = []
        self._test_ix = []
        self._fit_tes = []
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
        for train_ix, test_ix in kf.split(X):
            self._train_ix.append(train_ix)
            self._test_ix.append(test_ix)
            te = TargetEncoder(cols=self.cols)
            if isinstance(X, pd.DataFrame):
                self._fit_tes.append(te.fit(X.iloc[train_ix,:],
                                            y.iloc[train_ix]))
            elif isinstance(X, np.ndarray):
                self._fit_tes.append(te.fit(X[train_ix,:], y[train_ix]))
            else:
                raise TypeError('X must be DataFrame or ndarray')

        # Apply means across folds
        Xo = X.copy()
        for ix in range(len(self._test_ix)):
            test_ix = self._test_ix[ix]
            if isinstance(X, pd.DataFrame):
                Xo.iloc[test_ix,:] = self._fit_tes[ix].transform(X.iloc[test_ix,:])
            elif isinstance(X, np.ndarray):
                Xo[test_ix,:] = self._fit_tes[ix].transform(X[test_ix,:])
            else:
                raise TypeError('X must be DataFrame or ndarray')
        return Xo

            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)
