"""Encoding

Provides sklearn-compatible transformer classes for categorical encoding:

* :class:`.NullEncoder`
* :class:`.LabelEncoder`
* :class:`.OneHotEncoder`
* :class:`.TargetEncoder`
* :class:`.TargetEncoderCV`
* :class:`.TargetEncoderLOO`
* :class:`.TextMultiLabelBinarizer`
* :class:`.NhotEncoder`
* :class:`.JsonEncoder`
* :class:`.JoinTransformer`

Also provides functions to simply return an encoded DataFrame:

* :func:`.null_encode`
* :func:`.label_encode`
* :func:`.one_hot_encode`
* :func:`.target_encode`
* :func:`.target_encode_cv`
* :func:`.target_encode_loo`

"""

import ast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.exceptions import NotFittedError


class NullEncoder(BaseEstimator, TransformerMixin):
    """Null encoder.
    
    For each column with null values, adds a column containing indicators
    as to whether each sample in original column is null.

    """
    
    def __init__(self, cols=None, suffix='_isnull', dtype='uint8', 
                 delete_old=False, nocol=None):
        """Null encoder.
        
        Parameters
        ----------
        cols : list of str
            Columns to null encode.  Default is to null encode all columns in
            the DataFrame which contain null values.
        suffix : str
            Suffix to append to original column names to create null indicator
            column names
        dtype : str
            Datatype to use for encoded columns.
            Default = 'uint8'
        delete_old : bool
            Whether to delete the old column which was encoded
            Default = False
        nocol : None or str
            Action to take if a col in ``cols`` is not in the dataframe to 
            transform.  Valid values:

            * None - ignore cols in ``cols`` which are not in dataframe
            * 'warn' - issue a warning when a column is not in dataframe
            * 'err' - raise an error when a column is not in dataframe
        """

        # Check types
        if not isinstance(suffix, str):
            raise TypeError('suffix must be a string')
        if cols is not None and not isinstance(cols, (list, str)):
            raise TypeError('cols must be None, or a list or a string')
        if isinstance(cols, list):
            if not all(isinstance(c, str) for c in cols):
                raise TypeError('each element of cols must be a string')
        if not isinstance(dtype, str):
            raise TypeError('dtype must be a string (e.g. \'uint8\'')
        if nocol is not None and nocol not in ('warn', 'err'):
            raise ValueError('nocol must be None, \'warn\', or \'err\'')
        if not isinstance(delete_old, bool):
            raise TypeError('delete_old must be True or False')

        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.suffix = suffix
        self.dtype = dtype
        self.delete_old = delete_old
        self.nocol = nocol

        
    def fit(self, X, y):
        """Fit null encoder to X and y.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.
            
        Returns
        -------
        NullEncoder
            Returns self, the fit object.
        """
        
        # Encode all columns with any null values by default
        if self.cols is None:
            self.cols = [c for c in X if X[c].isnull().sum() > 0]

        # Check columns are in X
        if self.nocol == 'err':
            for col in self.cols:
                if col not in X:
                    raise ValueError('Column \''+col+'\' not in X')
        elif self.nocol == 'warn':
            for col in self.cols:
                if col not in X:
                    print('Column \''+col+'\' not in X')
                        
        # Return fit object
        return self

        
    def transform(self, X, y=None):
        """Perform the null encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """

        # Add null indicator column for each original column
        Xo = X.copy()
        for col in self.cols:
            Xo[col+self.suffix] = X[col].isnull().astype(self.dtype)
            if self.delete_old:
                del Xo[col]

        # Return encoded dataframe
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data with null encoding.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class LabelEncoder(BaseEstimator, TransformerMixin):
    """Label encoder.
    
    Replaces categorical column(s) with integer labels for each unique
    category in original column.

    """
    
    def __init__(self, cols=None, nocol=None):
        """Label encoder.
        
        Parameters
        ----------
        cols : list of str
            Columns to label encode.  Default is to label encode all
            categorical columns in the DataFrame.
        nocol : None or str
            Action to take if a col in ``cols`` is not in the dataframe to 
            transform.  Valid values:
            * None - ignore cols in ``cols`` which are not in dataframe
            * 'warn' - issue a warning when a column is not in dataframe
            * 'err' - raise an error when a column is not in dataframe
        """

        # Check types
        if cols is not None and not isinstance(cols, (list, str)):
            raise TypeError('cols must be None, or a list or a string')
        if isinstance(cols, list):
            if not all(isinstance(c, str) for c in cols):
                raise TypeError('each element of cols must be a string')
        if nocol is not None and nocol not in ('warn', 'err'):
            raise ValueError('nocol must be None, \'warn\', or \'err\'')

        # Store parameters
        self.nocol = nocol
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        
        
    def fit(self, X, y):
        """Fit label encoder to X and y.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.
            
        Returns
        -------
        LabelEncoder
            Returns self, the fit object.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [c for c in X if str(X[c].dtype)=='object']

        # Check columns are in X
        if self.nocol == 'err':
            for col in self.cols:
                if col not in X:
                    raise ValueError('Column \''+col+'\' not in X')
        elif self.nocol == 'warn':
            for col in self.cols:
                if col not in X:
                    print('Column \''+col+'\' not in X')

        # Create the map from objects to integers for each column
        self.maps = dict()
        for col in self.cols:
            Xu = X[col].dropna().unique()
            self.maps[col] = dict(zip(Xu, np.arange(len(Xu))))
                        
        # Return fit object
        return self

        
    def transform(self, X, y=None):
        """Perform the label encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, tmap in self.maps.items():
          
            # Map the column
            Xo[col] = Xo[col].map(tmap)
            
            # Convert to appropriate datatype
            max_val = max(tmap.values())
            if X[col].isnull().any(): #nulls, so need to use float!
                if max_val < 8388608:
                    dtype = 'float32'
                else:
                    dtype = 'float64'
            else:
                if max_val < 256:
                    dtype = 'uint8'
                elif max_val < 65536:
                    dtype = 'uint16'
                elif max_val < 4294967296:
                    dtype = 'uint32'
                else:
                    dtype = 'uint64'
            Xo[col] = Xo[col].astype(dtype)
            
        # Return encoded dataframe
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data with label encoding.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class OneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encoder.
    
    Replaces categorical column(s) with binary columns for each unique value
    in original column.

    """
    
    def __init__(self, cols=None, reduce_df=False, dtype='uint8', nocol=None):
        """One-hot encoder.
        
        Parameters
        ----------
        cols : list of str
            Columns to one-hot encode.  Default is to one-hot encode all
            categorical columns in the DataFrame.
        reduce_df : bool
            Whether to use reduced degrees of freedom for encoding (that is,
            add N-1 one-hot columns for a column with N categories). E.g. for
            a column with categories A, B, and C: When reduce_df is True,
            A=[1, 0], B=[0, 1], and C=[0, 0].  When reduce_df is False, 
            A=[1, 0, 0], B=[0, 1, 0], and C=[0, 0, 1].
            Default = False
        dtype : str
            Datatype to use for encoded columns. Default = 'uint8'
        nocol : None or str
            Action to take if a col in ``cols`` is not in the dataframe to 
            transform.  Valid values:
            * None - ignore cols in ``cols`` which are not in dataframe
            * 'warn' - issue a warning when a column is not in dataframe
            * 'err' - raise an error when a column is not in dataframe
        """

        # Check types
        if cols is not None and not isinstance(cols, (list, str)):
            raise TypeError('cols must be None, or a list or a string')
        if isinstance(cols, list):
            if not all(isinstance(c, str) for c in cols):
                raise TypeError('each element of cols must be a string')
        if not isinstance(reduce_df, bool):
            raise TypeError('reduce_df must be True or False')
        if not isinstance(dtype, str):
            raise TypeError('dtype must be a string (e.g. \'uint8\'')
        if nocol is not None and nocol not in ('warn', 'err'):
            raise ValueError('nocol must be None, \'warn\', or \'err\'')

        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.reduce_df = reduce_df
        self.dtype = dtype
        self.nocol = nocol

        
    def fit(self, X, y):
        """Fit one-hot encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.
            
        Returns
        -------
        OneHotEncoder
            Returns self, the fit object.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [c for c in X 
                         if str(X[c].dtype)=='object']

        # Check columns are in X
        if self.nocol == 'err':
            for col in self.cols:
                if col not in X:
                    raise ValueError('Column \''+col+'\' not in X')
        elif self.nocol == 'warn':
            for col in self.cols:
                if col not in X:
                    print('Column \''+col+'\' not in X')

        # Store each unique value
        self.maps = dict()
        for col in self.cols:
            self.maps[col] = []
            uniques = X[col].unique()
            for unique in uniques:
                self.maps[col].append(unique)

        # Remove last degree of freedom
        if self.reduce_df:
            for col in self.cols:
                del self.maps[col][-1]
        
        # Return fit object
        return self

        
    def transform(self, X, y=None):
        """Perform the one-hot encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, vals in self.maps.items():
            for val in vals:
                new_col = col+'_'+str(val)
                Xo[new_col] = (Xo[col]==val).astype(self.dtype)
            del Xo[col]
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data with one-hot encoding.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder.
    
    Replaces category values in categorical column(s) with the mean target
    (dependent variable) value for each category.

    """
    
    def __init__(self, cols=None, dtype='float64', nocol=None):
        """Target encoder.
        
        Parameters
        ----------
        cols : str or list of str
            Column(s) to target encode.  Default is to target encode all
            categorical columns in the DataFrame.
        dtype : str
            Datatype to use for encoded columns. Default = 'float64'
        nocol : None or str
            Action to take if a col in ``cols`` is not in the dataframe to 
            transform.  Valid values:
            * None - ignore cols in ``cols`` which are not in dataframe
            * 'warn' - issue a warning when a column is not in dataframe
            * 'err' - raise an error when a column is not in dataframe
        """

        # Check types
        if cols is not None and not isinstance(cols, (list, str)):
            raise TypeError('cols must be None, or a list or a string')
        if isinstance(cols, list):
            if not all(isinstance(c, str) for c in cols):
                raise TypeError('each element of cols must be a string')
        if not isinstance(dtype, str):
            raise TypeError('dtype must be a string (e.g. \'uint8\'')
        if nocol is not None and nocol not in ('warn', 'err'):
            raise ValueError('nocol must be None, \'warn\', or \'err\'')

        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.dtype = dtype
        self.nocol = nocol
        
        
    def fit(self, X, y):
        """Fit target encoder to X and y.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.
            
        Returns
        -------
        TargetEncoder
            Returns self, the fit object.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X if str(X[col].dtype)=='object']

        # Check columns are in X
        if self.nocol == 'err':
            for col in self.cols:
                if col not in X:
                    raise ValueError('Column \''+col+'\' not in X')
        elif self.nocol == 'warn':
            for col in self.cols:
                if col not in X:
                    print('Column \''+col+'\' not in X')

        # Encode each element of each column
        self.maps = dict()
        for col in self.cols:
            if col in X:
                tmap = dict()
                uniques = X[col].unique()
                for unique in uniques:
                    tmap[unique] = y[X[col]==unique].mean()
                self.maps[col] = tmap
            
        # Return fit object
        return self

        
    def transform(self, X, y=None):
        """Perform the target encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, tmap in self.maps.items():
            vals = np.full(X.shape[0], np.nan, dtype=self.dtype)
            for val, mean_target in tmap.items():
                vals[X[col]==val] = mean_target
            Xo[col] = vals
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data with target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class TargetEncoderCV(BaseEstimator, TransformerMixin):
    """Cross-fold target encoder.

    Replaces category values in categorical column(s) with the mean target
    (dependent variable) value for each category, using a cross-fold strategy
    such that no sample's target value is used in computing the target mean
    which is used to replace that sample's category value.

    """
    
    def __init__(self, cols=None, n_splits=3, shuffle=True, dtype='float64',
                 nocol=None):
        """Cross-fold target encoder.
        
        Parameters
        ----------
        cols : str or list of str
            Column(s) to target encode.  Default is to target encode all
            categorical columns in the DataFrame.
        n_splits : int
            Number of cross-fold splits. Default = 3.
        shuffle : bool
            Whether to shuffle the data when splitting into folds.
        dtype : str
            Datatype to use for encoded columns. Default = 'float64'
        nocol : None or str
            Action to take if a col in ``cols`` is not in the dataframe to 
            transform.  Valid values:
            * None - ignore cols in ``cols`` which are not in dataframe
            * 'warn' - issue a warning when a column is not in dataframe
            * 'err' - raise an error when a column is not in dataframe
        """

        # Check types
        if cols is not None and not isinstance(cols, (list, str)):
            raise TypeError('cols must be None, or a list or a string')
        if isinstance(cols, list):
            if not all(isinstance(c, str) for c in cols):
                raise TypeError('each element of cols must be a string')
        if not isinstance(n_splits, int):
            raise TypeError('n_splits must be an integer')
        if n_splits < 1:
            raise ValueError('n_splits must be positive')
        if not isinstance(shuffle, bool):
            raise TypeError('shuffle must be True or False')
        if not isinstance(dtype, str):
            raise TypeError('dtype must be a string (e.g. \'float64\'')
        if nocol is not None and nocol not in ('warn', 'err'):
            raise ValueError('nocol must be None, \'warn\', or \'err\'')

        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.dtype = dtype
        self.nocol = nocol


    def fit(self, X, y):
        """Fit cross-fold target encoder to X and y.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.
            
        Returns
        -------
        TargetEncoderCV
            Returns self, the fit object.
        """
        self._target_encoder = TargetEncoder(cols=self.cols, nocol=self.nocol)
        self._target_encoder.fit(X, y)
        return self

    
    def transform(self, X, y=None):
        """Perform the target encoding transformation.

        Uses cross-fold target encoding when given training data, and uses
        normal target encoding when given test data.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode

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
            self._fit_tes.append(te.fit(X.iloc[train_ix,:], y.iloc[train_ix]))

        # Apply means across folds
        Xo = X.copy()
        for ix in range(len(self._test_ix)):
            test_ix = self._test_ix[ix]
            Xo.iloc[test_ix,:] = self._fit_tes[ix].transform(X.iloc[test_ix,:])

        # Return transformed DataFrame
        return Xo

            
    def fit_transform(self, X, y=None):
        """Fit and transform the data with cross-fold target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class TargetEncoderLOO(BaseEstimator, TransformerMixin):
    """Leave-one-out target encoder.

    Replaces category values in categorical column(s) with the mean target
    (dependent variable) value for each category, using a leave-one-out
    strategy such that no sample's target value is used in computing the
    target mean which is used to replace that sample's category value.
    Can also optionally use a Bayesian estimation of the sample's mean target
    value, which sets a prior to the average of all encoding values, with the
    strength of that prior proportional to the ``bayesian_c`` parameter.

    """
    
    def __init__(self, cols=None, dtype='float64', nocol=None,
                 bayesian_c=None):
        """Leave-one-out target encoder.
        
        Parameters
        ----------
        cols : str or list of str
            Column(s) to target encode.  Default is to target encode all
            categorical columns in the DataFrame.
        dtype : str
            Datatype to use for encoded columns. Default = 'float64'
        bayesian_c : float
            Prior strength (C) for the Bayesian average
            https://en.wikipedia.org/wiki/Bayesian_average
        nocol : None or str
            Action to take if a col in ``cols`` is not in the dataframe to 
            transform.  Valid values:

            * None - ignore cols in ``cols`` which are not in dataframe
            * 'warn' - issue a warning when a column is not in dataframe
            * 'err' - raise an error when a column is not in dataframe
        """

        # Check types
        if cols is not None and not isinstance(cols, (list, str)):
            raise TypeError('cols must be None, or a list or a string')
        if isinstance(cols, list):
            if not all(isinstance(c, str) for c in cols):
                raise TypeError('each element of cols must be a string')
        if not isinstance(dtype, str):
            raise TypeError('dtype must be a string (e.g. \'float64\'')
        if nocol is not None and nocol not in ('warn', 'err'):
            raise ValueError('nocol must be None, \'warn\', or \'err\'')
        if bayesian_c is not None and not isinstance(bayesian_c, (float, int)):
            raise TypeError('bayesian_c must be None or float or int')

        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.dtype = dtype
        self.nocol = nocol
        if isinstance(bayesian_c, int):
            self.bayesian_c = float(bayesian_c)
        else:
            self.bayesian_c = bayesian_c
        self.overall_mean = None


    def fit(self, X, y):
        """Fit leave-one-out target encoder to X and y.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.
            
        Returns
        -------
        TargetEncoderLOO
            Returns self, the fit object.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X if str(X[col].dtype)=='object']

        # Check columns are in X
        if self.nocol == 'err':
            for col in self.cols:
                if col not in X:
                    raise ValueError('Column \''+col+'\' not in X')
        elif self.nocol == 'warn':
            for col in self.cols:
                if col not in X:
                    print('Column \''+col+'\' not in X')

        # Compute the overall mean
        self.overall_mean = np.mean(y)

        # Encode each element of each column
        self.sum_count = dict()
        for col in self.cols:
            self.sum_count[col] = dict()
            uniques = X[col].dropna().unique()
            for unique in uniques:
                ix = X[col]==unique
                self.sum_count[col][unique] = (y[ix].sum(),ix.sum())

        # Return the fit object
        return self

    
    def transform(self, X, y=None):
        """Perform the target encoding transformation.

        Uses leave-one-out target encoding when given training data, and uses
        normal target encoding when given test data.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        
        # Create output dataframe
        Xo = X.copy()

        # Bayesian C value
        if self.bayesian_c is not None:
            C = self.bayesian_c
            Cm = C*self.overall_mean

        # Use means from training data if passed test data
        if y is None:
            for col in self.sum_count:
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    if self.bayesian_c is None:
                        vals[X[col]==cat] = sum_count[0]/sum_count[1]
                    else: #use bayesian mean
                        vals[X[col]==cat] = (Cm+sum_count[0])/(C+sum_count[1])
                Xo[col] = vals

        # LOO target encode each column if this is training data
        else:
            for col in self.sum_count:
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    ix = X[col]==cat
                    if sum_count[1]<2:
                        vals[ix] = np.nan
                    else:
                        if self.bayesian_c is None:
                            vals[ix] = (sum_count[0]-y[ix])/(sum_count[1]-1)
                        else: #use Bayesian mean
                            vals[ix] = ((Cm+sum_count[0]-y[ix])
                                        /(C+sum_count[1]-1))
                Xo[col] = vals
            
        # Return encoded DataFrame
        return Xo
      
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data with leave-one-out target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class MultiTargetEncoderLOO(BaseEstimator, TransformerMixin):
    """Leave-one-out target encoder which handles multiple classes per sample.

    Replaces category values in categorical column(s) with the mean target
    (dependent variable) value for each category, using a leave-one-out
    strategy such that no sample's target value is used in computing the
    target mean which is used to replace that sample's category value.
    Can also optionally use a Bayesian estimation of the sample's mean target
    value, which sets a prior to the average of all encoding values, with the
    strength of that prior proportional to the ``bayesian_c`` parameter.

    Parameters
    ----------
    cols : str or list of str
        Column(s) to target encode.  Default is to target encode all
        categorical columns in the DataFrame.
    dtype : str
        Datatype to use for encoded columns. Default = 'float64'
    bayesian_c : float
        Prior strength (C) for the Bayesian average
        https://en.wikipedia.org/wiki/Bayesian_average
    sep : str
        Separator string which delimits the labels
    nocol : None or str
        Action to take if a col in ``cols`` is not in the dataframe to 
        transform.  Valid values:

        * None - (default) ignore cols which aren't in dataframe
        * 'warn' - issue a warning when a column is not in dataframe
        * 'err' - raise an error when a column is not in dataframe
    """
    
    def __init__(self, cols=None, dtype='float64', nocol=None,
                 bayesian_c=0.0, sep=','):

        # Check types
        if cols is not None and not isinstance(cols, (list, str)):
            raise TypeError('cols must be None, or a list or a string')
        if isinstance(cols, list):
            if not all(isinstance(c, str) for c in cols):
                raise TypeError('each element of cols must be a string')
        if not isinstance(dtype, str):
            raise TypeError('dtype must be a string (e.g. \'float64\'')
        if nocol is not None and nocol not in ('warn', 'err'):
            raise ValueError('nocol must be None, \'warn\', or \'err\'')
        if not isinstance(bayesian_c, (float, int)):
            raise TypeError('bayesian_c must be float or int')
        if not isinstance(sep, str):
            raise TypeError('sep must be a str')

        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.dtype = dtype
        self.nocol = nocol
        self.bayesian_c = float(bayesian_c)
        self.sep = sep
        self.overall_mean = None


    def _get_matches(self, data, val):
        data_o = data.astype('bool')
        for i in range(data.shape[0]):
            if isinstance(data.iloc[i], str):
                data_o.iloc[i] = val in data.iloc[i].split(self.sep)
            else:
                data_o.iloc[i] = False
        return data_o


    def fit(self, X, y):
        """Fit leave-one-out target encoder to X and y.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.
            
        Returns
        -------
        MultiTargetEncoderLOO
            Returns self, the fit object.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X if str(X[col].dtype)=='object']

        # Check columns are in X
        if self.nocol == 'err':
            for col in self.cols:
                if col not in X:
                    raise ValueError('Column \''+col+'\' not in X')
        elif self.nocol == 'warn':
            for col in self.cols:
                if col not in X:
                    print('Column \''+col+'\' not in X')

        # Compute the overall mean
        self.overall_mean = np.mean(y)

        # Count labels in each column
        self.sum_count = dict()
        for col in self.cols:
            self.sum_count[col] = dict()
            for i, tlist in enumerate(X[col].tolist()):
                if isinstance(tlist, str):
                    for val in tlist.split(self.sep):
                        if len(val)>0:
                            if val in self.sum_count[col]:
                                self.sum_count[col][val][0] += y.iloc[i]
                                self.sum_count[col][val][1] += 1
                            else:
                                self.sum_count[col][val] = [y.iloc[i], 1]

        # Return the fit object
        return self

    
    def transform(self, X, y=None):
        """Perform the target encoding transformation.

        Uses leave-one-out target encoding when given training data, and uses
        normal target encoding when given test data.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        
        # Create output dataframe
        Xo = X.copy()

        # Bayesian C value
        C = self.bayesian_c
        Cm = C*self.overall_mean

        # Flag for whether to perform LOO (depends on test vs train)
        lm = 0 if y is None else 1

        # Encode the columns
        for col in self.sum_count:
            vals = np.full(X.shape[0], 0.0)
            counts = np.full(X.shape[0], 0.0)
            for i, tlist in enumerate(X[col].tolist()):
                if isinstance(tlist, str) and len(tlist)>0:
                    for tval in tlist.split(self.sep):
                        SC = self.sum_count[col][tval]
                        if tval in self.sum_count[col] and (SC[1]>1 or lm==0):
                            val = (Cm+SC[0]-lm*y[i]) / (C+SC[1]-lm)
                            vals[i] += val
                            counts[i] += 1
            Xo[col] = vals/counts

        # Return encoded DataFrame
        return Xo
      
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data with leave-one-out target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class TextMultiLabelBinarizer(BaseEstimator, TransformerMixin):
    """Multi-label encode text data
    
    For each specified column, transform from a delimited list of text
    labels to a Nlabels-length binary vector.

    Parameters
    ----------
    cols : list of str
        Columns to encode.  Default is to encode all columns.
    dtype : str
        Datatype to use for encoded columns.
        Default = 'uint8'
    sep : str
        Separator character in the text data.  Default = ','
    labels : dict
        Labels for each column.  Dict with keys w/ column names and values
        w/ lists of labels
    nocol : None or str
        Action to take if a col in ``cols`` is not in the dataframe to 
        transform.  Valid values:
        * None - ignore cols in ``cols`` which are not in dataframe
        * 'warn' - issue a warning when a column is not in dataframe
        * 'err' - raise an error when a column is not in dataframe
    """
    
    def __init__(self, cols=None, dtype='uint8', nocol=None, 
                 sep=',', labels=None):
        """Multi-label encode text data"""

        # Check types
        if cols is not None and not isinstance(cols, (list, str)):
            raise TypeError('cols must be None, or a list or a string')
        if isinstance(cols, list):
            if not all(isinstance(c, str) for c in cols):
                raise TypeError('each element of cols must be a string')
        if not isinstance(dtype, str):
            raise TypeError('dtype must be a string (e.g. \'uint8\'')
        if nocol is not None and nocol not in ('warn', 'err'):
            raise ValueError('nocol must be None, \'warn\', or \'err\'')
        if not isinstance(sep, str):
            raise TypeError('sep must be a str')
        if labels is not None:
            if not isinstance(labels, dict):
                raise TypeError('labels must be a dict of lists of labels')
            for c, i in labels.items():
                if i is not None and not isinstance(i, list):
                    raise TypeError('labels must be a dict of lists of labels')
                if i is not None and not all(isinstance(t, str) for t in i):
                    raise TypeError('labels must be a dict of lists of labels')

        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.dtype = dtype
        self.nocol = nocol
        self.sep = sep
        if labels is None:
            self.labels = dict()
        else:
            self.labels = labels

        
    def fit(self, X, y=None):
        """Fit encoder to X and y.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
            
        Returns
        -------
        NullEncoder
            Returns self, the fit object.
        """
        
        # Encode all columns with any null values by default
        if self.cols is None:
            self.cols = [c for c in X]

        # Add Nones to labels
        for c in self.cols:
            if c not in self.labels:
                self.labels[c] = None

        # Check columns are in X
        if self.nocol == 'err':
            for col in self.cols:
                if col not in X:
                    raise ValueError('Column \''+col+'\' not in X')
        elif self.nocol == 'warn':
            for col in self.cols:
                if col not in X:
                    print('Column \''+col+'\' not in X')
                        
        # Return fit object
        return self


    def __onehot(self, series, unique_labels=None):
        """One-hot transform multi-label data
        
        Parameters
        ----------
        series : pandas Series
            Series containing text labels
        unique_labels : None or list of str
            Unique labels in the dataset.  Default is to generate list of 
            labels from unique labels in the data.
            
        Returns
        -------
        one_hot : ndarray
            Nsamples-by-Nclasses array of encoded data.  Each row is a 
            sample, and each column corresponds to a label.  If a sample
            has a given label, the value in that cell is 1, else it is 0.
        unique_labels : list of str
            Nclasses-length list of labels.
        """
        labels = [l.split(self.sep) for l in series.tolist()]
        if unique_labels is None:
            unique_labels = list(set(sum(labels, [])))
        one_hot = np.zeros((series.shape[0], len(unique_labels)))
        for i, sample in enumerate(labels):
            for label in sample:
                try:
                    one_hot[i, unique_labels.index(label)] = 1
                except:
                    pass
        return one_hot, unique_labels

        
    def transform(self, X, y=None):
        """Perform the null encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """

        # Add null indicator column for each original column
        Xo = X.copy()
        for col in self.cols:
            one_hot, labels = (
                self.__onehot(X[col], unique_labels=self.labels[col]))
            for i, label in enumerate(labels):
                Xo[col+'_'+label] = one_hot[:, i].astype(self.dtype)
            del Xo[col]

        # Return encoded dataframe
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data with null encoding.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class NhotEncoder(BaseEstimator, TransformerMixin):
    """N-hot encode multilabel data.
    
    Replaces column(s) containing lists of categories with binary columns.

    TODO: uhhh d'oh just realized this is basically the same thing as 
    TextMultiLabelBinarizer, should just keep one...  Probs this one.

    Parameters
    ----------
    cols : list of str
        Columns to encode
    sep : str
        Separator
    dtype : str
        Datatype to use for encoded columns. Default = 'uint8'

    Examples
    --------

    TODO

    """
    
    def __init__(self, cols, sep=',', dtype='float32', 
                 top_n=None, top_prc=None):

        # Check types
        if not isinstance(cols, (list, str)):
            raise TypeError('cols must be a str or list of str')
        if not isinstance(sep, str):
            raise TypeError('sep must be a str')
        if not isinstance(dtype, str):
            raise TypeError('dtype must be a str')
        if top_n is not None:
            if not isinstance(top_n, int):
                raise TypeError('top_n must be an int')
            if top_n < 1:
                raise TypeError('top_n must be at least 1')
        if top_prc is not None:
            if not isinstance(top_prc, float):
                raise TypeError('top_prc must be a float')
            if top_prc<0.0 or top_prc>1.0:
                raise TypeError('top_prc must be between 0 and 1')

        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.sep = sep
        self.dtype = dtype
        self.top_n = top_n
        self.top_prc = top_prc
        self.maps = None


    def _get_top(self, labels):
        """Get most frequent labels"""
        if self.top_n is not None and self.top_n < len(labels):
            df = pd.DataFrame([labels.keys(), labels.values()]).T
            df.sort_values(1, ascending=False, inplace=True)
            return df[0][:self.top_n].tolist()
        elif self.top_prc is not None:
            df = pd.DataFrame([labels.keys(), labels.values()]).T
            df.sort_values(1, ascending=False, inplace=True)
            return df[0][:int(self.top_prc*len(labels))].tolist()
        else:
            return list(labels.keys())

        
    def fit(self, X, y):
        """Fit N-hot encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.
            
        Returns
        -------
        NhotEncoder object
            Returns self, the fit object.
        """

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Store each unique value
        self.maps = dict()
        for col in self.cols:
            labels = dict()
            for vals in X[col].tolist():
                if isinstance(vals, str):
                    for val in vals.split(self.sep):
                        if val in labels:
                            labels[val] += 1
                        else:
                            labels[val] = 1
            self.maps[col] = self._get_top(labels)
        
        # Return fit object
        return self

        
    def transform(self, X, y=None):
        """Perform the one-hot encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, vals in self.maps.items():
            for val in vals:
                new_col = col+'_'+str(val)
                matches = np.full(X.shape[0], np.nan)
                for i, e in enumerate(Xo[col].tolist()):
                    if isinstance(e, str):
                        matches[i] = val in e.split(self.sep)
                Xo[new_col] = matches.astype(self.dtype)
            del Xo[col]
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data with one-hot encoding.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class JsonEncoder(BaseEstimator, TransformerMixin):
    """Replace columns with JSON data with columns containing values from 
    specific JSON fields.
    
    Parameters
    ----------
    fields : dict
        Fields from each column to extract.  Keys should be column names,
        and values should be lists of either:

        * str - field name to extract
        * (str, str, ???) tuple - first str is field name to extract, second
          str is conditional field to use, and third element is value to 
          compare to the vaue in the conditional field.  Will only encode the
          value from the field if the value of the condition field is equal
          to the third element.

    sep : str
        Separator to use in the output data when there are multiple values.
        Default = ','


    Examples
    --------

    TODO

    """
    
    def __init__(self, fields, sep=','):

        # Check types
        if not isinstance(fields, dict):
            raise TypeError('fields must be a dict')
        if not isinstance(sep, str):
            raise TypeError('sep must be a str')

        # Ensure all fields are correct type
        for col in fields:
            if not isinstance(fields[col], list):
                fields[col] = [fields[col]]
            for i, field in enumerate(fields[col]):
                if isinstance(field, str):
                    fields[col][i] = (field, None, None)
                elif isinstance(field, tuple):
                    if not (isinstance(field[0], str) and
                            isinstance(field[1], str)):
                        raise TypeError('fields must be (str,str,???) tuples')
                else:
                    raise TypeError('fields must be dict with values ' 
                                    'containing str or tuple of list of them')

        # Store parameters
        self.fields = fields
        self.sep = sep        


    def _extract_field(self, data, field, cond_field, cond_val):
        """Extract a field from JSON data

        Parameters
        ----------
        data : pandas Series
            With the json data
        field : str
            Key for the field(s) in the JSON data to extract
        cond_field : str
            Key for the field(s) in the JSON data to apply cond_fn to.
        cond_val : any value
            Value which cond_field must take in order to record the value from 
            field.
        """
        data_o = data.copy()
        for i in range(data.shape[0]):
            try:
                vals = []
                for jdict in ast.literal_eval(data.iloc[i]):
                    try:
                        if cond_field is None or jdict[cond_field] == cond_val:
                            vals += [str(jdict[field])]
                    except:
                        pass
                if len(vals) < 1:
                    data_o.iloc[i] = np.nan
                else:
                    data_o.iloc[i] = self.sep.join(vals)
            except:
                data_o.iloc[i] = np.nan
        return data_o


    def fit(self, X, y):
        """Fit the JSON encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.
            
        Returns
        -------
        JsonEncoder object
            Returns self, the fit object.
        """
        return self

        
    def transform(self, X, y=None):
        """Perform the JSON encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, fields in self.fields.items():
            for field in fields:
                if field[1] is None:
                    new_col = col+'_'+field[0]
                else:
                    new_col = col+'_'+field[1]+'_'+str(field[2])+'_'+field[0]
                Xo[new_col] = self._extract_field(X[col], field[0], 
                                                  field[1], field[2])                
            del Xo[col]
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data with JSON encoding.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class DateEncoder(BaseEstimator, TransformerMixin):
    """Replace datetime columns with date/time features.
    
    Parameters
    ----------
    cols : dict
        What columns to replace with date/time features, and what features to
        compute for each column.


    Examples
    --------

    TODO

    """
    
    def __init__(self, cols):

        # Valid features
        self.valids = ['year', 'month', 'day', 'hour', 'minute', 'second',
                       'week', 'weekofyear', 'dayofweek', 'dayofyear']

        # Check input
        if not isinstance(cols, dict):
            raise TypeError('cols must be a dict')
        for col in cols:
            if not isinstance(cols[col], tuple) or len(cols[col])!=2:
                raise TypeError('cols must be dict of len-2 tuples')
            if not isinstance(cols[col][0], str):
                raise TypeError('first element of cols values must be '
                                'str containing the date format')
            if isinstance(cols[col][1], str):
                cols[col][1] = [cols[col][1]]
            if not isinstance(cols[col][1], list):
                raise TypeError('second element of cols values must be '
                                'list containing date features to extract')
            if not all(isinstance(e, str) for e in cols[col][1]):
                raise TypeError('second element of cols values must be '
                                'list containing str')
            if not all(e in self.valids for e in cols[col][1]):
                raise ValueError('second element of cols values must be list '
                                'containing one of: '+', '.join(self.valids))

        # Store parameters
        self.cols = cols


    def _feat_from_str(self, data, feat):
        """Get datetime feature from string"""
        if feat == 'year':
            return data.dt.year
        elif feat == 'month':
            return data.dt.month
        elif feat == 'day':
            return data.dt.day
        elif feat == 'hour':
            return data.dt.hour
        elif feat == 'minute':
            return data.dt.minute
        elif feat == 'second':
            return data.dt.second
        elif feat == 'week':
            return data.dt.week
        elif feat == 'weekofyear':
            return data.dt.weekofyear
        elif feat == 'dayofweek':
            return data.dt.dayofweek
        elif feat == 'dayofyear':
            return data.dt.dayofyear
        else:
            raise ValueError('second element of cols values must be list '
                            'containing one of: '+', '.join(self.valids))


    def fit(self, X, y):
        """Nothing needs to be done here"""
        return self

        
    def transform(self, X, y=None):
        """Encode the date/times as features.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, features in self.cols.items():
            fmt = features[0]
            feats = features[1]
            dt_col = pd.to_datetime(X[col], format=fmt)
            for feat in feats:
                new_col = col+'_'+feat
                Xo[new_col] = self._feat_from_str(dt_col, feat)
            del Xo[col]
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data with JSON encoding.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class JoinTransformer(BaseEstimator, TransformerMixin):
    """Join a dataframe to the X data.
    
    Parameters
    ----------
    df : pandas DataFrame
        Table to join with the X data.  Is treated as the right table.
    left_on : str
        Column in the X data dataframe to join on
    right_on : str
        Column in ``df`` to join on.
    how : str {'left', 'right', 'outer', 'inner'}
        How to join the two tables.
        Default = 'left'
    delete_old : bool
        Whether to delete the old column (``X[left_on]``)
        Default = True

    Examples
    --------

    TODO

    """
    
    def __init__(self, df, left_on, right_on, how='left', delete_old=True):

        # Check types
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a pandas DataFrame')
        if not isinstance(left_on, str):
            raise TypeError('left_on must be a str')
        if not isinstance(right_on, str):
            raise TypeError('right_on must be a str')
        if not isinstance(how, str):
            raise TypeError('how must be a str')
        if how not in ['left', 'right', 'outer', 'inner']:
            raise TypeError('how must be left, right, outer, or inner')
        if not isinstance(delete_old, bool):
            raise TypeError('delete_old must be a bool')

        # Store parameters
        self.df = df
        self.left_on = left_on
        self.right_on = right_on
        self.how = how
        self.delete_old = delete_old


    def fit(self, X, y):
        """Nothing needs to be done here"""
        return self

        
    def transform(self, X, y=None):
        """Perform the join transformation.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        index_name = 'index' if Xo.index.name is None else Xo.index.name
        Xo = (Xo.reset_index()
                .merge(self.df, left_on=self.left_on, 
                       right_on=self.right_on, how=self.how)
                .set_index(index_name))
        if self.delete_old:
            if self.right_on in Xo:
                del Xo[self.right_on]
            if self.left_on in Xo:
                del Xo[self.left_on]
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class JoinColumns(BaseEstimator, TransformerMixin):
    """Join multiple columns.
    
    Parameters
    ----------
    cols : list of str
        Columns to join
    name : str
        Name for the new column
    sep : str
        Separator string to use.
        Default = ','
    delete_old : bool
        Whether to delete the columns merged to make the new columns.
        Default = True

    Examples
    --------

    TODO

    """
    
    def __init__(self, cols, name, sep=',', delete_old=True):

        # Check types
        if not isinstance(cols, (str, list)):
            raise TypeError('cols must be a str or list of str')
        if not isinstance(name, str):
            raise TypeError('name must be a str')
        if not isinstance(sep, str):
            raise TypeError('sep must be a str')
        if not isinstance(delete_old, bool):
            raise TypeError('delete_old must be a bool')

        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.name = name
        self.sep = sep
        self.delete_old = delete_old


    def fit(self, X, y):
        """Nothing needs to be done here"""
        return self

        
    def transform(self, X, y=None):
        """Join the columns
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        data = [Xo[c].tolist() for c in self.cols]
        Xo[self.name] = [self.sep.join([e[i] for e in data 
                                        if isinstance(e[i], str) 
                                        and len(e[i])>0]) 
                         for i in range(X.shape[0])]
        if self.delete_old:
            for col in self.cols:
                del Xo[col]
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class LambdaTransformer(BaseEstimator, TransformerMixin):
    """Transform column(s) with lambda functions
    
    Parameters
    ----------
    transforms : dict
        Dictionary of transforms to perform on each column.  Keys should be 
        column names, and values should be lambda functions.

    Examples
    --------

    TODO

    """
    
    def __init__(self, transforms):

        # Check types
        if not isinstance(transforms, dict):
            raise TypeError('transforms must be a dict')
        if not all(isinstance(e, str) for e in transforms.keys()):
            raise TypeError('transforms keys must be str')
        if not all(callable(e) for e in transforms.values()):
            raise TypeError('transforms values must be callable')

        # Store parameters
        self.transforms = transforms


    def fit(self, X, y):
        """Nothing needs to be done here"""
        return self

        
    def transform(self, X, y=None):
        """Perform the join transformation.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, transform in self.transforms.items():
            Xo[col] = Xo[col].apply(transform)
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



class LambdaFeatures(BaseEstimator, TransformerMixin):
    """Create new features.
    
    Parameters
    ----------
    features : dict
        Dictionary of features to create.  Keys should contain names for the 
        new columns, and values should be functions.  The function should take
        one argument (the X dataframe), and return a series containing
        the new feature.

    Examples
    --------

    TODO

    """
    
    def __init__(self, features):

        # Check types
        if not isinstance(features, dict):
            raise TypeError('features must be a dict')
        for col, feat in features.items():
            if not isinstance(col, str):
                raise TypeError('features keys must be str')
            if not callable(feat):
                raise TypeError('features values must be callable')

        # Store parameters
        self.features = features


    def fit(self, X, y):
        """Nothing needs to be done here"""
        return self

        
    def transform(self, X, y=None):
        """Create the new features.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, feat in self.features.items():
            Xo[col] = feat(Xo)
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)



def null_encode(X, y=None, cols=None, suffix='_isnull', dtype='uint8',
                delete_old=False):
    """Null encode columns in a DataFrame.
    
    For each column with null values, adds a column containing indicators
    as to whether each sample in original column is null.
        
    Parameters
    ----------
    cols : list of str
        Columns to null encode.  Default is to null encode all columns in
        the DataFrame which contain null values.
    suffix : str
        Suffix to append to original column names to create null indicator
        column names
    dtype : str
        Datatype to use for encoded columns.
        Default = 'uint8'
    delete_old : bool
        Whether to delete the old column which was encoded
        Default = False

    Returns
    -------
    pandas DataFrame
        Null encoded DataFrame
    """
    ne = NullEncoder(cols=cols, suffix=suffix, dtype=dtype, 
                     delete_old=delete_old)
    return ne.fit_transform(X, y)



def label_encode(X, y=None, cols=None):
    """Label encode columns in a DataFrame.
    
    Replaces categorical column(s) with integer labels for each unique
    category in original column.
    
    Parameters
    ----------
    cols : list of str
        Columns to label encode.  Default is to label encode all categorical
        columns in the DataFrame.

    Returns
    -------
    pandas DataFrame
        Label encoded DataFrame
    """
    le = LabelEncoder(cols=cols)
    return le.fit_transform(X, y)



def one_hot_encode(X, y=None, cols=None, reduce_df=False, dtype='uint8'):
    """One-hot encode columns in a DataFrame.
    
    Replaces categorical column(s) with binary columns for each unique value
    in original column.
    
    Parameters
    ----------
    cols : list of str
        Columns to one-hot encode.  Default is to one-hot encode all
        categorical columns in the DataFrame.
    reduce_df : bool
        Whether to use reduced degrees of freedom for encoding (that is,
        add N-1 one-hot columns for a column with N categories). E.g. for
        a column with categories A, B, and C: When reduce_df is True,
        A=[1, 0], B=[0, 1], and C=[0, 0].  When reduce_df is False, 
        A=[1, 0, 0], B=[0, 1, 0], and C=[0, 0, 1].
        Default = False
    dtype : str
        Datatype to use for encoded columns. Default = 'uint8'

    Returns
    -------
    pandas DataFrame
        One-hot encoded DataFrame
    """
    ohe = OneHotEncoder(cols=cols, reduce_df=reduce_df, dtype=dtype)
    return ohe.fit_transform(X, y)



def target_encode(X, y=None, cols=None, dtype='float64'):
    """Target encode columns in a DataFrame.
    
    Replaces category values in categorical column(s) with the mean target
    (dependent variable) value for each category.
    
    Parameters
    ----------
    cols : str or list of str
        Column(s) to target encode.  Default is to target encode all
        categorical columns in the DataFrame.
    dtype : str
        Datatype to use for encoded columns. Default = 'float64'

    Returns
    -------
    pandas DataFrame
        Target encoded DataFrame
    """
    te = TargetEncoder(cols=cols, dtype=dtype)
    return te.fit_transform(X, y)



def target_encode_cv(X,
                     y=None,
                     cols=None,
                     n_splits=3,
                     shuffle=True,
                     dtype='float64'):
    """Cross-fold target encode columns in a DataFrame.

    Replaces category values in categorical column(s) with the mean target
    (dependent variable) value for each category, using a cross-fold strategy
    such that no sample's target value is used in computing the target mean
    which is used to replace that sample's category value.
    
    Parameters
    ----------
    cols : str or list of str
        Column(s) to target encode.  Default is to target encode all
        categorical columns in the DataFrame.
    n_splits : int
        Number of cross-fold splits. Default = 3.
    shuffle : bool
        Whether to shuffle the data when splitting into folds.
    dtype : str
        Datatype to use for encoded columns. Default = 'float64'

    Returns
    -------
    pandas DataFrame
        Target encoded DataFrame
    """
    te = TargetEncoderCV(cols=cols, n_splits=n_splits, 
                         shuffle=shuffle, dtype=dtype)
    return te.fit_transform(X, y)



def target_encode_loo(X, y=None, cols=None, dtype='float64', bayesian_c=None):
    """Leave-one-out target encode columns in a DataFrame.

    Replaces category values in categorical column(s) with the mean target
    (dependent variable) value for each category, using a leave-one-out
    strategy such that no sample's target value is used in computing the
    target mean which is used to replace that sample's category value.
    
    Parameters
    ----------
    cols : str or list of str
        Column(s) to target encode.  Default is to target encode all
        categorical columns in the DataFrame.
    dtype : str
        Datatype to use for encoded columns. Default = 'float64'
    bayesian_c : float
        Prior strength (C) for the Bayesian average
        https://en.wikipedia.org/wiki/Bayesian_average

    Returns
    -------
    pandas DataFrame
        Target encoded DataFrame
    """
    te = TargetEncoderLOO(cols=cols, dtype=dtype, bayesian_c=bayesian_c)
    return te.fit_transform(X, y)



def text_multi_label_binarize(X, y=None,  cols=None, dtype='uint8', 
                              nocol=None, sep=',', labels=None):
    """Multi-label encode text data
    
    For each specified column, transform from a delimited list of text
    labels to a Nlabels-length binary vector.

    Parameters
    ----------
    cols : list of str
        Columns to encode.  Default is to encode all columns.
    dtype : str
        Datatype to use for encoded columns.
        Default = 'uint8'
    sep : str
        Separator character in the text data.  Default = ','
    labels : dict
        Labels for each column.  Dict with keys w/ column names and values
        w/ sets or lists of labels
    nocol : None or str
        Action to take if a col in ``cols`` is not in the dataframe to 
        transform.  Valid values:
        * None - ignore cols in ``cols`` which are not in dataframe
        * 'warn' - issue a warning when a column is not in dataframe
        * 'err' - raise an error when a column is not in dataframe

    Returns
    -------
    pandas DataFrame
        Encoded DataFrame
    """
    tmlb = TextMultiLabelBinarizer(cols=cols, dtype=dtype, nocol=nocol, 
                                   sep=sep, labels=labels)
    return tmlb.fit_transform(X, y)
