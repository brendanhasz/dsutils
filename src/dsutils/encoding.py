"""Encoding

Provides sklearn-compatible transformer classes for categorical encoding:

* :class:`.NullEncoder`
* :class:`.LabelEncoder`
* :class:`.OneHotEncoder`
* :class:`.TargetEncoder`
* :class:`.TargetEncoderCV`
* :class:`.TargetEncoderLOO`

Also provides functions to simply return an encoded DataFrame:

* :func:`.null_encode`
* :func:`.label_encode`
* :func:`.one_hot_encode`
* :func:`.target_encode`
* :func:`.target_encode_cv`
* :func:`.target_encode_loo`

"""



import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold



class NullEncoder(BaseEstimator, TransformerMixin):
    """Null encoder.
    
    For each column with null values, adds a column containing indicators
    as to whether each sample in original column is null.

    """
    
    def __init__(self, cols=None, suffix='_isnull', dtype='uint8', 
                 nocol=None):
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

        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.suffix = suffix
        self.dtype = dtype
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
            self.maps[col] = dict(zip(
                X[col].values,
                X[col].astype('category').cat.codes.values
            ))
                        
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
            if Xo[col].isnull().any(): #nulls, so need to use float!
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

    """
    
    def __init__(self, cols=None, dtype='float64', nocol=None):
        """Leave-one-out target encoder.
        
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
            raise TypeError('dtype must be a string (e.g. \'float64\'')
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

        # Encode each element of each column
        self.sum_count = dict()
        for col in self.cols:
            self.sum_count[col] = dict()
            uniques = X[col].unique()
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

        # Use means from training data if passed test data
        if y is None:
            for col in self.sum_count:
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    vals[X[col]==cat] = sum_count[0]/sum_count[1]
                Xo[col] = vals

        # LOO target encode each column if this is training data
        else:
            for col in self.sum_count:
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    ix = X[col]==cat
                    vals[ix] = (sum_count[0]-y[ix])/(sum_count[1]-1)
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



def null_encode(X, y=None, cols=None, suffix='_isnull', dtype='uint8'):
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

    Returns
    -------
    pandas DataFrame
        Null encoded DataFrame
    """
    ne = NullEncoder(cols=cols, suffix=suffix, dtype=dtype)
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



def target_encode_loo(X, y=None, cols=None, dtype='float64'):
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

    Returns
    -------
    pandas DataFrame
        Target encoded DataFrame
    """
    te = TargetEncoderLOO(cols=cols, dtype=dtype)
    return te.fit_transform(X, y)
