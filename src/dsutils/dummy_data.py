"""Dummy data

* :func:`.permutation_importance`
* :func:`.permutation_importance_cv`

"""


import numpy as np
import pandas as pd



def make_categorical_regression(n_samples=100,
                                n_features=10,
                                n_informative=10,
                                n_categories=10,
                                noise=1.0):
    """Generate a regression problem with only categorical features.
  
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of categorical features to generate
    n_informative : int
        Number of features which carry information about the target
    n_categories : int or list or ndarray
        Number of categories per feature.
    noise : float
        Noise to add to target

    Returns
    -------
    X : pandas DataFrame of shape (n_samples, n_features)
        Categorical features.
    y : pandas Series of shape (n_samples,)
        Target variable.
    """

    # Check inputs
    if not isinstance(n_samples, int):
        raise TypeError('n_samples must be an int')
    if n_samples < 2:
        raise ValueError('n_samples must be one or greater')
    if not isinstance(n_features, int):
        raise TypeError('n_features must be an int')
    if n_features < 2:
        raise ValueError('n_features must be one or greater')
    if not isinstance(n_informative, int):
        raise TypeError('n_informative must be an int')
    if n_informative < 2:
        raise ValueError('n_informative must be one or greater')
    if not isinstance(n_categories, int):
        raise TypeError('n_categories must be an int')
    if n_categories < 2:
        raise ValueError('n_categories must be one or greater')
    if not isinstance(noise, float):
        raise TypeError('noise must be a float')
    if noise < 0:
        raise ValueError('noise must be positive')
        
    # Generate random categorical data
    categories = np.random.randint(n_categories,
                                   size=(n_samples, n_features))
    
    # Generate random values for each category
    cat_vals = np.random.randn(n_categories, n_features)
    
    # Set non-informative columns' effect to 0
    cat_vals[:,:(n_features-n_informative)] = 0
    
    # Compute target variable from those categories and their values
    y = np.zeros(n_samples)
    for iC in range(n_features):
      y += cat_vals[categories[:,iC], iC]
    
    # Add noise
    y += noise*np.random.rand(n_samples)
    
    # Generate dataframe from categories
    cat_strs = [''.join([chr(ord(c)+49) for c in str(n)]) 
                for n in range(n_categories)]
    X = pd.DataFrame()
    for iC in range(n_features):
        col_str = 'feature_'+str(iC)
        X[col_str] = [cat_strs[i] for i in categories[:,iC]]
            
    # Generate series from target
    y = pd.Series(data=y, index=X.index)
    
    # Return features and target
    return X, y
