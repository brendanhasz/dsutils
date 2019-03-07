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
                                imbalance=0.0,
                                noise=1.0,
                                n_cont_features=0,
                                cont_weight=0.1,
                                interactions=0.0):
    """Generate a regression problem with categorical features.
  
    Parameters
    ----------
    n_samples : int > 0
        Number of samples to generate
        Default = 100
    n_features : int > 0
        Number of categorical features to generate
        Default = 10
    n_informative : int >= 0
        Number of features to carry information about the target.
        Default = 10
    n_categories : int > 0
        Number of categories per feature.  Default = 10
    imbalance : float > 0
        How much imbalance there is in the number of occurrences of
        each category.  Larger values yield a higher concentration
        of samples in only a few categories.  An imbalance of 0 
        yields the same number of samples in each category.
        Default = 0.0
    noise : float > 0
        Noise to add to target.  Default = 1.0
    n_cont_features : int >= 0
        Number of continuous (non-categorical) features.
        Default = 0
    cont_weight : float > 0
        Weight of the continuous variables' effect.
        Default = 0.1
    interactions : float >= 0 and <= 1
        Proportion of the variance due to interaction effects.
        Note that this only adds interaction effects between the 
        categorical features, not the continuous features.
        Default = 0.0
        
    Returns
    -------
    X : pandas DataFrame
        Features.  Of shape (n_samples, n_features+n_cont_features)
    y : pandas Series of shape (n_samples,)
        Target variable.
    """
    
    
    def beta_binomial(n, a, b):
        """Beta-binomial probability mass function.
        
        Parameters
        ----------
        n : int
            Number of trials
        a : float > 0
            Alpha parameter
        b : float > 0
            Beta parameter
            
        Returns
        -------
        ndarray of size (n,)
            Probability mass function.
        """
        from scipy.special import beta
        from scipy.misc import comb
        k = np.arange(n+1)
        return comb(n, k)*beta(k+a, n-k+b)/beta(a, b)


    # Check inputs
    if not isinstance(n_samples, int):
        raise TypeError('n_samples must be an int')
    if n_samples < 1:
        raise ValueError('n_samples must be one or greater')
    if not isinstance(n_features, int):
        raise TypeError('n_features must be an int')
    if n_features < 1:
        raise ValueError('n_features must be one or greater')
    if not isinstance(n_informative, int):
        raise TypeError('n_informative must be an int')
    if n_informative < 0:
        raise ValueError('n_informative must be non-negative')
    if not isinstance(n_categories, int):
        raise TypeError('n_categories must be an int')
    if n_categories < 1:
        raise ValueError('n_categories must be one or greater')
    if not isinstance(imbalance, float):
        raise TypeError('imbalance must be a float')
    if imbalance < 0:
        raise ValueError('imbalance must be non-negative')
    if not isinstance(noise, float):
        raise TypeError('noise must be a float')
    if noise < 0:
        raise ValueError('noise must be positive')
    if not isinstance(n_cont_features, int):
        raise TypeError('n_cont_features must be an int')
    if n_cont_features < 0:
        raise ValueError('n_cont_features must be non-negative')
    if not isinstance(cont_weight, float):
        raise TypeError('cont_weight must be a float')
    if cont_weight < 0:
        raise ValueError('cont_weight must be non-negative')
    if not isinstance(interactions, float):
        raise TypeError('interactions must be a float')
    if interactions < 0:
        raise ValueError('interactions must be non-negative')
        
    # Generate random categorical data (using category probs drawn
    # from a beta-binomial dist w/ alpha=1, beta=imbalance+1)
    cat_probs = beta_binomial(n_categories-1, 1.0, imbalance+1)
    categories = np.empty((n_samples, n_features), dtype='uint64')
    for iC in range(n_features):
        categories[:,iC] = np.random.choice(np.arange(n_categories),
                                            size=n_samples,
                                            p=cat_probs)
        
    # Generate random values for each category
    cat_vals = np.random.randn(n_categories, n_features)
    
    # Set non-informative columns' effect to 0
    cat_vals[:,:(n_features-n_informative)] = 0
    
    # Compute target variable from categories and their values
    y = np.zeros(n_samples)
    for iC in range(n_features):
        y += (1.0-interactions) * cat_vals[categories[:,iC], iC]
      
    # Add interaction effects
    if interactions > 0:
        for iC1 in range(n_informative):
            for iC2 in range(iC1+1, n_informative):
                int_vals = np.random.randn(n_categories,
                                           n_categories)
                y += interactions * int_vals[categories[:,iC1],
                                             categories[:,iC2]]
    
    # Add noise
    y += noise*np.random.randn(n_samples)
    
    # Generate dataframe from categories
    cat_strs = [''.join([chr(ord(c)+49) for c in str(n)]) 
                for n in range(n_categories)]
    X = pd.DataFrame()
    for iC in range(n_features):
        col_str = 'categorical_'+str(iC)
        X[col_str] = [cat_strs[i] for i in categories[:,iC]]
        
    # Add continuous features
    for iC in range(n_cont_features):
        col_str = 'continuous_'+str(iC)
        X[col_str] = cont_weight*np.random.randn(n_samples)
        y += np.random.randn()*X[col_str]
                    
    # Generate series from target
    y = pd.Series(data=y, index=X.index)
    
    # Return features and target
    return X, y
    