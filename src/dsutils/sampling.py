"""Sampling

* :func:`.add_negative_samples`

"""



import numpy as np
import pandas as pd



def add_negative_samples(A, B, k=5):
    """Add pairs which do not exist to positive pairs.
    
    If `A` and `B` are two corresponding lists , this function
    returns a table with two copies of elements in `A`.
    For the first copy, corresponding elements in `B` are unchaged.
    However, for the second copy, elements in `B` are elements
    which exist in `B`, but the corresponding `A`-`B` pair
    does not exist in the original pairs.
    
    Parameters
    ----------
    A : list or ndarray or pandas Series
        Indexes
    B : list or ndarray or pandas Series
        Values
    k : int
        Number of negative samples per positive sample.
        Default=5
    
    Returns
    -------
    Ao : list
        Output indexes w/ both positive and negative samples.
    Bo : list
        Output indexes w/ both positive and negative samples.
    E : list
        Whether the corresponding `Ao`-`Bo` pair exists (1) or
        does not (0) in the original input data.
    """
    
    # Convert to lists
    if isinstance(A, (np.ndarray, pd.Series)):
        A = A.tolist()
    if isinstance(B, (np.ndarray, pd.Series)):
        B = B.tolist()
    
    # Construct a dict of pairs for each unique value in A
    df = pd.DataFrame()
    df['A'] = A
    df['B'] = B
    to_sets = lambda g: set(g.values.tolist())
    pairs = df.groupby('A')['B'].apply(to_sets).to_dict()
    
    # Randomize B
    uB = np.unique(B) # unique elements of B
    nB = np.random.choice(uB, k*len(A)).tolist() #(hopefully) negative samples
        
    # Ensure pairs do not exist
    for i in range(k*len(A)):
        while nB[i] in pairs[A[i%len(A)]]:
            nB[i] = np.random.choice(uB)
            # TODO: this will run forever if there's an element 
            # in A which has pairs w/ *all* unique values of B...
            
    # Construct output lists
    Ao = A*(k+1)
    Bo = B+nB
    E = [1]*len(A) + [0]*(k*len(A))
    return Ao, Bo, E
    