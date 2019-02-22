"""Data cleaning

* :func:`.remove_noninformative_cols`
* :func:`.categorical_to_int`

"""



import pandas as pd
from hashlib import sha256



def remove_duplicate_cols(df):
    """Remove duplicate columns from a DataFrame.

    Uses hashing to quickly determine where there are duplicate columns, 
    and removes the duplicates.
    
    Parameters
    ----------
    df : pandas DataFrame or a list of them
        Dataframe from which to remove the non-informative columns

    Returns
    -------
        Nothing, deletes the columns in-place.
    """

    # Hash columns
    hashes = dict()
    for col in df:
        hashes[col] = sha256(df[col].values).hexdigest()
        
    # Get list of duplicate column lists
    Ncol = df.shape[1] #number of columns
    dup_list = []
    dup_labels = -np.ones(Ncol)
    for i1 in range(Ncol):
        if dup_labels[i1]<0: #if not already merged,
            col1 = df.columns[i1]
            t_dup = [] #list of duplicates matching col1
            for i2 in range(i1+1, Ncol):
                col2 = df.columns[i2]
                if ( dup_labels[i2]<0 #not already merged
                     and hashes[col1]==hashes[col2] #hashes match
                     and df[col1].equals(df[col2])): #cols are equal
                    #then this is actually a duplicate
                    t_dup.append(col2)
                    dup_labels[i2] = i1
            if len(t_dup)>0: #duplicates of col1 were found!
                t_dup.append(col1)
                dup_list.append(t_dup)
            
    # Remove duplicate columns
    for iM in range(len(dup_list)):
        o_col = dup_list[-1] #original column
        for iD in range(len(dup_list[iM])-1):
            t_dup = dup_list[iM][iD]
            print('Removing \''+t_dup+'\' (duplicate of \''+o_col'\')')
            df.drop(columns=t_dup, inplace=True)



def remove_noninformative_cols(df):
    """Remove non-informative columns (all nan, all same value, or duplicate).

    Parameters
    ----------
    df : pandas DataFrame or a list of them
        Dataframe from which to remove the non-informative columns

    Returns
    -------
        Nothing, deletes the columns in-place.
    """

    # Check inputs
    if not isinstance(df, (list, pd.DataFrame)):
        raise TypeError('df must be a pandas DataFrame or list of them')

    # Perform on each element if passed list
    if isinstance(df, list):
        for i in range(len(df)):
            print('DataFrame '+str(i))
            remove_noninformative_cols(df[i])

    # Remove non-informative columns
    for col in df:
        if df[col].isnull().all():
            print('Removing \''+col+'\' (all NaN)')
            del df[col]
        elif df[col].nunique()<2:
            print('Removing \''+col+'\' (<2 unique values)')
            del df[col]

    # Remove duplicate columns
    remove_duplicate_cols(df)



def categorical_to_int(df, cols=None):
    """Convert categorical columns of a DataFrame to unique integers, inplace.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame in which to convert columns
    cols : list
    """

    # Do for all "object" columns if not specified
    if cols is None:
        cols = [col for col in df if str(df[col].dtype)=='object']
    if len(cols) == 0:
        return

    # Make list if not
    if isinstance(cols, str):
        cols = [cols]

    # Map each column
    maps = dict()
    for col in cols:

        # Create the map from objects to integers
        maps[col] = dict(zip(
            df[col].values, 
            df[col].astype('category').cat.codes.values
        ))

        # Determine appropriate datatype
        max_val = max(maps[col].values())
        if df[col].isnull().any(): #nulls, so have to use float!
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

        # Map the column
        df[col] = df[col].map(maps[col]).astype(dtype)

    # Return the maps used
    return maps
