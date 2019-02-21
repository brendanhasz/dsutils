"""Data cleaning

"""


import pandas as pd


def remove_noninformative_cols(df):
    """Remove non-informative columns (all nan, or all same value).

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
            print('Removing '+col+' (all NaN)')
            del df[col]
        elif df[col].nunique()<2:
            print('Removing '+col+' (<2 unique values)')
            del df[col]


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
