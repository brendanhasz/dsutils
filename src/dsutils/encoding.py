"""Encoding

"""


import pandas as pd


def one_hot_encode(df, cols=None):
    """One-hot encode columns.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe from which to one-hot encode columns
    cols : list of str
        Columns in df to one-hot encode

    Returns
    -------
        Nothing, modifies df in-place.
    """

    # Do for all "object" columns if not specified
    if cols is None:
        cols = [col for col in df if str(df[col].dtype)=='object']
    if len(cols) == 0:
        return

    # Make list if not
    if isinstance(cols, str):
        cols = [cols]

    # One-hot encode each column
    for col in cols:
        uniques = df[col].unique()
        for u_val in uniques:
            new_col = col+'_'+str(u_val)
            df[new_col] = (df[col] == u_val).astype('uint8')

    # Delete original columns from dataframe
    for col in cols:
        del df[col]
