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