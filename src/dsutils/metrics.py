"""Metrics

* :func:`.mutual_information`
* :func:`.q_mut_info`

"""

import numpy as np
import pandas as pd

from .transforms import quantile_transform



def root_mean_squared_error(y_true, y_pred):
    """Root mean squared error regression loss"""
    return np.sqrt(np.mean(np.square(y_true-y_pred)))


def mutual_information(xi, yi, res=20):
    """Compute the mutual information between two vectors

    TODO: support categorical inputs via cat_x and cat_y bool args

    Parameters
    ----------
    xi : ndarray or series
    yi : ndarray or series

    """
    ix = ~(np.isnan(xi) | np.isinf(xi) | np.isnan(yi) | np.isinf(yi))
    x = xi[ix]
    y = yi[ix]
    N, xe, ye = np.histogram2d(x, y, res)
    Nx, _ = np.histogram(x, xe)
    Ny, _ = np.histogram(y, ye)
    N = N / len(x) #normalize
    Nx = Nx / len(x)
    Ny = Ny / len(y)
    Ni = np.outer(Nx, Ny)
    Ni[Ni == 0] = np.nan
    N[N == 0] = np.nan
    return np.nansum(N * np.log(N / Ni))
    


def q_mut_info(x, y):
    """Compute the mutual information between two quantile-transformed vectors

    Parameters
    ----------
    x : ndarray or series
    y : ndarray or series

    """
    return mutual_information(quantile_transform(x),
                              quantile_transform(y))




def columnwise_mut_info(y, df, res=20, q_transform=True):
    """Print the mutual information between target and other cols.

    Parameters
    ----------
    y : str
        Column in df to use as the target
    df : pandas DataFrame
        DataFrame with the values
    q_transform : bool
        Whether to quantile-transform values before computing mut info

    Returns
    -------
    pandas DataFrame
        Prints the mutual information and returns a dataframe of size
        (Ncols,2).
    """

    # Check inputs 
    if not isinstance(y, str):
        raise TypeError('y must be a string with the column to use as target')
    if not isinstance(y, str):
        raise TypeError('y must be a string with the column to use as target')

    # Compute the mutual info for each column
    mis = []
    cols = []
    for col in df:
        if col == y: continue 
        if q_transform:
            mi = q_mut_info(df[y], df[col])
        else:
            mi = mutual_information(df[y], df[col])
        cols.append(col)
        mis.append(mi)

    # Print the mutual information for each column
    print_table(['Column', 'Mutual Information'], [cols, mis])

    # Return dataframe w/ the values
    mi_df = pd.DataFrame()
    mi_df['Column'] = cols
    mi_df['Mutual Information'] = mis
    return mi_df
