"""Metrics

* :func:`.mutual_information`
* :func:`.q_mut_info`

"""

import numpy as np

from .transforms import quantile_transform


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