"""Utility functions

* :func:`.meshgridn`
* :func:`.linspacen`

"""


import numpy as np
import pandas as pd



def meshgridn(E):
    """Create an N-dimensional grid

    Parameters
    ----------
    E : list of ndarray
        List of grid edges in each dimension.

    Returns
    -------
    G : ndarray
        The grid vertexes.  2D array of size (N_grid_points, dimensions)
    """

    # Base case: 1D
    if len(E) == 1:
        return E[0].reshape(-1, 1)

    # Multidim
    else:
        gridn1 = np.tile(meshgridn(E[1:]), (len(E[0]), 1))
        grid0 = np.repeat(E[0].reshape(-1), len(E[0])).reshape(-1, 1)
        return np.concatenate((grid0, gridn1), axis=1)



def linspacen(lb, ub, N):
    """Create a linearly-spaced grid in N dimensions

    Parameters
    ----------
    lb : list or ndarray
        Lower bound in each dimension
    ub : list or ndarray
        Upper bound in each dimension
    N : int
        Number of points in the grid.  Will create the smallest grid with at 
        least N points.

    Returns
    -------
    G : ndarray
        The grid vertexes.  2D array of size (N, len(lb))

    Examples
    --------
    >> linspacen([0, 2], [1, 3], 9)
    [[0.0, 2.0],
     [0.0, 2.5],
     [0.0, 3.0],
     [0.5, 2.0],
     [0.5, 2.5],
     [0.5, 3.0],
     [1.0, 2.0],
     [1.0, 2.5],
     [1.0, 3.0]]

    When an entire grid cannot be made with only N points, the smallest grid
    with at least N points is returned:

    >> linspacen([0, 2], [1, 3], 5)
    [[0.0, 2.0],
     [0.0, 2.5],
     [0.0, 3.0],
     [0.5, 2.0],
     [0.5, 2.5],
     [0.5, 3.0],
     [1.0, 2.0],
     [1.0, 2.5],
     [1.0, 3.0]]

    """

    # Check inputs
    if not isinstance(lb, (list, np.ndarray)):
        raise TypeError('lb must be a list of lower bounds in each dim')
    if not isinstance(ub, (list, np.ndarray)):
        raise TypeError('ub must be a list of upper bounds in each dim')
    if len(lb) != len(ub):
        raise ValueError('lb and ub must be same length')
    if not isinstance(N, int):
        raise TypeError('N must be an integer')

    # Compute the grid
    Nd = len(lb) #number of dimensions
    Npd = int(np.ceil(np.power(N, 1.0/Nd))) #edges per dimension
    edges = [np.linspace(lb[i], ub[i], Npd) for i in range(Nd)]
    return meshgridn(edges)
