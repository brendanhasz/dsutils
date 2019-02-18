"""Exploratory data analysis

"""


import pandas as pd


def countplot2d(x, y, data, log=False):
    """2-d countplot

    Parameters
    ----------
    x : str
        Column in data for x
    y : str
        Column in data for y
    df : pandas DataFrame
        Data
    log : bool
        Whether to plot the log count
    """

    # Compute bins automatically
    # if <100 bins use use linspaced

    # TODO: find xtick labels from unique vals in x and y 
    # and apss to histogram2d

    N, e1, e2 = np.histogram2d(df[x], df[y]
                               bins=[[-0.5, 0.5, 1.5], [-0.5, 0.5, 1.5]])
    sns.heatmap(N.astype('int64'), annot=True, fmt='d')
    plt.xlabel(x_i)
    plt.ylabel(y_i)