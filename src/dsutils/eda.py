"""Exploratory data analysis

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def countplot2d(x, y, df, log=False, bins_x=None, bins_y=None, annot=True):
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
    bins_x : None or int
        Number of bins for x.  If None, assumes data[x] is categorical, and
        uses a bin for each category (the default).
    bins_y : None or int
        Number of bins for y.  If None, assumes data[y] is categorical, and
        uses a bin for each category (the default).
    annot : bool
        Whether to display the count value in each element of the matrix.
    """

    def get_bins(series, bins):
        if str(series.dtype) == 'object' or bins is None:
            cat_s = series.astype('category')
            data = cat_s.cat.codes
            edges = np.linspace(-0.5, max(data)+0.5, max(data)+2)
            labels = cat_s.cat.categories
        else:
            data = series.values
            edges = np.linspace(min(data), max(data), bins+1)
            labels = edges[:-1] + np.diff(edges)
            labels = ['%0.3g' % e for e in labels.tolist()]
        return data, edges, labels

    # Get bins for x and y
    x_d, x_e, x_l = get_bins(df[x], bins_x)
    y_d, y_e, y_l = get_bins(df[y], bins_y)

    # Compute the 2D histogram
    N, _, _ = np.histogram2d(x_d, y_d, bins=[x_e, y_e])
    Xc, Yc = np.meshgrid(x_l, y_l)

    # Create dataframe to map
    hm = pd.DataFrame()
    hm['count'] = N.T.flatten().astype('uint64')
    hm[x] = Xc.flatten()
    hm[y] = Yc.flatten()
    hm = hm.pivot(y, x, 'count')

    # Plot the heatmap
    sns.heatmap(hm, annot=annot, cbar_kws={'label': 'Count'})
