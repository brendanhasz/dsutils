"""Tests eda module

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dsutils.eda import countplot2d


def test_countplot2d_cat_cat(plot):
    """Tests eda.countplot2d w/ 2 categorical variables"""

    df = pd.DataFrame()
    df['A'] = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    df['B'] = np.array([1, 1, 0, 1, 1, 1])

    countplot2d('A', 'B', df)
    if plot:
        plt.show()


def test_countplot2d_cat_cont(plot):
    """Tests eda.countplot2d w/ a categorical vs a continuous variable"""

    df = pd.DataFrame()
    df['A'] = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    df['B'] = np.random.randn(6)

    countplot2d('A', 'B', df, bins_y=3)
    if plot:
        plt.show()


def test_countplot2d_cont_cont(plot):
    """Tests eda.countplot2d w/ 2 continuous variables"""

    df = pd.DataFrame()
    df['A'] = np.random.randn(6)
    df['B'] = np.random.randn(6)

    countplot2d('A', 'B', df, bins_x=3, bins_y=3)
    if plot:
        plt.show()
