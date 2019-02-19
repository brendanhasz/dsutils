"""Tests printing

"""

import numpy as np
import pandas as pd

from dsutils.printing import describe_df
from dsutils.printing import print_table


def test_print_table():
    """Tests printing.print_table"""
    print_table(['a', 'b', 'cccccc'], 
                [[1, 2.1234, 3, 4, 5], 
                 ['aaa', 'b', 'c', 'd', 'e'],
                 [1.2, 2, 3, 4, 5.5]])


def test_print_table_latex():
    """Tests printing.print_table w/ LaTeX output"""
    print_table(['a', 'b', 'cccccc'], 
                [[1, 2.1234, 3, 4, 5], 
                 ['aaa', 'b', 'c', 'd', 'e'],
                 [1.2, 2, 3, 4, 5.5]],
                latex=True)


def test_describe_df():
    """Tests printing.describe_df"""

    df = pd.DataFrame(np.random.randn(10,2),
                      columns=['a', 'b'])
    df['c'] = [1, 2, 3, 4, 5, 6, 7, np.nan, 9, 5]
    df['d'] = ['a', 'b', 'c', 'd', 'e',
               'f', 'g', 'h', 'i', 'e']
    df['the_nan_one'] = np.nan

    describe_df(df)
