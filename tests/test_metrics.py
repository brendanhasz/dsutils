"""Tests metrics module

"""


import numpy as np
import pandas as pd

from dsutils.metrics import root_mean_squared_error
from dsutils.metrics import mutual_information
from dsutils.metrics import q_mut_info
from dsutils.metrics import columnwise_mut_info
from dsutils.metrics import jaccard_similarity
from dsutils.metrics import jaccard_similarity_df



# TODO: root_mean_squared_error



# TODO: mutual_information



# TODO: q_mut_info



# TODO: columnwise_mut_info



def test_jaccard_similarity():
    """Tests dsutils.metrics.jaccard_similarity"""

    # Complete overlap
    js = jaccard_similarity(['a', 'b', 'c'], ['a', 'b', 'c'])
    assert js == 1.0

    # Complete overlap but different order
    js = jaccard_similarity(['a', 'b', 'c'], ['c', 'a', 'b'])
    assert js == 1.0

    # Completely non-overlaping
    js = jaccard_similarity(['a', 'b', 'c'], ['d', 'e', 'f'])
    assert js == 0.0

    # Partially overlaping
    js = jaccard_similarity(['a', 'b', 'c'], ['a', 'e', 'f'])
    assert js == 1.0/5.0

    # Partially overlaping differen lengths
    js = jaccard_similarity(['a', 'b', 'c'], ['a', 'e', 'f', 'g'])
    assert js == 1.0/6.0



def test_jaccard_similarity_df():
    """Tests dsutils.metrics.jaccard_similarity_df"""

    # Default behavior
    df = pd.DataFrame()
    df['col1'] = ['a b c',
                  'a b c',
                  'a b c',
                  'a b c',
                  'a b c']
    df['col2'] = ['a b c',
                  'c a b',
                  'd e f',
                  'a e f',
                  'a e f g']
    js = jaccard_similarity_df(df, 'col1', 'col2')
    assert js[0] == 1.0
    assert js[1] == 1.0
    assert js[2] == 0.0
    assert js[3] == 1.0/5.0
    assert js[4] == 1.0/6.0

    # Using custom separator
    df = pd.DataFrame()
    df['col1'] = ['a,b,c',
                  'a,b,c',
                  'a,b,c',
                  'a,b,c',
                  'a,b,c']
    df['col2'] = ['a,b,c',
                  'c,a,b',
                  'd,e,f',
                  'a,e,f',
                  'a,e,f,g']
    js = jaccard_similarity_df(df, 'col1', 'col2', sep=',')
    assert js[0] == 1.0
    assert js[1] == 1.0
    assert js[2] == 0.0
    assert js[3] == 1.0/5.0
    assert js[4] == 1.0/6.0
