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



def test_root_mean_squared_error():
    """Tests dsutils.metrics.root_mean_squared_error"""

    # Should return RMS of scalars
    rmse = root_mean_squared_error(0, 1)
    assert isinstance(rmse, float)
    assert rmse == 1.0

    # Should return RMS of scalars
    a = np.array([0.0, 0.0, -2.0])
    b = np.array([1.0, 2.0, 2.0])
    rmse = root_mean_squared_error(a, b)
    assert isinstance(rmse, float)
    assert rmse == np.sqrt(7.0)



def test_mutual_information():
    """Tests dsutils.metrics.mutual_information"""

    # Complete mutual information
    a = np.array([0, 1])
    b = np.array([0, 1])
    mi = mutual_information(a, b)
    assert isinstance(mi, float)
    assert mi == np.log(2.0)

    # No mutual information
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    mi = mutual_information(a, b)
    assert isinstance(mi, float)
    assert mi == 0.0



def test_q_mut_info():
    """Tests dsutils.metrics.q_mut_info"""

    # Highly skewed dists should show higher MI w/ transform
    # (b/c of discretization)
    a = np.exp(np.linspace(0, 10, 100))
    b = np.exp(np.linspace(0, 10, 100))
    mi = mutual_information(a, b)
    qmi = q_mut_info(a, b)
    assert isinstance(qmi, float)
    assert qmi > mi

    # Non-skewed dists should show no difference
    # (b/c of discretization)
    a = np.linspace(0, 10, 3)
    b = np.linspace(0, 10, 3)
    mi = mutual_information(a, b)
    qmi = q_mut_info(a, b)
    assert isinstance(qmi, float)
    assert qmi == mi



def test_columnwise_mut_info():
    """Tests dsutils.metrics.q_mut_info"""

    # Dummy data
    N = 200
    df = pd.DataFrame()
    df['a'] = np.linspace(0, 1, N)
    df['b'] = np.linspace(0, 1, N)
    df['c'] = np.sin(np.linspace(0, 10, N)) + 0.2*np.random.randn(N)
    df['d'] = np.random.randn(N)

    # Should compute MI between y col and all others
    mi_df = columnwise_mut_info('a', df)
    assert isinstance(mi_df, pd.DataFrame)
    assert mi_df.shape[0] == 3
    assert mi_df.shape[1] == 2
    mi_s = (mi_df.set_index('Column'))['Mutual Information']
    assert mi_s['b'] > mi_s['c']
    assert mi_s['c'] > mi_s['d']



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
