"""Tests encoding

"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from dsutils.encoding import null_encode
from dsutils.encoding import label_encode
from dsutils.encoding import one_hot_encode
from dsutils.encoding import target_encode
from dsutils.encoding import target_encode_cv
from dsutils.encoding import target_encode_loo
from dsutils.encoding import text_multi_label_binarize
from dsutils.encoding import TargetEncoder
from dsutils.encoding import TargetEncoderCV
from dsutils.encoding import NhotEncoder
from dsutils.encoding import JsonEncoder


def test_null_encode():
    """Tests encoding.null_encode"""

    # Should not change df if no nulls
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = np.random.randn(10)
    df = null_encode(df)
    assert 'a' in df
    assert 'b' in df
    assert df.shape[0] == 10
    assert df.shape[1] == 2

    # Should add one col for col in cols
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = np.random.randn(10)
    df.loc[2, 'a'] = np.nan
    df = null_encode(df, cols=['a'])
    assert 'a' in df
    assert 'b' in df
    assert 'a_isnull' in df
    assert df.shape[0] == 10
    assert df.shape[1] == 3

    # Should do all cols if none specified
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = np.random.randn(10)
    df['c'] = np.random.randn(10)
    df.loc[2, 'a'] = np.nan
    df.loc[5, 'b'] = np.nan
    df = null_encode(df)
    assert 'a' in df
    assert 'b' in df
    assert 'c' in df
    assert 'a_isnull' in df
    assert 'b_isnull' in df
    assert 'c_isnull' not in df
    assert df.shape[0] == 10
    assert df.shape[1] == 5

    # Should do only cols specified
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = np.random.randn(10)
    df['c'] = np.random.randn(10)
    df.loc[2, 'a'] = np.nan
    df.loc[5, 'b'] = np.nan
    df.loc[6, 'c'] = np.nan
    df = null_encode(df, cols=['a', 'b'])
    assert 'a' in df
    assert 'b' in df
    assert 'c' in df
    assert 'a_isnull' in df
    assert 'b_isnull' in df
    assert 'c_isnull' not in df
    assert df.shape[0] == 10
    assert df.shape[1] == 5

    # Should work the same way if passed a string
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = np.random.randn(10)
    df['c'] = np.random.randn(10)
    df.loc[2, 'a'] = np.nan
    df.loc[5, 'b'] = np.nan
    df.loc[6, 'c'] = np.nan
    df = null_encode(df, cols='a')
    assert 'a' in df
    assert 'b' in df
    assert 'c' in df
    assert 'a_isnull' in df
    assert 'b_isnull' not in df
    assert 'c_isnull' not in df
    assert df.shape[0] == 10
    assert df.shape[1] == 4


def test_label_encode():
    """Tests encoding.label_encode"""

    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = np.random.randn(10)

    # Not specifying cols w/ not categorical cols should not change the df
    df = label_encode(df)
    assert 'a' in df
    assert 'b' in df
    assert df.shape[0] == 10
    assert df.shape[1] == 2

    # Should auto-detect categorical cols and label encode
    df['c'] = ['a', 'a', 'a', 'b', 'b', 'c', 'a', 'a', 'c', 'b']
    df = label_encode(df, cols=['c'])
    assert 'a' in df
    assert 'b' in df
    assert 'c' in df
    assert df.shape[0] == 10
    assert df.shape[1] == 3
    assert df['c'].nunique() == 3
    assert df.iloc[0, 2] == 0
    assert df.iloc[1, 2] == 0
    assert df.iloc[2, 2] == 0
    assert df.iloc[3, 2] == 1
    assert df.iloc[4, 2] == 1
    assert df.iloc[5, 2] == 2
    assert df.iloc[6, 2] == 0
    assert df.iloc[7, 2] == 0
    assert df.iloc[8, 2] == 2
    assert df.iloc[9, 2] == 1

    # Passing a string instead of a list for cols should work the same way
    df['c'] = ['a', 'a', 'a', 'b', 'b', 'b', 'a', 'a', 'a', 'b']
    df = label_encode(df, cols='c')
    assert 'a' in df
    assert 'b' in df
    assert 'c' in df
    assert df.shape[0] == 10
    assert df.shape[1] == 3
    assert df['c'].nunique() == 2

    # Should propagate nans
    df['c'] = ['a', 'a', np.nan, 'b', 'b', 'b', 'a', 'a', 'a', 'b']
    df = label_encode(df, cols='c')
    assert 'a' in df
    assert 'b' in df
    assert 'c' in df
    assert df.shape[0] == 10
    assert df.shape[1] == 3
    assert df['c'].nunique() == 2
    assert np.isnan(df['c'][2])
    assert not np.isnan(df['c'][1])



def test_one_hot_encode():
    """Tests encoding.one_hot_encode"""

    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = np.random.randn(10)

    # Not specifying cols w/ not categorical cols should not change the df
    df = one_hot_encode(df)
    assert 'a' in df
    assert 'b' in df
    assert df.shape[0] == 10
    assert df.shape[1] == 2

    # One binary column w/ reduce_df=True should only add 1 col 
    df['c'] = ['a', 'a', 'a', 'b', 'b', 'b', 'a', 'a', 'a', 'b']
    df = one_hot_encode(df, cols=['c'], reduce_df=True)
    assert 'a' in df
    assert 'b' in df
    assert 'c_a' in df
    assert df.shape[0] == 10
    assert df.shape[1] == 3

    # Passing a string instead of a list for cols should work the same way
    df['c'] = ['a', 'a', 'a', 'b', 'b', 'b', 'a', 'a', 'a', 'b']
    df = one_hot_encode(df, cols='c', reduce_df=True)
    assert 'a' in df
    assert 'b' in df
    assert 'c_a' in df
    assert df.shape[0] == 10
    assert df.shape[1] == 3

    # Non-binary cols should add appropriate number of columns
    del df
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = np.random.randn(10)
    df['c'] = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
    df = one_hot_encode(df, cols=['c'])
    assert 'a' in df
    assert 'b' in df
    assert 'c_0' in df
    assert 'c_1' in df
    assert 'c_2' in df
    assert 'c_3' in df
    assert df.shape[0] == 10
    assert df.shape[1] == 6

    # reduce_df should reduce the unnecessary degrees of freedom
    del df
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = np.random.randn(10)
    df['c'] = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
    df = one_hot_encode(df, cols=['c'], reduce_df=True)
    assert 'a' in df
    assert 'b' in df
    assert 'c_0' in df
    assert 'c_1' in df
    assert 'c_2' in df
    assert 'c_3' not in df
    assert df.shape[0] == 10
    assert df.shape[1] == 5

    # cols as a list of columns should encode each col in the list
    del df
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = np.random.randn(10)
    df['c'] = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
    df['d'] = ['a', 'a', 'a', 'b', 'b', 'b', 'a', 'a', 'a', 'b']
    df = one_hot_encode(df, cols=['c', 'd'])
    assert 'a' in df
    assert 'b' in df
    assert 'c_0' in df
    assert 'c_1' in df
    assert 'c_2' in df
    assert 'c_3' in df
    assert 'd_a' in df
    assert 'd_b' in df
    assert df.shape[0] == 10
    assert df.shape[1] == 8

    # When cols not specified, should auto-detect categorical columns
    del df
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = np.random.randn(10)
    df['c'] = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c']
    df['d'] = ['a', 'a', 'a', 'b', 'b', 'b', 'a', 'a', 'a', 'b']
    df = one_hot_encode(df)
    assert 'a' in df
    assert 'b' in df
    assert 'c_a' in df
    assert 'c_b' in df
    assert 'c_c' in df
    assert 'd_a' in df
    assert 'd_b' in df
    assert df.shape[0] == 10
    assert df.shape[1] == 7


def test_TargetEncoder():
    """Tests encoding.TargetEncoder"""

    # Data
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'd', np.nan]
    df['y'] = [0, 2, 4, 6, 8, 10, 19, 20, 21, 1000]

    # Encode
    te = TargetEncoder(cols='b')
    dfo = te.fit_transform(df[['a', 'b']], df['y'])

    # Check outputs
    assert 'a' in dfo
    assert 'b' in dfo
    assert 'y' not in dfo
    assert dfo.shape[0] == 10
    assert dfo.shape[1] == 2
    assert dfo.loc[0, 'b'] == 1
    assert dfo.loc[1, 'b'] == 1
    assert dfo.loc[2, 'b'] == 5
    assert dfo.loc[3, 'b'] == 5
    assert dfo.loc[4, 'b'] == 9
    assert dfo.loc[5, 'b'] == 9
    assert dfo.loc[6, 'b'] == 20
    assert dfo.loc[7, 'b'] == 20
    assert dfo.loc[8, 'b'] == 20
    assert np.isnan(dfo.loc[9, 'b'])

    # Two columns
    df['b'] = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'd', np.nan]
    df['c'] = ['a', 'a', 'a', 'b', 'b', 'b', np.nan, np.nan, np.nan, 'c']
    te = TargetEncoder(cols=['b', 'c'])
    dfo = te.fit_transform(df[['a', 'b', 'c']], df['y'])
    assert 'a' in dfo
    assert 'b' in dfo
    assert 'c' in dfo
    assert 'y' not in dfo
    assert dfo.shape[0] == 10
    assert dfo.shape[1] == 3
    assert dfo.loc[0, 'b'] == 1
    assert dfo.loc[1, 'b'] == 1
    assert dfo.loc[2, 'b'] == 5
    assert dfo.loc[3, 'b'] == 5
    assert dfo.loc[4, 'b'] == 9
    assert dfo.loc[5, 'b'] == 9
    assert dfo.loc[6, 'b'] == 20
    assert dfo.loc[7, 'b'] == 20
    assert dfo.loc[8, 'b'] == 20
    assert np.isnan(dfo.loc[9, 'b'])
    assert dfo.loc[0, 'c'] == 2
    assert dfo.loc[1, 'c'] == 2
    assert dfo.loc[2, 'c'] == 2
    assert dfo.loc[3, 'c'] == 8
    assert dfo.loc[4, 'c'] == 8
    assert dfo.loc[5, 'c'] == 8
    assert np.isnan(dfo.loc[6, 'c'])
    assert np.isnan(dfo.loc[7, 'c'])
    assert np.isnan(dfo.loc[8, 'c'])
    assert dfo.loc[9, 'c'] == 1000

    # Should auto-detect categorical columns if cols is not specified
    df['b'] = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'd', np.nan]
    df['c'] = ['a', 'a', 'a', 'b', 'b', 'b', np.nan, np.nan, np.nan, 'c']
    te = TargetEncoder()
    dfo = te.fit_transform(df[['a', 'b', 'c']], df['y'])
    assert 'a' in dfo
    assert 'b' in dfo
    assert 'c' in dfo
    assert 'y' not in dfo
    assert dfo.shape[0] == 10
    assert dfo.shape[1] == 3
    assert dfo.loc[0, 'b'] == 1
    assert dfo.loc[1, 'b'] == 1
    assert dfo.loc[2, 'b'] == 5
    assert dfo.loc[3, 'b'] == 5
    assert dfo.loc[4, 'b'] == 9
    assert dfo.loc[5, 'b'] == 9
    assert dfo.loc[6, 'b'] == 20
    assert dfo.loc[7, 'b'] == 20
    assert dfo.loc[8, 'b'] == 20
    assert np.isnan(dfo.loc[9, 'b'])
    assert dfo.loc[0, 'c'] == 2
    assert dfo.loc[1, 'c'] == 2
    assert dfo.loc[2, 'c'] == 2
    assert dfo.loc[3, 'c'] == 8
    assert dfo.loc[4, 'c'] == 8
    assert dfo.loc[5, 'c'] == 8
    assert np.isnan(dfo.loc[6, 'c'])
    assert np.isnan(dfo.loc[7, 'c'])
    assert np.isnan(dfo.loc[8, 'c'])
    assert dfo.loc[9, 'c'] == 1000


def test_TargetEncoderCV():
    """Tests encoding.TargetEncoderCV"""

    # Data
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = ['a', 'a', 'b', 'b', np.nan, 'a', 'a', 'b', 'b', 'c']
    df['y'] = [0, 2, 8, 10, -1000, 4, 6, 12, 14, 1000]

    # Encode
    te = TargetEncoderCV(cols='b', n_splits=2, shuffle=False)
    dfo = te.fit_transform(df[['a', 'b']], df['y'])

    # Check outputs
    assert 'a' in dfo
    assert 'b' in dfo
    assert 'y' not in dfo
    assert dfo.shape[0] == 10
    assert dfo.shape[1] == 2
    assert dfo.loc[0, 'b'] == 5
    assert dfo.loc[1, 'b'] == 5
    assert dfo.loc[2, 'b'] == 13
    assert dfo.loc[3, 'b'] == 13
    assert np.isnan(dfo.loc[4, 'b']) #nans should propagate
    assert dfo.loc[5, 'b'] == 1
    assert dfo.loc[6, 'b'] == 1
    assert dfo.loc[7, 'b'] == 9
    assert dfo.loc[8, 'b'] == 9
    assert np.isnan(dfo.loc[9, 'b']) #cat only in test fold should be nan!

    # Ensure setting n_splits works
    df = pd.DataFrame()
    df['a'] = np.random.randn(9)
    df['b'] = ['a', 'a', 'b', 'b', 'b', 'a', 'a', 'a', 'b']
    df['y'] = [0, 1, 11,
               12, 13, 2,
               3, 4, 14]
    te = TargetEncoderCV(cols='b', n_splits=3, shuffle=False)
    dfo = te.fit_transform(df[['a', 'b']], df['y'])
    assert 'a' in dfo
    assert 'b' in dfo
    assert 'y' not in dfo
    assert dfo.shape[0] == 9
    assert dfo.shape[1] == 2
    assert dfo.loc[0, 'b'] == 3
    assert dfo.loc[1, 'b'] == 3
    assert dfo.loc[2, 'b'] == 13
    assert dfo.loc[3, 'b'] == 12.5
    assert dfo.loc[4, 'b'] == 12.5
    assert dfo.loc[5, 'b'] == 2
    assert dfo.loc[6, 'b'] == 1
    assert dfo.loc[7, 'b'] == 1
    assert dfo.loc[8, 'b'] == 12

    # Check shuffle works
    df = pd.DataFrame()
    df['a'] = np.random.randn(100)
    df['b'] = 50*['aa', 'bb']
    df['y'] = np.zeros(100)
    df.loc[df['b']=='aa', 'y'] = np.random.randn(50)
    df.loc[df['b']=='bb', 'y'] = 1000+np.random.randn(50)
    te = TargetEncoderCV(cols='b', n_splits=3, shuffle=True)
    dfo = te.fit_transform(df[['a', 'b']], df['y'])
    assert dfo['b'].nunique() == 6
    aa_vals = dfo.loc[df['b']=='aa', 'b']
    bb_vals = dfo.loc[df['b']=='bb', 'b']
    assert aa_vals.nunique() == 3
    assert bb_vals.nunique() == 3
    assert not (aa_vals.unique() == aa_vals.mean()).any()
    assert not (bb_vals.unique() == bb_vals.mean()).any()
    assert (abs(aa_vals.unique())<1.0).all()
    assert (abs(bb_vals.unique()-1000)<1.0).all()

    # Check multiple cols works
    df = pd.DataFrame()
    df['a'] = np.random.randn(8)
    df['b'] = ['a', 'a', 'b', 'b', 
               'a', 'a', 'b', 'b']
    df['c'] = ['a', 'b', 'a', 'b',
               'a', 'b', 'a', 'b']
    df['y'] = [0, 1, 10, 11, 
               2, 3, 12, 13]
    te = TargetEncoderCV(cols=['b', 'c'], n_splits=2, shuffle=False)
    dfo = te.fit_transform(df[['a', 'b', 'c']], df['y'])
    assert 'a' in dfo
    assert 'b' in dfo
    assert 'c' in dfo
    assert 'y' not in dfo
    assert dfo.shape[0] == 8
    assert dfo.shape[1] == 3
    # b column
    assert dfo.loc[0, 'b'] == 2.5
    assert dfo.loc[1, 'b'] == 2.5
    assert dfo.loc[2, 'b'] == 12.5
    assert dfo.loc[3, 'b'] == 12.5
    assert dfo.loc[4, 'b'] == 0.5
    assert dfo.loc[5, 'b'] == 0.5
    assert dfo.loc[6, 'b'] == 10.5
    assert dfo.loc[7, 'b'] == 10.5
    # c column
    assert dfo.loc[0, 'c'] == 7
    assert dfo.loc[1, 'c'] == 8
    assert dfo.loc[2, 'c'] == 7
    assert dfo.loc[3, 'c'] == 8
    assert dfo.loc[4, 'c'] == 5
    assert dfo.loc[5, 'c'] == 6
    assert dfo.loc[6, 'c'] == 5
    assert dfo.loc[7, 'c'] == 6

    # Should auto-detect categorical cols if cols was not specified
    df = pd.DataFrame()
    df['a'] = np.random.randn(8)
    df['b'] = ['a', 'a', 'b', 'b', 
               'a', 'a', 'b', 'b']
    df['c'] = ['a', 'b', 'a', 'b',
               'a', 'b', 'a', 'b']
    df['y'] = [0, 1, 10, 11, 
               2, 3, 12, 13]
    te = TargetEncoderCV(n_splits=2, shuffle=False)
    dfo = te.fit_transform(df[['a', 'b', 'c']], df['y'])
    assert 'a' in dfo
    assert 'b' in dfo
    assert 'c' in dfo
    assert 'y' not in dfo
    assert dfo.shape[0] == 8
    assert dfo.shape[1] == 3
    # b column
    assert dfo.loc[0, 'b'] == 2.5
    assert dfo.loc[1, 'b'] == 2.5
    assert dfo.loc[2, 'b'] == 12.5
    assert dfo.loc[3, 'b'] == 12.5
    assert dfo.loc[4, 'b'] == 0.5
    assert dfo.loc[5, 'b'] == 0.5
    assert dfo.loc[6, 'b'] == 10.5
    assert dfo.loc[7, 'b'] == 10.5
    # c column
    assert dfo.loc[0, 'c'] == 7
    assert dfo.loc[1, 'c'] == 8
    assert dfo.loc[2, 'c'] == 7
    assert dfo.loc[3, 'c'] == 8
    assert dfo.loc[4, 'c'] == 5
    assert dfo.loc[5, 'c'] == 6
    assert dfo.loc[6, 'c'] == 5
    assert dfo.loc[7, 'c'] == 6


def test_target_encode():
    """Tests encoding.target_encode"""

    # Data
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'd', np.nan]
    df['y'] = [0, 2, 4, 6, 8, 10, 19, 20, 21, 1000]

    # Encode
    dfo = target_encode(df[['a', 'b']], df['y'], cols='b')

    # Check outputs
    assert 'a' in dfo
    assert 'b' in dfo
    assert 'y' not in dfo
    assert dfo.shape[0] == 10
    assert dfo.shape[1] == 2
    assert dfo.loc[0, 'b'] == 1
    assert dfo.loc[1, 'b'] == 1
    assert dfo.loc[2, 'b'] == 5
    assert dfo.loc[3, 'b'] == 5
    assert dfo.loc[4, 'b'] == 9
    assert dfo.loc[5, 'b'] == 9
    assert dfo.loc[6, 'b'] == 20
    assert dfo.loc[7, 'b'] == 20
    assert dfo.loc[8, 'b'] == 20
    assert np.isnan(dfo.loc[9, 'b'])


def test_target_encode_cv():
    """Tests encoding.target_encode_cv"""

    # Data
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'd', np.nan]
    df['y'] = [0, 2, 4, 6, 8, 10, 19, 20, 21, 1000]

    # Encode
    dfo = target_encode_cv(df[['a', 'b']], df['y'], cols='b')
    
    # Check outputs
    assert 'a' in dfo
    assert 'b' in dfo
    assert 'y' not in dfo
    assert dfo.shape[0] == 10
    assert dfo.shape[1] == 2


def test_target_encode_loo():
    """Tests encoding.target_encode_loo"""

    # Data
    df = pd.DataFrame()
    df['a'] = np.random.randn(10)
    df['b'] = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'd', np.nan]
    df['y'] = [0, 2, 4, 6, 8, 10, 19, 20, 21, 1000]

    # Encode
    dfo = target_encode_loo(df[['a', 'b']], df['y'], cols='b')
    
    # Check outputs
    assert 'a' in dfo
    assert 'b' in dfo
    assert 'y' not in dfo
    assert dfo.shape[0] == 10
    assert dfo.shape[1] == 2


def test_text_multi_label_binarize():
    """Tests encoding.text_multi_label_binarize"""

    # Should multi-label encode both specified columns, and not non-specified
    df = pd.DataFrame()
    df['a'] = ['aa,bb', 'aa', 'bb', 'bb,cc']
    df['b'] = ['asdf,bb_r', 'haha', 'haha,asdf', 'asdf,asdf']
    df['c'] = np.random.randn(4)
    df = text_multi_label_binarize(df, cols=['a', 'b'])
    assert 'a' not in df
    assert 'b' not in df
    assert 'c' in df
    assert 'a_aa' in df
    assert 'a_bb' in df
    assert 'a_cc' in df
    assert 'b_asdf' in df
    assert 'b_bb_r' in df
    assert 'b_haha' in df
    assert df.shape[0] == 4
    assert df.shape[1] == 7
    assert df.loc[0, 'a_aa'] == 1
    assert df.loc[1, 'a_aa'] == 1
    assert df.loc[2, 'a_aa'] == 0
    assert df.loc[3, 'a_aa'] == 0

    assert df.loc[0, 'a_bb'] == 1
    assert df.loc[1, 'a_bb'] == 0
    assert df.loc[2, 'a_bb'] == 1
    assert df.loc[3, 'a_bb'] == 1

    assert df.loc[0, 'a_cc'] == 0
    assert df.loc[1, 'a_cc'] == 0
    assert df.loc[2, 'a_cc'] == 0
    assert df.loc[3, 'a_cc'] == 1

    assert df.loc[0, 'b_asdf'] == 1
    assert df.loc[1, 'b_asdf'] == 0
    assert df.loc[2, 'b_asdf'] == 1
    assert df.loc[3, 'b_asdf'] == 1

    assert df.loc[0, 'b_bb_r'] == 1
    assert df.loc[1, 'b_bb_r'] == 0
    assert df.loc[2, 'b_bb_r'] == 0
    assert df.loc[3, 'b_bb_r'] == 0

    assert df.loc[0, 'b_haha'] == 0
    assert df.loc[1, 'b_haha'] == 1
    assert df.loc[2, 'b_haha'] == 1
    assert df.loc[3, 'b_haha'] == 0

    # Should only encode labels which were passed
    df = pd.DataFrame()
    df['b'] = ['asdf,bb_r', 'haha', 'haha,asdf', 'asdf,asdf']
    df['c'] = np.random.randn(4)
    labels = {'b': ['bb_r', 'haha', 'lala']}
    df = text_multi_label_binarize(df, cols='b', labels=labels)
    assert 'b' not in df
    assert 'c' in df
    assert 'b_asdf' not in df
    assert 'b_bb_r' in df
    assert 'b_haha' in df
    assert 'b_lala' in df
    assert df.shape[0] == 4
    assert df.shape[1] == 4

    assert df.loc[0, 'b_bb_r'] == 1
    assert df.loc[1, 'b_bb_r'] == 0
    assert df.loc[2, 'b_bb_r'] == 0
    assert df.loc[3, 'b_bb_r'] == 0

    assert df.loc[0, 'b_haha'] == 0
    assert df.loc[1, 'b_haha'] == 1
    assert df.loc[2, 'b_haha'] == 1
    assert df.loc[3, 'b_haha'] == 0

    assert df.loc[0, 'b_lala'] == 0
    assert df.loc[1, 'b_lala'] == 0
    assert df.loc[2, 'b_lala'] == 0
    assert df.loc[3, 'b_lala'] == 0



def test_NhotEncoder():
    """Tests encoding.NhotEncoder"""

    # Data
    df = pd.DataFrame()
    df['a'] = np.random.randn(6)
    df['b'] = ['aa,bb', 'aa,cc', 'aa,aa', 'cc', 'aa,bb,cc', np.nan]
    df['y'] = [0, 2, 4, 6, 8, 10]

    # Encode
    ne = NhotEncoder(cols='b')
    dfo = ne.fit_transform(df)
    assert dfo.shape[0] == 6
    assert dfo.shape[1] == 5
    assert 'b_aa' in dfo
    assert 'b_bb' in dfo
    assert 'b_cc' in dfo
    assert 'b' not in dfo
    assert dfo.loc[0, 'b_aa'] == 1
    assert dfo.loc[1, 'b_aa'] == 1
    assert dfo.loc[2, 'b_aa'] == 1
    assert dfo.loc[3, 'b_aa'] == 0
    assert dfo.loc[4, 'b_aa'] == 1
    assert np.isnan(dfo.loc[5, 'b_aa'])
    assert dfo.loc[0, 'b_bb'] == 1
    assert dfo.loc[1, 'b_bb'] == 0
    assert dfo.loc[2, 'b_bb'] == 0
    assert dfo.loc[3, 'b_bb'] == 0
    assert dfo.loc[4, 'b_bb'] == 1
    assert np.isnan(dfo.loc[5, 'b_bb'])
    assert dfo.loc[0, 'b_cc'] == 0
    assert dfo.loc[1, 'b_cc'] == 1
    assert dfo.loc[2, 'b_cc'] == 0
    assert dfo.loc[3, 'b_cc'] == 1
    assert dfo.loc[4, 'b_cc'] == 1
    assert np.isnan(dfo.loc[5, 'b_cc'])

    # Check top_n works
    ne = NhotEncoder(cols='b', top_n=2)
    dfo = ne.fit_transform(df)
    assert dfo.shape[0] == 6
    assert dfo.shape[1] == 4
    assert 'b_aa' in dfo
    assert 'b_bb' not in dfo
    assert 'b_cc' in dfo
    assert 'b' not in dfo
    assert dfo.loc[0, 'b_aa'] == 1
    assert dfo.loc[1, 'b_aa'] == 1
    assert dfo.loc[2, 'b_aa'] == 1
    assert dfo.loc[3, 'b_aa'] == 0
    assert dfo.loc[4, 'b_aa'] == 1
    assert np.isnan(dfo.loc[5, 'b_aa'])
    assert dfo.loc[0, 'b_cc'] == 0
    assert dfo.loc[1, 'b_cc'] == 1
    assert dfo.loc[2, 'b_cc'] == 0
    assert dfo.loc[3, 'b_cc'] == 1
    assert dfo.loc[4, 'b_cc'] == 1
    assert np.isnan(dfo.loc[5, 'b_cc'])

    # Check top_prc works
    ne = NhotEncoder(cols='b', top_prc=0.67)
    dfo = ne.fit_transform(df)
    assert dfo.shape[0] == 6
    assert dfo.shape[1] == 4
    assert 'b_aa' in dfo
    assert 'b_bb' not in dfo
    assert 'b_cc' in dfo
    assert 'b' not in dfo
    assert dfo.loc[0, 'b_aa'] == 1
    assert dfo.loc[1, 'b_aa'] == 1
    assert dfo.loc[2, 'b_aa'] == 1
    assert dfo.loc[3, 'b_aa'] == 0
    assert dfo.loc[4, 'b_aa'] == 1
    assert np.isnan(dfo.loc[5, 'b_aa'])
    assert dfo.loc[0, 'b_cc'] == 0
    assert dfo.loc[1, 'b_cc'] == 1
    assert dfo.loc[2, 'b_cc'] == 0
    assert dfo.loc[3, 'b_cc'] == 1
    assert dfo.loc[4, 'b_cc'] == 1
    assert np.isnan(dfo.loc[5, 'b_cc'])



def test_JsonEncoder():
    """Tests encoding.JsonEncoder"""

    # Data
    df = pd.DataFrame()
    df['a'] = np.random.randn(6)
    df['b'] = ['[{\'id\': 12, \'Genre\': \'Comedy\'}, {\'id\': 13, \'Genre\': \'Romance\'}, {\'id\': 14, \'Genre\': \'Horror\'}]', 
               '[{\'id\': 22, \'Genre\': \'Drama\'}, {\'id\': 23, \'Genre\': \'Cyberpunk\'}]', 
               '[{\'id\': 32}, {\'id\': 33, \'Genre\': \'Scifi\'}]', 
               '', 
               'nan', 
               np.nan]
    df['c'] = ['[{\'id\': 12, \'Genre\': \'Comedy\'}, {\'id\': 13, \'Genre\': \'Romance\'}, {\'id\': 14, \'Genre\': \'Horror\'}]', 
               '[{\'id\': 22, \'Genre\': \'Drama\'}, {\'id\': 23, \'Genre\': \'Cyberpunk\'}]', 
               '[{\'id\': 32}, {\'id\': 33, \'Genre\': \'Scifi\'}]', 
               '', 
               'nan', 
               np.nan]
    df['d'] = ['[{\'Name\': \'Andy\', \'Role\': \'Director\'}]', 
               '[{\'Name\': \'Sue\', \'Role\': \'Producer\'}, {\'Name\': \'Betty\', \'Role\': \'Director\'}]', 
               '[{\'Name\': \'Carol\', \'Role\': \'Director\'}, {\'Name\': \'Rachel\', \'Role\': \'Producer\'}]', 
               '', 
               'nan', 
               np.nan]
    df['y'] = [0, 2, 4, 6, 8, 10]

    # Fields to encode
    fields = {
        'b': 'Genre',
        'c': ['Genre', 'id', 'not_there'],
        'd': [('Name', 'Role', 'Director'),
              ('Name', 'Role', 'Producer')],
    }

    # Encode
    je = JsonEncoder(fields=fields)
    dfo = je.fit_transform(df)
    assert dfo.shape[0] == 6
    assert dfo.shape[1] == 8
    assert 'a' in dfo
    assert 'y' in dfo
    assert 'b_Genre' in dfo
    assert 'b' not in dfo
    assert 'c_Genre' in dfo
    assert 'c_id' in dfo
    assert 'c_not_there' in dfo
    assert 'c' not in dfo
    assert 'd_Role_Director_Name' in dfo
    assert 'd_Role_Producer_Name' in dfo
    assert 'd' not in dfo

    assert dfo.loc[0, 'y'] == 0
    assert dfo.loc[1, 'y'] == 2
    assert dfo.loc[2, 'y'] == 4
    assert dfo.loc[3, 'y'] == 6
    assert dfo.loc[4, 'y'] == 8
    assert dfo.loc[5, 'y'] == 10

    assert dfo.loc[0, 'b_Genre'] == 'Comedy,Romance,Horror'
    assert dfo.loc[1, 'b_Genre'] == 'Drama,Cyberpunk'
    assert dfo.loc[2, 'b_Genre'] == 'Scifi'
    assert np.isnan(dfo.loc[3, 'b_Genre'])
    assert np.isnan(dfo.loc[4, 'b_Genre'])
    assert np.isnan(dfo.loc[5, 'b_Genre'])

    assert dfo.loc[0, 'c_Genre'] == 'Comedy,Romance,Horror'
    assert dfo.loc[1, 'c_Genre'] == 'Drama,Cyberpunk'
    assert dfo.loc[2, 'c_Genre'] == 'Scifi'
    assert np.isnan(dfo.loc[3, 'c_Genre'])
    assert np.isnan(dfo.loc[4, 'c_Genre'])
    assert np.isnan(dfo.loc[5, 'c_Genre'])

    assert dfo.loc[0, 'c_id'] == '12,13,14'
    assert dfo.loc[1, 'c_id'] == '22,23'
    assert dfo.loc[2, 'c_id'] == '32,33'
    assert np.isnan(dfo.loc[3, 'c_id'])
    assert np.isnan(dfo.loc[4, 'c_id'])
    assert np.isnan(dfo.loc[5, 'c_id'])

    assert np.isnan(dfo.loc[0, 'c_not_there'])
    assert np.isnan(dfo.loc[1, 'c_not_there'])
    assert np.isnan(dfo.loc[2, 'c_not_there'])
    assert np.isnan(dfo.loc[3, 'c_not_there'])
    assert np.isnan(dfo.loc[4, 'c_not_there'])
    assert np.isnan(dfo.loc[5, 'c_not_there'])

    assert dfo.loc[0, 'd_Role_Director_Name'] == 'Andy'
    assert dfo.loc[1, 'd_Role_Director_Name'] == 'Betty'
    assert dfo.loc[2, 'd_Role_Director_Name'] == 'Carol'
    assert np.isnan(dfo.loc[3, 'd_Role_Director_Name'])
    assert np.isnan(dfo.loc[4, 'd_Role_Director_Name'])
    assert np.isnan(dfo.loc[5, 'd_Role_Director_Name'])

    assert np.isnan(dfo.loc[0, 'd_Role_Producer_Name'])
    assert dfo.loc[1, 'd_Role_Producer_Name'] == 'Sue'
    assert dfo.loc[2, 'd_Role_Producer_Name'] == 'Rachel'
    assert np.isnan(dfo.loc[3, 'd_Role_Producer_Name'])
    assert np.isnan(dfo.loc[4, 'd_Role_Producer_Name'])
    assert np.isnan(dfo.loc[5, 'd_Role_Producer_Name'])

