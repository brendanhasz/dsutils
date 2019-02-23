"""Tests cleaning module

"""


import numpy as np
import pandas as pd

from dsutils.cleaning import remove_duplicate_cols
from dsutils.cleaning import remove_noninformative_cols
from dsutils.cleaning import categorical_to_int



def test_remove_duplicate_cols():
    """Tests cleaning.remove_duplicate_cols"""

    # Should remove duplicate cols
    df = pd.DataFrame()
    df['A'] = np.array([0, 1, 2, 3, 4, 5])
    df['B'] = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    df['C'] = np.array(['a', 'a', 'a', 'b', 'b', 'b']) #dup
    df['D'] = np.array([1, 1, 2, 2, 3, 3])
    df['E'] = np.array([0.01, 1.01, 2.01, 3.01, 4.01, 5.01])
    df['F'] = np.array([0, 1, 2, 3, 4, 5]) #dup
    df['G'] = np.array([0.01, 1.01, 2.01, 3.01, 4.01, 5.01]) #dup
    df['H'] = np.array([11, 12, 13, 14, 15, 16])

    remove_noninformative_cols(df)
    assert 'A' in df
    assert 'B' in df
    assert 'C' not in df
    assert 'D' in df
    assert 'E' in df
    assert 'F' not in df
    assert 'G' not in df
    assert 'H' in df



def test_remove_noninformative_cols():
    """Tests cleaning.remove_noninformative_cols"""

    # Should remove entirely empty columns
    df = pd.DataFrame()
    df['A'] = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    df['B'] = np.nan
    df['C'] = np.array([0, 1, 2, 3, 4, 5])
    df['D'] = np.nan
    remove_noninformative_cols(df)
    assert 'A' in df
    assert 'B' not in df
    assert 'C' in df
    assert 'D' not in df

    # Should remove cols w/ only 1 unique value
    df = pd.DataFrame()
    df['A'] = np.array(['a', 'a', 'a', 'a', 'a', 'a'])
    df['B'] = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    df['C'] = np.array([1, 1, 1, 1, 1, 1])
    df['D'] = np.array([0, 1, 2, 3, 4, 5])
    remove_noninformative_cols(df)
    assert 'A' not in df
    assert 'B' in df
    assert 'C' not in df
    assert 'D' in df

    # Should remove duplicate cols
    df = pd.DataFrame()
    df['A'] = np.array([0, 1, 2, 3, 4, 5])
    df['B'] = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    df['C'] = np.array([1, 1, 2, 2, 3, 3])
    df['D'] = np.array([0, 1, 2, 3, 4, 5])
    remove_noninformative_cols(df)
    assert 'A' in df
    assert 'B' in df
    assert 'C' in df
    assert 'D' not in df



def test_categorical_to_int():
    """Tests cleaning.categorical_to_int"""

    # Should only encode cols in cols arg
    df = pd.DataFrame()
    df['A'] = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    df['B'] = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
    df['C'] = np.array([1, 1, 0, 1, 1, 1])
    categorical_to_int(df, cols=['A'])
    assert 'A' in df
    assert 'B' in df
    assert 'C' in df
    assert df.shape[0] == 6
    assert df.shape[1] == 3
    assert str(df['A'].dtype) == 'uint8'
    assert str(df['B'].dtype) == 'object'

    # Should work the same way if passed a string
    df['A'] = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    categorical_to_int(df, cols='A')
    assert 'A' in df
    assert 'B' in df
    assert 'C' in df
    assert df.shape[0] == 6
    assert df.shape[1] == 3
    assert str(df['A'].dtype) == 'uint8'
    assert str(df['B'].dtype) == 'object'

    # Should encode all categorical columns if not specified
    df['A'] = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    categorical_to_int(df)
    assert 'A' in df
    assert 'B' in df
    assert 'C' in df
    assert df.shape[0] == 6
    assert df.shape[1] == 3
    assert str(df['A'].dtype) == 'uint8'
    assert str(df['B'].dtype) == 'uint8'

    # Should encode as float if there are NaN
    df['A'] = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    df.loc[3, 'A'] = np.nan
    categorical_to_int(df)
    assert 'A' in df
    assert 'B' in df
    assert 'C' in df
    assert df.shape[0] == 6
    assert df.shape[1] == 3
    assert str(df['A'].dtype) == 'float32'

    # Should use uint16 if more unique vals than will fit in uint8
    df = pd.DataFrame()
    df['A'] = [str(n) for n in range(300)]
    categorical_to_int(df, cols=['A'])
    assert 'A' in df
    assert df.shape[0] == 300
    assert df.shape[1] == 1
    assert str(df['A'].dtype) == 'uint16'

    # Should use uint32 if more unique vals than will fit in uint16
    df = pd.DataFrame()
    df['A'] = [str(n) for n in range(70000)]
    categorical_to_int(df, cols=['A'])
    assert 'A' in df
    assert df.shape[0] == 70000
    assert df.shape[1] == 1
    assert str(df['A'].dtype) == 'uint32'

    # and in theory should use uint64 if too many vals to fit in uint32
