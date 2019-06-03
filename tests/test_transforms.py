"""Tests transforms module

"""

import numpy as np
import pandas as pd

from dsutils.transforms import Scaler
from dsutils.transforms import Imputer

# TODO: quantile_transform

def test_transforms_scaler():
    """Tests dsutils.transforms.Scaler"""

    df = pd.DataFrame()
    df['a'] = [1, 2, 3, 4, 5]
    df['b'] = [1, 2, 3, 4, 5]
    df['c'] = [10, 20, 30, 40, 50]

    out = Scaler().fit_transform(df)

    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == 5
    assert out.shape[1] == 3
    assert out.iloc[2, 0] == 0.0
    assert out.iloc[0, 0] < out.iloc[1, 0]
    assert out.iloc[1, 0] < out.iloc[2, 0]
    assert out.iloc[2, 0] < out.iloc[3, 0]
    assert out.iloc[3, 0] < out.iloc[4, 0]
    assert out.iloc[2, 1] == 0.0
    assert out.iloc[2, 2] == 0.0
    assert out.iloc[0, 2] < out.iloc[1, 2]
    assert out.iloc[1, 2] < out.iloc[2, 2]
    assert out.iloc[2, 2] < out.iloc[3, 2]
    assert out.iloc[3, 2] < out.iloc[4, 2]


def test_transform_imputer():
    """Tests dsutils.transforms.Imputer"""

    df = pd.DataFrame()
    df['a'] = [np.nan, np.nan, 3, 4, 5]
    df['b'] = [1, np.nan, 3, np.nan, 5]
    df['c'] = [10, 20, 30, np.nan, np.nan]

    # Default should be to use median
    out = Imputer().fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == 5
    assert out.shape[1] == 3
    assert np.count_nonzero(out.isnull()) == 0
    assert out.iloc[0, 0] == 4
    assert out.iloc[1, 0] == 4
    assert out.iloc[2, 0] == 3
    assert out.iloc[0, 1] == 1
    assert out.iloc[1, 1] == 3
    assert out.iloc[2, 1] == 3
    assert out.iloc[3, 1] == 3
    assert out.iloc[0, 2] == 10
    assert out.iloc[3, 2] == 20
    assert out.iloc[4, 2] == 20

    df = pd.DataFrame()
    df['a'] = [2, np.nan, 3, 4, 5]
    df['b'] = [1, np.nan, 3, np.nan, 5]
    df['c'] = [10, 20, 30, np.nan, 40]

    # Using mean
    out = Imputer(method='mean').fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == 5
    assert out.shape[1] == 3
    assert np.count_nonzero(out.isnull()) == 0
    assert out.iloc[0, 0] == 2
    assert out.iloc[1, 0] == 3.5
    assert out.iloc[2, 0] == 3
    assert out.iloc[3, 0] == 4
    assert out.iloc[4, 0] == 5
    assert out.iloc[0, 1] == 1
    assert out.iloc[1, 1] == 3
    assert out.iloc[2, 1] == 3
    assert out.iloc[3, 1] == 3
    assert out.iloc[4, 1] == 5
    assert out.iloc[0, 2] == 10
    assert out.iloc[1, 2] == 20
    assert out.iloc[2, 2] == 30
    assert out.iloc[3, 2] == 25
    assert out.iloc[4, 2] == 40
