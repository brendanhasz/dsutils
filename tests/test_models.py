"""Tests models

"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from dsutils.models import InterpolatingPredictor


def test_InterpolatingPredictor():
    """Tests ensembling.EnsembleRegressor"""

    # Make dummy data
    N = 100
    D = 3
    X = pd.DataFrame(data=np.random.randn(N,D))
    y = pd.Series(index=X.index)
    w = np.random.rand(D)
    w = w/sum(w)
    y = (X*w.reshape(1, -1)).sum(axis=1)

    # Create the interpolating predictor
    ip = InterpolatingPredictor()

    # Weights should be none
    assert hasattr(ip, '_weights')
    assert ip._weights is None

    # Fit to data
    ip = ip.fit(X, y)

    # Check weights are consistent with w
    assert hasattr(ip, '_weights')
    assert ip._weights is not None
    assert all(abs(w[i]-ip._weights[i])<0.001 for i in range(D))

    # Predict y values
    y_pred = ip.predict(X)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert y_pred.shape[0] == y.shape[0]
    assert all(abs(y_pred[i]-y[i])<0.1 for i in range(len(y)))
