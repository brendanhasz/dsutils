"""Models

* :class:`.InterpolatingPredictor`
* :class:`.SvdRegressor`

"""



import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin



class InterpolatingPredictor(BaseEstimator):
    
    def __init__(self, loss_func=mean_squared_error):
        """Linear model which predicts the best interpolation between features

        References
        ----------
            Breiman, L. (1996). Stacked regressions. Machine learning, 24(1),
            49-64.
        """
        
        self._weights = None
        self.loss_func = loss_func


    def fit(self, X, y):
        """
        TODO
        """

        # Function to optimize
        def func(x):
            return self.loss_func(y, (x.reshape(1, -1)*X).sum(axis=1))

        # Number of dimensions of input
        Nd = X.shape[1]

        # Initial parameter guess
        x0 = np.ones(Nd)/float(Nd)
        
        # Constrain weights to sum to 1
        constraints = ({'type': 'eq',
                        'fun': lambda x: 1.0 - sum(x)})

        # Constain weights to be positive
        bounds = tuple((0.0, 1.0) for _ in range(Nd))

        # Solve for weights which optimize error
        res = minimize(func, x0, method='SLSQP',
                       bounds=bounds, constraints=constraints)
        self._weights = res.x
            
        # Return the fit object
        return self


    def predict(self, X):
        """
        TODO
        """

        # Check if model has been fit
        if self._weights is None:
            raise RuntimeError('model has not been fit')
        
        # Return the predictions
        return (self._weights.reshape(1, -1)*X).sum(axis=1)


    def fit_predict(self, X, y):
        """
        TOOD
        """
        return self.fit(X, y).predict(X)
