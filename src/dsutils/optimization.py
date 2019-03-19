"""Optimization

* :class:`.GaussianProcessOptimizer`
* :func:`.optimize_params_cv`

"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Kernel

from .plotting import plot_err



class GaussianProcessOptimizer():
    """Bayesian function optimizer which uses a Gaussian process to model the
    expensive function, and expected improvement as the acquisition function.

    """

    def __init__(self, lb, ub,
                 dtype=None,
                 param_names=None,
                 minimize=True,
                 kernel=RationalQuadratic()+WhiteKernel(noise_level=1e-4),
                 n_restarts_optimizer=5):
        """Gaussian process-based optimizer

        Parameters
        ----------
        lb : list
            Lower bound for each parameter.
        ub : list
            Upper bound for each parameter.
        dtype : list
            Datatype for each parameter.  int or float
            Default is to assume float for all parameters.
        param_names : list of str
            Parameter names
        minimize : bool
            Whether to minimize (True) or maximize (False).
        kernel : sklearn.gaussian_process.kernels.Kernel
            Kernel for the Gaussian process.
            Default = RationalQuadratic + WhiteKernel with a noise_level
            of 1e-4, which allows us to also model the noise.
        n_restarts_optimizer : int >= 0
            Number of times to re-seed the GP optimizer.
        """

        # If passed info for a single parameter, don't require list
        if isinstance(lb, (int, float)) and isinstance(ub, (int, float)):
            lb = [lb]
            ub = [ub]
        if dtype is None or dtype is int or dtype is float:
            dtype = [dtype]
        if param_names is None or param_names is str:
            param_names = [param_names]

        # Check types
        if not isinstance(lb, list) or not isinstance(ub, list):
            raise TypeError('lb and ub must be lists')
        if not isinstance(dtype, list):
            raise TypeError('dtype must be a list')
        if len(lb) != len(ub) or len(ub) != len(dtype):
            raise ValueError('lb, ub, and dtype must be same length')
        if not isinstance(minimize, bool):
            raise TypeError('minimize must be True or False')
        if not isinstance(kernel, Kernel):
            raise TypeError('kernel must be an sklearn GP kernel')
        if not isinstance(n_restarts_optimizer, int):
            raise TypeError('n_restarts_optimizer must be an int')
        if n_restarts_optimizer < 0:
            raise ValueError('n_restarts_optimizer must be non-negative')

        # Assume float if dtype not specified
        self.num_dims = len(lb)
        if dtype[0] is None:
            dtype = [float]*self.num_dims

        # Assign parameter names if not specified
        if param_names[0] is None:
            param_names = [None]*self.num_dims
            for iP in range(self.num_dims):
                param_names[iP] = 'Parameter_'+str(iP)

        # Store parameters
        self.lb = lb
        self.ub = ub
        self.db = [ub[i]-lb[i] for i in range(self.num_dims)]
        self.bounds = [(lb[i], ub[i]) for i in range(self.num_dims)]
        self.x = []
        self.y = []
        self.dtype = dtype
        self.param_names = param_names
        self.minimize = minimize
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=0.0,
            n_restarts_optimizer=n_restarts_optimizer,
        )

        # Keep track of highest (or lowest) point so far
        if minimize:
            self.opt_y = np.inf
            self.opt_x = None
        else:
            self.opt_y = -np.inf
            self.opt_x = None


    def _ensure_types(self, x):
        """Ensure types in x match dtype."""
        if isinstance(x, np.ndarray):
            x = x.tolist()
        for iP in range(self.num_dims):
            if self.dtype[iP] is int:
                x[iP] = round(x[iP])
        return x


    def _make_dict(self, x):
        """Make x a dict w/ keys=dimension names"""
        xo = dict()
        for iP in range(self.num_dims):
            xo[self.param_names[iP]] = x[iP]
        return xo


    def _fit_gp(self, step=None):
        """Fit the Gaussian process to data

        """

        # Set step if not specified
        if step is None:
            step = len(self.x)

        # Fit GP to 1st ``step`` steps
        x = self.x[:step]
        y = self.y[:step]
        
        # Convert to numpy arrays
        x = np.array(x).astype('float64')
        y = np.array(y)

        # Normalize y
        self._y_mean = y.mean()
        self._y_std = y.std()
        y = (y-self._y_mean)/self._y_std

        # Normalize x
        for iD in range(self.num_dims):
            x[:, iD] = (x[:, iD] - self.lb[iD]) / self.db[iD]

        # Add jitter to x
        x += 1e-5*np.random.standard_normal(x.shape)

        # Fit the Gaussian process
        self.gp = self.gp.fit(x, y)


    def _pred_gp(self, x, return_std=False):
        """Predict y with the Gaussian process.

        """

        # Convert to numpy array
        x = np.array(x).astype('float64')

        # Normalize x
        for iD in range(self.num_dims):
            x[:, iD] = (x[:, iD] - self.lb[iD]) / self.db[iD]

        # Add jitter to x
        x = x + 1e-5*np.random.standard_normal(x.shape)

        # Predict y
        y, y_std = self.gp.predict(x, return_std=True)

        # Convert back to true scale
        y = y*self._y_std+self._y_mean
        y_std = y_std*self._y_std

        # Return std dev if requested
        if return_std:
            return y, y_std
        else:
            return y


    def add_point(self, x, y):
        """Add a point to the history of sampled points.

        Parameters
        ----------
        x : list or int or float
            The point's coordinates.
        y : list or float
            Function value(s) at this point.
        """

        # Convert to list if passed single values
        if isinstance(x, (int, float)):
            x = [x]
        if isinstance(y, float):
            y = [y]

        # Check inputs
        if not all(isinstance(e, (int, float)) for e in x):
            raise TypeError('x must be a list of ints or floats')
        if not isinstance(y, (float, list)):
            raise TypeError('y must be a float or a list')
        if isinstance(y, list):
            if not all(isinstance(e, float) for e in y):
                raise TypeError('y must be a list of floats')
        if len(x) != self.num_dims:
            raise RuntimeError('x has incorrect length')

        # Append to sample record
        if isinstance(y, list): #repeated x values
            for ty in y:
                self.x.append(x)
                self.y.append(ty)
            ty = np.array(y).mean()
        else:
            self.x.append(x)
            self.y.append(y)
            ty = y

        # Store best point so far
        if self.minimize:
            if ty < self.opt_y:
                self.opt_y = ty
                self.opt_x = x
        else:
            if ty > self.opt_y:
                self.opt_y = ty
                self.opt_x = x


    def random_point(self, get_dict=False):
        """Get a random point within the bounds.

        Parameters
        ----------
        get_dict : bool
            Whether to return a dict w/ keys=dimension names (True), or just
            a list (False, the default).

        Returns
        -------
        list or dict
            List of parameter values, or a dict if dict=True
        """
        x = np.random.uniform(self.lb, self.ub)
        x = self._ensure_types(x)
        if get_dict: 
            x = self._make_dict(x)
        return x


    def _expected_improvement(self, x):
        """Compute the expected improvement at x.

        Parameters
        ----------
        x : ndarray
            Point at which to evaluate the expected improvement

        Returns
        -------
        float
            The expected improvement at x
        """

        # Predict performance at x
        mu, sigma = self._pred_gp(x.reshape(-1, self.num_dims),
                                  return_std=True)

        # Compute and return expected improvement
        flip = np.power(-1, self.minimize)
        z = flip*(mu-self.opt_y)/sigma
        return flip*(mu-self.opt_y)*norm.cdf(z) + sigma*norm.pdf(z)


    def next_point(self, get_dict=False, n_restarts=10):
        """Get the point with the highest expected improvement.

        Parameters
        ----------
        get_dict : bool
            Whether to return a dict w/ keys=dimension names (True), or just
            a list (False, the default).
        n_restarts : int
            Number of times to restart the optimizer.
            Default = 10

        Returns
        -------
        list or dict
            List of parameter values for the next suggested point to sample,
            or a dict if dict=True
        """

        # Fit the Gaussian process to samples so far
        self._fit_gp()

        # Find x with greatest expected improvement
        x = self.random_point()
        best_score = np.inf
        for iR in range(n_restarts):

            # Maximize expected improvement
            res = minimize(lambda x: -self._expected_improvement(x),
                           self.random_point(),
                           method='L-BFGS-B',
                           bounds=self.bounds)

            # Keep x if it's the best so far
            if res.fun < best_score:
                best_score = res.fun
                x = res.x

        # Return x with highest expected improvement
        x = self._ensure_types(x)
        if get_dict: 
            x = self._make_dict(x)
        return x


    def best_point(self, expected=True, get_dict=False, n_restarts=10):
        """Get the expected best point.

        Parameters
        ----------
        expected : bool
            Whether to return the expected best point based on the fit 
            Gaussian process (True, the default), or to return the best
            point which was actually sampled so far (False).
        get_dict : bool
            Whether to return a dict w/ keys=dimension names (True), or just
            a list (False, the default).
        n_restarts : int
            Number of times to restart the optimizer.
            Default = 10

        Returns
        -------
        list or dict
            List of x values for the best expected point,
            or a dict if dict=True
        """

        # Fit the Gaussian process to samples so far
        self._fit_gp()

        # Find x with greatest expected score
        flip = np.power(-1, self.minimize)
        score_func = lambda x: flip*self._pred_gp(x.reshape(-1,self.num_dims))
        x = self.random_point()
        best_score = np.inf
        for iR in range(n_restarts):

            # Maximize expected improvement
            res = minimize(score_func, 
                           self.random_point(),
                           method='L-BFGS-B',
                           bounds=self.bounds)

            # Keep x if it's the best so far
            if res.fun < best_score:
                best_score = res.fun
                x = res.x

        # Return x with highest expected improvement
        x = self._ensure_types(x)
        if get_dict: 
            x = self._make_dict(x)
        return x


    def get_x(self, get_dict=False):
        """Get the sampled x values.

        Parameters
        ----------
        get_dict : bool
            Whether to return a dict w/ keys=dimension names (True), or just
            a list (False, the default).

        Returns
        -------
        list of lists or dict of lists
            List sampled x values.  Each element of the list is a list with 
            the x values for that sample.
        """
        if get_dict:
            xo = dict()
            for iP in range(self.num_dims):
                xo[self.param_names[iP]] = [s[iP] for s in self.x]
            return xo
        else:
            return self.x


    def get_y(self):
        """Get the sampled y values."""
        return self.y


    def plot_surface(self, x_dim=None, y_dim=None, res=100, step=None,
                     refit=True):
        """Plot the estimated surface of the function being evaluated.

        Parameters
        ----------
        x_dim : None, str, or int
            Name or index of parameter to plot on the x axis.
            If none specified, uses the first parameter.
        y_dim : None, str, or int
            Name or index of parameter to plot on the y axis.
            If none specified, plots only a 1D plot of x_dim
        res : int > 1
            Resolution of the plot
        step : None or int
            Plot loss surface only from points up to step step.
        refit : bool
            Whether to re-fit the Gaussian process.
            Default = True
        """

        # Check inputs
        if x_dim is not None and not isinstance(x_dim, (int, str)):
            raise TypeError('x_dim must be None, str, or int')
        if y_dim is not None and not isinstance(y_dim, (int, str)):
            raise TypeError('y_dim must be None, str, or int')
        if not isinstance(res, int):
            raise TypeError('res must be an int')
        if res < 1:
            raise ValueError('res must be positive')
        if step is not None and not isinstance(step, int):
            raise TypeError('step must be None or an int')
        if isinstance(step, int) and step < 0:
            raise ValueError('step must be non-negative')

        # Convert x_dim and y_dim to int if they are strings
        if isinstance(x_dim, str):
            x_dim = self.param_names.index(x_dim)
        if isinstance(y_dim, str):
            y_dim = self.param_names.index(y_dim)

        # Set x_dim if not specified
        if x_dim is None:
            x_dim = 0

        # Fit GP to 1st ``step`` steps
        if refit:
            self._fit_gp(step=step)

        # 1D plot
        if y_dim is None:

            # Predict y as a fn of x (other params being @ middle of bounds)
            x_pred = np.ones((res, self.num_dims))
            x_pred *= (np.array(self.bounds)
                       .mean(axis=1)
                       .reshape(-1, self.num_dims))
            x_pred[:,x_dim] = np.linspace(self.lb[x_dim], self.ub[x_dim], res)
            y_pred, y_err = self._pred_gp(x_pred, return_std=True)

            # Plot the Gaussian process' estimate of the function
            plot_err(x_pred[:, x_dim], y_pred, y_err)
            plt.xlabel(self.param_names[x_dim])
            plt.ylabel('Value')

            # Plot the sampled points
            for iP in range(len(self.x) if step is None else step):
                plt.plot(self.x[iP][x_dim], self.y[iP], '.', color='0.6')

        # 2D plot
        else:
            pass
            # TODO
            """
            # Predict y as a fn of x (all other params being 0)
            x_pred = np.zeros((res*res, self.num_dims))
            xp, yp = np.meshgrid(
                np.linspace(self.lb[x_dim], self.ub[x_dim], res),
                np.linspace(self.lb[y_dim], self.ub[y_dim], res))
            x_pred[x_dim] = xp
            x_pred[y_dim] = yp
            y_pred = self._pred_gp(x_pred, return_std=True)

            # Plot the Gaussian process
            plt.imshow(y_pred.reshape((res, res)), aspect='auto', 
                       interpolation='bicubic', origin='lower')
            """

    def plot_ei_surface(self, x_dim=None, y_dim=None, res=100, step=None,
                        refit=True):
        """Plot the expected improvement surface.

        Parameters
        ----------
        x_dim : str or int
            Name or index of parameter to plot on the x axis.
            If none specified, uses the first parameter.
        y_dim : str or int
            Name or index of parameter to plot on the y axis.
            If none specified, plots only a 1D plot of x_dim
        res : int > 1
            Resolution of the plot
        step : None or int
            Plot loss surface only from points up to step step.
        refit : bool
            Whether to re-fit the Gaussian process.
            Default = True
        """

        # Check inputs
        if x_dim is not None and not isinstance(x_dim, (int, str)):
            raise TypeError('x_dim must be None, str, or int')
        if y_dim is not None and not isinstance(y_dim, (int, str)):
            raise TypeError('y_dim must be None, str, or int')
        if not isinstance(res, int):
            raise TypeError('res must be an int')
        if res < 1:
            raise ValueError('res must be positive')
        if step is not None and not isinstance(step, int):
            raise TypeError('step must be None or an int')
        if isinstance(step, int) and step < 0:
            raise ValueError('step must be non-negative')

        # Convert x_dim and y_dim to int if they are strings
        if isinstance(x_dim, str):
            x_dim = self.param_names.index(x_dim)
        if isinstance(y_dim, str):
            y_dim = self.param_names.index(y_dim)

        # Set x_dim if not specified
        if x_dim is None:
            x_dim = 0

        # Fit GP to 1st ``step`` steps
        if refit:
            self._fit_gp(step=step)

        # 1D plot
        if y_dim is None:

            # Predict y as a fn of x (other params being @ middle of bounds)
            x_pred = np.ones((res, self.num_dims))
            x_pred *= (np.array(self.bounds)
                       .mean(axis=1)
                       .reshape(-1, self.num_dims))
            x_pred[:,x_dim] = np.linspace(self.lb[x_dim], self.ub[x_dim], res)
            ei = self._expected_improvement(x_pred)

            # Plot the expected improvement
            plt.plot(x_pred[:, x_dim], ei)
            plt.xlabel(self.param_names[x_dim])
            plt.ylabel('Expected Improvement')

            # Plot the sampled points
            #for iP in range(len(self.x) if step is None else step):
            #    plt.plot(self.x[iP][x_dim], self.y[iP], '.', color='0.6')

        # 2D plot
        else:
            pass
            # TODO


    def plot_surfaces(self, x_dim=None, y_dim=None, res=100, step=None):
        """Plot both the estimated function and the expected improvement.
        """
        plt.subplot(211)
        self.plot_surface(x_dim=x_dim, y_dim=y_dim, res=res, step=step,
                          refit=True)
        plt.subplot(212)
        self.plot_ei_surface(x_dim=x_dim, y_dim=y_dim, res=res, step=step,
                             refit=False)



def optimize_params_cv(X, y, model, bounds,
                       n_splits=3,
                       max_time=None,
                       max_evals=50,
                       n_random=5,
                       n_jobs=1,
                       metric='mse'):
    """Optimize model parameters using cross-fold validation.

    Parameters
    ----------
    X : pandas DataFrame
        Independent variable values (features)
    y : pandas Series
        Dependent variable values (target)
    model : sklearn Estimator
        Predictive model to optimize
    bounds : dict
        Parameter bounds.  A dict where the keys are the parameter names, and
        the values are the parameter bounds and type (as a tuple).  Each
        parameter name should be the Pipeline step string, then a double
        underscore, then the parameter name (see example below). Each 
        tuple should be (lower_bound, upper_bound, type), where type is int or
        float.
    n_splits : int
        Number of cross-validation folds.
    max_time : None or float
        Give up after this many seconds
    max_evals : int
        Max number of cross-validation evaluations to perform.
    n_random : int
        Number of evaluations to use random parameter combinations before
        switching to Bayesian global optimization.
    n_jobs : int
        Number of parallel jobs to run (for cross-validation).    
    metric : str or sklearn scorer
        What metric to use for evaluation.  One of:

        * 'r2' - coefficient of determination (maximize)
        * 'mse' - mean squared error (minimize)
        * 'mae' - mean absolute error (minimize)
        * 'accuracy' or 'acc' - accuracy (maximize)
        * 'auc' - area under the ROC curve (maximize)

    Returns
    -------
    opt_params : dict
        Optimal parameters.  Dict of the same format as bounds, except instead
        of tuples, the values contain the optimal parameter values.
    optimizer : dsutils.optimization.GaussianProcessOptimizer
        Optimizer used to select the points.  Contains the history of all
        points which were sampled.

    Example
    -------

    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    model = Pipeline([
        ('pca', PCA(n_components=5)),
        ('regressor', Ridge(alpha=1.0))
    ])

    bounds = {
        'pca__n_components': [1, 100, int],
        'regressor__alpha': [0, 10, float]
    }

    opt_params, gpo = optimize_cv(X, y, model, bounds)

    opt_params['pca']['n_components'] #optimal # components
    opt_params['regressor']['alpha'] #optimal alpha value

    gpo.plot_surface('pca__n_components') #show loss curve vs #components
    gpo.plot_surface('pca__n_components',
                     'regressor__alpha') #show loss surface vs both
    """

    # Check inputs
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas DataFrame')
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError('y must be a pandas Series')
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same number of samples')
    if not isinstance(model, BaseEstimator):
        raise TypeError('model must be an sklearn Pipeline')
    if not isinstance(bounds, dict):
        raise TypeError('bounds must be a dict')
    if not isinstance(n_splits, int):
        raise TypeError('n_splits must be an integer')
    if n_splits < 1:
        raise ValueError('n_splits must be one or greater')
    if max_time is not None and not isinstance(max_time, float):
        raise TypeError('max_time must be None or a float')
    if max_time is not None and max_time < 0:
        raise ValueError('max_time must be positive')
    if not isinstance(max_evals, int):
        raise TypeError('max_evals must be an int')
    if max_evals < 1:
        raise ValueError('max_evals must be positive')
    if not isinstance(n_random, int):
        raise TypeError('n_random must be an int')
    if n_random < 0:
        raise ValueError('n_random must be non-negative')
    if not isinstance(n_jobs, int):
        raise TypeError('n_jobs must be an int')
    if n_jobs < 1:
        raise ValueError('n_jobs must be positive')

    # Create scorer
    if metric == 'r2':
        scorer = make_scorer(r2_score)
    elif metric == 'mse':
        scorer = make_scorer(mean_squared_error)
    elif metric == 'mae':
        scorer = make_scorer(mean_absolute_error)
    elif metric == 'accuracy' or metric == 'acc':
        scorer = make_scorer(accuracy_score)
    elif metric == 'auc':
        scorer = make_scorer(roc_auc_score)
    elif hasattr(metric, '__call__'):
        scorer = metric
    else:
        raise ValueError('metric must be a metric string or a callable')

    # Flip the score depending on the metric, such that lower is better
    if metric == 'mse' or metric == 'mae':
        minimize = True
    else:
        minimize = False

    # Collect info about parameters to optimize
    Np = len(bounds) #number of parameters
    step_params = [e for e in bounds]
    steps = [e.split('__')[0] for e in step_params]
    params = [e.split('__')[1] for e in step_params]
    lb = [bounds[e][0] for e in step_params]
    ub = [bounds[e][1] for e in step_params]
    dtypes = [bounds[e][2] for e in step_params]

    # Initialize the Gaussian process optimizer
    gpo = GaussianProcessOptimizer(lb, ub, dtypes, step_params,
                                   minimize=minimize)

    # Search for optimal parameters
    start_time = time.time()
    for i in range(max_evals):

        # Give up if we've spent too much time
        if max_time is not None and time.time()-start_time > max_time:
            break

        # Get next set of parameters to try
        if i < n_random:
            new_params = gpo.random_point()
        else:
            new_params = gpo.next_point()

        # Modify model to use new parameters
        for iP in range(Np):
            model.named_steps[steps[iP]].set_params({params[iP],
                                                     new_params[iP]})

        # Compute and store cross-validated metric
        scores = cross_val_score(model, X, y, cv=n_splits,
                                 scoring=scorer, n_jobs=n_jobs)

        # Store parameters and scores
        gpo.add_points(new_params, scores)

    # Return optimal parameters, all evaluated parameters, and scores
    opt_params = gpo.best_point(dict=True)
    return opt_params, gpo
