"""
Auxiliary functions to compute kernel density estimates
"""

from warnings import warn
import numpy as np

# Own
from .bandwidth import bw_scott, bw_silverman, bw_lscv, bw_sj, bw_isj, bw_experimental

def check_type(x):
    """
    Checks the input is of the correct type.
    It only accepts numeric lists/numpy arrays of 1 dimension.
    If input is not of the correect type, an informative message is thrown.
    
    Parameters
    ----------
    x : Object whose type is checked before computing the KDE.
    
    Returns
    -------
    x : 1-D numpy array
        If no error is thrown, a 1 dimensional array of 
        sample data from the variable for which a density estimate is desired.
    
    """
    if not isinstance(x, (list, np.ndarray)):
        raise ValueError((
            f"`x` is of the wrong type.\n"
            f"Can't produce a density estimator for {type(x)}.\n"
            f"Please input a numeric list or numpy array."
        ))
    else:
        any_bool = any(isinstance(x_i, (bool, np.bool_)) for x_i in x)
        if any_bool:
            raise ValueError((
                f"At least one element in `x` is boolean.\n"
                f"Can't produce a density estimator for booleans.\n"
                f"Please input a numeric list or numpy array."
            ))
        
    # Will raise an error if `x` can't be casted to numeric
    x = np.asfarray(x)

    if x.ndim != 1:
        raise ValueError((
            f"Unsupported dimension number.\n"
            f"Density estimator only works with 1-dimensional data, not {x.ndim}."
        ))
        
    return x
    
def check_custom_lims(custom_lims):
    """
    Checks whether `custom_lims` are of the correct type.
    It accepts numeric lists/tuples of length 2.
    
    Parameters
    ----------
    custom_lims : Object whose type is checked.

    Returns
    -------
    None: Object of type None
    
    """
    if not isinstance(custom_lims, (list, tuple)):
        raise TypeError((
            f"`custom_lims` must be a numeric list or tuple of length 2.\n"
            f"Not an object of {type(custom_lims)}."
        ))
        
    any_bool = any(isinstance(i, bool) for i in custom_lims)
    if any_bool:
        raise TypeError("Elements of `custom_lims` must be numeric, not bool")
        
    all_numeric = all(isinstance(i, (int, float)) for i in custom_lims)
    if not all_numeric:
        raise TypeError((
            f"Elements of `custom_lims` must be numeric.\n"
            f"At least one of them is not."
        ))
        
    if len(custom_lims) != 2:
        raise AttributeError(f"`len(custom_lims)` must be 2, not {len(custom_lims)}.")
    
    if not custom_lims[0] < custom_lims[1]:
        raise AttributeError(f"`custom_lims[0]` must be smaller than `custom_lims[1]`.")


def len_warning(x):
    """
    Checks whether the length of the array used to estimate 
    the density function is not too short. 
    If `len(x)` < 50 it raises a warning.
    
    Parameters
    ----------
    x : 1-D numpy array
        1 dimensional array of sample data from the variable for which a
        density estimate is desired.
    
    Returns
    -------
    None: Object of type None
    
    """
    if x < 50: # Maybe a larger number?
        warn((
            f"The estimation may be unstable for such a few data points.\n"
            f"Try a histogram or dotplot instead."
        ), Warning)

def get_grid(
    x_min, x_max, x_std, extend_fct, grid_len, 
    custom_lims, extend=True, bound_correction=False): # pylint: disable=too-many-arguments
    """
    Computes the grid that bins the data used to estimate the density function
    
    Parameters
    ----------
    x_min : float
        Minimum value of the data
    x_max: float
        Maximum value of the data.
    x_std: float
        Standard deviation of the data.
    extend_fct: bool
        Indicates the factor by which `x_std` is multiplied
        to extend the range of the data.
    grid_len: int
        Number of bins
    custom_lims: tuple or list
        Custom limits for the domain of the density estimation. 
        Must be numeric of length 2.
    extend: bool, optional
        Whether to extend the range of the data or not.
        Default is True.
    bound_correction: bool, optional
        Whether the density estimations performs boundary correction or not.
        This does not impacts in the directly in the output, but is used
        to override extend.
        Default is False.
        
    Returns
    -------
    grid_len: int
        Number of bins
    grid_min: float
        Minimum value of the grid
    grid_max: float
        Maximum value of the grid
        
    """
    
    # Set up number of bins
    # Should I enable larger grids?
    if grid_len > 1024:
        grid_len = 1024
    if grid_len < 100:
        grid_len = 100
    grid_len = int(grid_len)

    # Set up domain
    # `custom_lims` overrides `extend`
    # `bound_correction` overrides `extend`
    if custom_lims is not None:
        check_custom_lims(custom_lims)
        grid_min = custom_lims[0]
        grid_max = custom_lims[1]
    elif extend and not bound_correction:
        grid_extend = extend_fct * x_std
        grid_min = x_min - grid_extend
        grid_max = x_max + grid_extend
    else:
        grid_min = x_min
        grid_max = x_max
    return grid_len, grid_min, grid_max


def get_bw(x, bw):
    """
    Computes bandwidth for a given data `x` and `bw`.
    Also checks `bw` is correctly specified.
    
    Parameters
    ----------
    x : 1-D numpy array
        1 dimensional array of sample data from the 
        variable for which a density estimate is desired.
    bw: int, float or str
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth.
    
    Returns
    -------
    bw: float
        Bandwidth
    """
    if isinstance(bw, bool):
        raise ValueError((
            f"`bw` must not be of type `bool`.\n"
            f"Expected a positive numeric or one of the following strings:\n"
            f"{list(BW_METHODS.keys())}."))
    
    if isinstance(bw, (int, float)):
        if bw < 0:
            raise ValueError(f"Numeric `bw` must be positive.\nInput: {bw:.4f}.")

    elif isinstance(bw, str):
        bw_fun = select_bw_method(bw)
        bw = bw_fun(x)
    else:
        raise ValueError((
            f"Unrecognized `bw` argument.\n"
            f"Expected a positive numeric or one of the following strings:\n"
            f"{list(BW_METHODS.keys())}."))
    return bw

BW_METHODS = {
    "scott": bw_scott,
    "silverman": bw_silverman,
    "lscv": bw_lscv,
    "sj": bw_sj,
    "isj": bw_isj,
    "experimental": bw_experimental,
}

def select_bw_method(method="isj"):
    """
    Selects a function to compute the bandwidth.
    Also checks method `bw` is correctly specified.
    Otherwise, throws an error.
    
    Parameters
    ----------
    method : str
        Method to estimate the bandwidth.
    
    Returns
    -------
    bw_fun: function
        Function to compute the bandwidth.
    """
    method_lower = method.lower()

    if method_lower not in BW_METHODS.keys():
        raise ValueError((
            f"Unrecognized bandwidth method.\n"
            f"Input is: {method}.\n"
            f"Expected one of: {list(BW_METHODS.keys())}."
        ))
    bw_fun = BW_METHODS[method_lower]
    return bw_fun

def get_mixture(grid, mu, var, weight):
    """
    Computes the probability density function of a 
    mixture of Gaussian distributions.
    
    Length of mu, var and weight must be the same.
    
    Parameters
    ----------
    grid : 1-D numpy array
        Numeric and dense grid that represents 
        the domain of the mixture
    mu : list, numeric
        Mean of the gaussian components
    var : list, numeric
        Variance of the gaussian components
    weight: list, numeric
        Weight of the gaussian components
    
    Returns
    -------
    out: 1-D numpy array
        The value of the mixture probability density function at each
        point in `grid`.
    """
    out = np.sum(list((map(lambda m, v, w: norm_pdf(grid, m, v) * w, mu, var, weight))), axis=0)
    return out


def norm_pdf(grid, mu, var):
    """
    A helper function of `get_mixture`. 
    Computes the probability density function of a Gaussian distribution.
    
    Length of mu, var and weight must be the same.
    
    Parameters
    ----------
    grid : 1-D numpy array
        Numeric and dense grid that represents 
        the domain of the mixture
    mu : list, numeric
        Mean of the gaussian components
    var : list, numeric
        Variance of the gaussian components
    weight: list, numeric
        Weight of the gaussian components
    
    Returns
    -------
    out: 1-D numpy array
        The value of the Gaussian pdf at each point in `grid`.
    """
    # 1 / np.sqrt(2 * np.pi) = 0.3989423
    out = np.exp(-0.5 * ((grid - mu)) ** 2 / var) * 0.3989423 * (var ** -0.50)
    return out
