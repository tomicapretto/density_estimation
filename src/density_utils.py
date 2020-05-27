"""
Auxiliary functions to compute kernel density estimates
"""

from warnings import warn
import numpy as np

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
    #else:
        # SUPPRESSED THIS BECAUSE IT CONSUMES A LOT OF TIME.
        # any_bool = any(isinstance(x_i, (bool, np.bool_)) for x_i in x)
        # if any_bool:
        #    raise ValueError((
        #        f"At least one element in `x` is boolean.\n"
        #        f"Can't produce a density estimator for booleans.\n"
        #        f"Please input a numeric list or numpy array."
        #    ))
        
    # Will raise an error if `x` can't be casted to numeric
    x = np.asfarray(x)

    if x.ndim != 1:
        raise ValueError((
            f"Unsupported dimension number.\n"
            f"Density estimator only works with 1-dimensional data, not {x.ndim}."
        ))
        
    return x
    
def check_custom_lims(custom_lims, x_min, x_max):
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
        
    if len(custom_lims) != 2:
        raise AttributeError(f"`len(custom_lims)` must be 2, not {len(custom_lims)}.")
        
    any_bool = any(isinstance(i, bool) for i in custom_lims)
    if any_bool:
        raise TypeError("Elements of `custom_lims` must be numeric or None, not bool.")
    
    if custom_lims[0] is None:
        custom_lims[0] = x_min
        
    if custom_lims[1] is None:
        custom_lims[1] = x_max
    
    types = (int, float, np.integer, np.float)    
    all_numeric = all(isinstance(i, types) for i in custom_lims)
    if not all_numeric:
        raise TypeError((
            f"Elements of `custom_lims` must be numeric or None.\n"
            f"At least one of them is not."
        ))
    
    if not custom_lims[0] < custom_lims[1]:
        raise AttributeError(f"`custom_lims[0]` must be smaller than `custom_lims[1]`.")
    
    return custom_lims

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
        This does not impacts directly in the output, but is used
        to override `extend`.
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
        custom_lims = check_custom_lims(custom_lims, x_min, x_max)
        grid_min = custom_lims[0]
        grid_max = custom_lims[1]
    elif extend and not bound_correction:
        grid_extend = extend_fct * x_std
        grid_min = x_min - grid_extend
        grid_max = x_max + grid_extend
    else:
        grid_min = x_min
        grid_max = x_max
    return grid_min, grid_max, grid_len

def gaussian_mixture(grid, mu, var, weight):
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
    A helper function of `gaussian_mixture`. 
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
