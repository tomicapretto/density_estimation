import importlib
import functools
import re
import warnings

import numpy as np
from scipy.signal import convolve
from scipy.signal.windows import gaussian

from fast_histogram import histogram1d

def numba_check():
    """Check if numba is installed."""
    numba = importlib.util.find_spec("numba")
    return numba is not None

class Numba:
    """A class to toggle numba states."""

    numba_flag = numba_check()

    @classmethod
    def disable_numba(cls):
        """To disable numba."""
        cls.numba_flag = False

    @classmethod
    def enable_numba(cls):
        """To enable numba."""
        if numba_check():
            cls.numba_flag = True
        else:
            raise ValueError("Numba is not installed")

class lazy_property:  # pylint: disable=invalid-name
    """Used to load numba first time it is needed."""

    def __init__(self, fget):
        """Lazy load a property with `fget`."""
        self.fget = fget

        # copy the getter function's docstring and other attributes
        functools.update_wrapper(self, fget)

    def __get__(self, obj, cls):
        """Call the function, set the attribute."""
        if obj is None:
            return self

        value = self.fget(obj)
        setattr(obj, self.fget.__name__, value)
        return value


class maybe_numba_fn:  # pylint: disable=invalid-name
    """Wrap a function to (maybe) use a (lazy) jit-compiled version."""

    def __init__(self, function, **kwargs):
        """Wrap a function and save compilation keywords."""
        self.function = function
        self.kwargs = kwargs

    @lazy_property
    def numba_fn(self):
        """Memoized compiled function."""
        try:
            numba = importlib.import_module("numba")
            numba_fn = numba.jit(**self.kwargs)(self.function)
        except ImportError:
            numba_fn = self.function
        return numba_fn

    def __call__(self, *args, **kwargs):
        """Call the jitted function or normal, depending on flag."""
        if Numba.numba_flag:
            return self.numba_fn(*args, **kwargs)
        else:
            return self.function(*args, **kwargs)


def conditional_jit(_func=None, **kwargs):
    """Use numba's jit decorator if numba is installed.
    Notes
    -----
        If called without arguments  then return wrapped function.
        @conditional_jit
        def my_func():
            return
        else called with arguments
        @conditional_jit(nopython=True)
        def my_func():
            return
    """
    if _func is None:
        return lambda fn: functools.wraps(fn)(maybe_numba_fn(fn, **kwargs))
    else:
        lazy_numba = maybe_numba_fn(_func, **kwargs)
        return functools.wraps(_func)(lazy_numba)

@conditional_jit(cache=True)
def histogram(data, bins, range_hist=None):
    bin_counts, bin_edges = np.histogram(data, bins=bins, range=range_hist)
    bin_density = bin_counts /  (bin_edges[1] - bin_edges[0]) / len(data)
    return bin_edges, bin_counts, bin_density

def fast_kde2(x, cumulative=False, bw=4.5, x_min=None, x_max=None, grid_len=200, bound_correction=True):
    
    # Check data
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        warnings.warn("KDE plot failed. Can't generate a density estimate for non-finite data.")
        return np.array([np.nan]), np.nan
    
    # Check/specify boundaries
    data_min = x.min()
    data_max = x.max()
    
    if x_min is None:
        x_min = data_min
    else:
        assert x_min <= data_min, "`x_min` can't be larger than the minimum of the data."   
    if x_max is None:
        x_max = data_max
    else:
        assert x_max >= data_max, "`x_max` can't be smaller than the maximum of the data."
        
    x_range = (x_max - x_min)
    data_range = (data_max - data_min)
        
    # Set up number of bins / grid length
    if isinstance(grid_len, bool):
        raise TypeError("`grid_len` must be numeric, not bool.")
    elif not isinstance(grid_len, (int, float)):
        raise TypeError("`grid_len` must be numeric, not {}".format(type(grid_len)))
    
    # Should this go before of after checking size?
    grid_len = int(grid_len * (x_range / data_range))
    if grid_len > 512:
        grid_len = 512
    if grid_len < 100:
        grid_len = 100

    bin_edges, bin_counts, bin_dens = histogram(x, grid_len, range_hist=(x_min, x_max))
    
    # AZ bandwidth rule
    # This rule is constructed in such a way it is ready for the binned KDE
    # i.e. no need to divide by `bin_width`.
    x_len = len(x)
    x_log_len = np.log(x_len) * bw
    bw_az = x_log_len * x_len ** (-0.2)
    
    # Instantiate kernel signal
    # Two options for `kern_nx`
    # Decreasing, starting in 50 for 100 points and ~ 40 for 20k points.
    kernel_n = int(bw_az * 2 * np.pi)  
    
    # Fixed in 50 or other suitable number?.
    # kernel_n = 50
    kernel = gaussian(kernel_n, bw_az)
    
    if bound_correction:
        npad = int(grid_len / 5)
        bin_dens = np.concatenate([
            bin_dens[npad - 1:: -1], 
            bin_dens, 
            bin_dens[grid_len : grid_len - npad - 1: -1]]
        )
        pdf = convolve(bin_dens, kernel, mode="same", method="direct")[npad : npad + grid_len]
        pdf /= bw_az * (2 * np.pi) ** 0.5
    else:
        pdf = convolve(bin_dens, kernel, mode="same", method="direct")
        pdf /= bw_az * (2 * np.pi) ** 0.5       
                         
    grid = (bin_edges[1:] + bin_edges[:-1]) / 2 
    
    if cumulative:
        pdf = pdf.cumsum() / pdf.sum()

    return grid, pdf


# VERSION ORIGINAL!
def fast_kde(x, cumulative=False, bw=4.5, xmin=None, xmax=None):
    """Fast Fourier transform-based Gaussian kernel density estimate (KDE).
    The code was adapted from https://github.com/mfouesneau/faststats
    Parameters
    ----------
    x : Numpy array or list
    cumulative : bool
        If true, estimate the cdf instead of the pdf
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy).
    xmin : float
        Manually set lower limit.
    xmax : float
        Manually set upper limit.
    Returns
    -------
    density: A gridded 1D KDE of the input points (x)
    xmin: minimum value of x
    xmax: maximum value of x
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        warnings.warn("kde plot failed, you may want to check your data")
        return np.array([np.nan]), np.nan, np.nan

    len_x = len(x)
    n_points = 200 if (xmin or xmax) is None else 500

    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)

    assert np.min(x) >= xmin
    assert np.max(x) <= xmax

    log_len_x = np.log(len_x) * bw

    n_bins = min(int(len_x ** (1 / 3) * log_len_x * 2), n_points)
    if n_bins < 2:
        warnings.warn("kde plot failed, you may want to check your data")
        return np.array([np.nan]), np.nan, np.nan

    # hist, bin_edges = np.histogram(x, bins=n_bins, range=(xmin, xmax))
    # grid = hist / (hist.sum() * np.diff(bin_edges))

    _, _, grid = histogram(x, n_bins, range_hist=(xmin, xmax))

    scotts_factor = len_x ** (-0.2)
    kern_nx = int(scotts_factor * 2 * np.pi * log_len_x)
    kernel = gaussian(kern_nx, scotts_factor * log_len_x)

    npad = min(n_bins, 2 * kern_nx)
    grid = np.concatenate([grid[npad:0:-1], grid, grid[n_bins : n_bins - npad : -1]])
    density = convolve(grid, kernel, mode="same", method="direct")[npad : npad + n_bins]
    norm_factor = (2 * np.pi * log_len_x ** 2 * scotts_factor ** 2) ** 0.5

    density /= norm_factor

    if cumulative:
        density = density.cumsum() / density.sum()

    return density, xmin, xmax



