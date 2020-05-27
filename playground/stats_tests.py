import importlib
import functools
import re
import warnings

import numpy as np

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


@conditional_jit(cache=True)
def std(rvs, size):
    return ((np.sum(rvs ** 2) / size) - (rvs.sum() / size) ** 2) ** 0.5
