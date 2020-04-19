"""
Auxiliary functions to compute kernel density estimates
"""
from warnings import warn
import numpy as np
from bandwidth import bw_scott, bw_silverman, bw_lscv, bw_sj, bw_isj, bw_experimental


def check_type(x):
    """
    Write me!
    """
    if not isinstance(x, (list, np.ndarray)):
        error_str = f"`x` is of the wrong type.\n"
        error_str += f"Can't produce a density estimator for {type(x)}.\n"
        error_str += f"Please input a numeric list or numpy array."
        raise ValueError(error_str)

    # Will raise an error if `x` is not numeric
    x = np.asfarray(x)

    if x.ndim != 1:
        error_str = f"Unsupported dimension number.\n"
        error_str += f"Density estimator only works with 1-dimensional data, "
        error_str += f"not {x.ndim}."
        raise ValueError(error_str)

    return x


def len_warning(x):
    """
    Write me!
    """
    if x < 50:
        warn_str = f"The estimation may be unstable for such a few data points.\n"
        warn_str += f"Try a histogram or dotplot instead."
        warn(warn_str, Warning)


def get_grid(
    x_min, x_max, x_std, extend_fct, grid_len, custom_lims, extend=True, bound_correction=False
):
    # pylint: disable=too-many-arguments
    """
    Write me!
    """
    # Set up number of bins
    # Should I enable larger grids?
    if grid_len > 512:
        grid_len = 512
    if grid_len < 100:
        grid_len = 100
    grid_len = int(grid_len)

    # Set up domain
    # `custom_lims` overrides `extend`
    # `bound_correction` overrides `extend`
    if custom_lims is not None:
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
    Write me!
    """
    if isinstance(bw, (int, float)):
        if bw < 0:
            raise ValueError(f"Numeric `bw` must be positive.\nInput: {bw:.4f}.")

    elif isinstance(bw, str):
        bw = select_bw_method(x, bw)
    else:
        raise ValueError(
            f"""Unrecognized `bw` argument.\nExpected a positive numeric or one of """
            """the following strings:{list(BW_METHODS.keys())}."""
        )
    return bw


BW_METHODS = {
    "scott": bw_scott,
    "silverman": bw_silverman,
    "lscv": bw_lscv,
    "sj": bw_sj,
    "isj": bw_isj,
    "experimental": bw_experimental,
}


def select_bw_method(x, method="isj"):
    """
    Write me!
    """
    method_lower = method.lower()

    if method_lower not in BW_METHODS.keys():
        error_string = "Unrecognized bandwidth method.\n"
        error_string += f"Input is: {method}.\n"
        error_string += f"Expected one of: {list(BW_METHODS.keys())}."
        raise ValueError(error_string)

    bw = BW_METHODS[method_lower](x)
    return bw


def get_mixture(grid, mu, var, weight):
    """
    Write me!
    """
    out = np.sum(list((map(lambda m, v, w: norm_pdf(grid, m, v, w), mu, var, weight))), axis=0)
    return out


def norm_pdf(grid, mu, var, weight):
    """
    Write me!
    """
    # 1 / np.sqrt(2 * np.pi) = 0.3989423
    out = np.exp(-0.5 * ((grid - mu)) ** 2 / var) * 0.3989423 * (var ** -0.50) * weight
    return out
