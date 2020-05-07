"""
Functions to compute kernel density estimates
"""

import numpy as np
from scipy.signal import gaussian, convolve  # pylint: disable=no-name-in-module

# Own
from .density_utils import check_type, len_warning, get_grid, get_bw, get_mixture

def estimate_density(
    # pylint: disable=too-many-arguments,too-many-locals
    x,
    bw="silverman",
    grid_len=256,
    extend=True,
    bound_correction=False,
    adaptive=False,
    extend_fct=0.5,
    bw_fct=1,
    bw_return=False,
    custom_lims=None,
):
    """
    1 dimensional density estimation.
    
    Given an array of data points `x` it returns an estimate of
    the probability density function that generated the samples in `x`.
    
    Parameters
    ----------
    x : 1-D numpy array
        1 dimensional array of sample data from the 
        variable for which a density estimate is desired.
    bw: int, float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "lscv", "sj", "isj" or "experimental".
        Defaults to "silverman".
    grid_len: int, optional
        The number of intervals used to bin the data points.
        Defaults to 256.
    extend: boolean, optional
        Whether to extend the observed range for `x` in the estimation.
        It extends each bound by a multiple of the standard deviation of `x`
        given by `extend_fct`. Defaults to True.
    bound_correction: boolean, optional
        Whether to perform boundary correction on the bounds of `x` or not.
        Defaults to False.
    adaptive: boolean, optional
        Indicates if the bandwidth is adaptative or not.
        It is the recommended approach when there are multiple modalities
        with different spread. 
        It is not compatible with convolution. Defaults to False.
    extend_fct: float, optional
        Number of standard deviations used to widen the 
        lower and upper bounds of `x`. Defaults to 0.5.
    bw_fct: float, optional
        A value that multiplies `bw` which enables tuning smoothness by hand.
        Must be positive. Defaults to 1 (no modification).
    bw_return: bool, optional
        Whether to return the estimated bandwidth in addition to the 
        other objects. Defaults to False.
    custom_lims: list or tuple, optional
        A list or tuple of length 2 indicating custom bounds
        for the range of `x`. Defaults to None which disables custom bounds.
    
    Returns
    -------
    grid : Gridded numpy array for the x values.
    pdf : Numpy array for the density estimates.
    bw: optional, the estimated bandwidth.
    """

    # Check `x` is from appropiate type
    x = check_type(x)
    
    # Assert `bw_fct` is numeric and positive
    # Note: a `bool` will not trigger the first AssertionError, 
    #       but it is not a problem since True will be 1
    #       and False will be 0, which triggers the second AssertionError.
    assert isinstance(bw_fct, (int, float))
    assert bw_fct > 0

    # Preliminary calculations
    x_len = len(x)
    x_min = np.min(x)
    x_max = np.max(x)
    x_std = np.std(x)

    # Length warning: Not completely sure if it is necessary
    len_warning(x_len)

    grid_len, grid_min, grid_max = get_grid(
        x_min, x_max, x_std, extend_fct, grid_len, custom_lims, extend, bound_correction
    )

    # Bandwidth estimation
    bw = bw_fct * get_bw(x, bw)

    # Density estimation
    if adaptive:
        grid, pdf = kde_adaptive(x, bw, grid_len, grid_min, grid_max, bound_correction)
    else:
        grid, pdf = kde_convolution(x, bw, grid_len, grid_min, grid_max, bound_correction)

    if bw_return:
        return grid, pdf, bw

    return grid, pdf


def estimate_density_em(
    # pylint: disable=too-many-arguments,too-many-locals
    x,
    gauss_n=None,
    grid_len=256,
    extend=True,
    extend_fct=0.5,
    custom_lims=None,
    iter_max=200,
    tol=0.002,
    verbose=False,
):
    """
    Adaptative Gaussian KDE by E-M algorithm.

    Parameters
    ----------
    x : numpy array
        1 dimensional array of sample data from the variable for which a
        density estimate is desired.
    gauss_n : float, optional
        Number of Gaussian kernels to be used in the mixture.
        More kernels mean more accuracy but larger computation times.
        Based on own experience, upper bound was set to 30.
        Defaults to None, which picks the number heuristically.
    grid_len : int, optional
        Number of points where the kernel is evaluated.
        Defaults to 256.
    extend: boolean, optional
        Whether to extend the domain of the observed data or not.
        Defaults to True.
    extend_fct: float, optional
        The value that multiplies the standard deviation of `x`
        that is used o extend the domain.
        Defaults to 0.5.
    custom_lims: list or tuple, optional
        Custom limits for the domain of `x`.
        The length of the list/tuple must be 2.
        Defaults to None.
    iter_max: float, optional
        Number of maximum iterations for the E-M algorithm
        Defaults to 200.
    tol: float, optional
        Maximum tolerated difference between steps of the E-M algorithm
        Defaults to 0.005 (experimental).
    verbose: boolean, optional
        Indicates whether to print information related to each E-M step.
        Defaults to False.

    Returns
    -------
    grid : Gridded numpy array for the x values.
    pdf : Numpy array for the density estimates.

    """
    assert isinstance(custom_lims, (list, tuple))
    assert len(custom_lims) == 2
    assert custom_lims[0] < custom_lims[1]

    # Check `x` is from appropiate type
    x = check_type(x)

    # Preliminary calculations
    x_len = len(x)
    x_min = np.min(x)
    x_max = np.max(x)
    x_std = np.std(x)

    # Length warning:
    # Not completely sure whether it is necessary
    len_warning(x_len)

    grid_len, grid_min, grid_max = get_grid(
        x_min, x_max, x_std, extend_fct, grid_len, custom_lims, extend
    )

    grid = np.linspace(grid_min, grid_max, num=grid_len)

    # Set up number of (Gaussian) kernels - heuristic
    if gauss_n is None:
        gauss_n = int(np.ceil(x_len ** 0.20)) + 10

    if gauss_n > 30:
        gauss_n = 30
    gauss_n = int(gauss_n)

    # Set up initial values for EM
    gauss_w = np.full((gauss_n), 1 / gauss_n)
    mean = x[np.linspace(0, x_len - 1, gauss_n, dtype="int32")]
    variance = np.full((gauss_n), (x_std ** 2) / (gauss_n * 0.5))  # heuristic

    llh_matrix = np.zeros((x_len, gauss_n))
    llh_current = float("-inf")

    for ite in range(iter_max):

        llh_prev = llh_current

        # Expectation step
        z_sq = ((x - mean[:, None]) ** 2) / variance[:, None]
        llh_matrix = (
            -0.5 * (np.log(2 * np.pi) + z_sq)
            - np.log(np.sqrt(variance[:, None]))
            + np.log(gauss_w[:, None])
        )
        llh_matrix = np.transpose(llh_matrix)

        # Log-sum-exp trick
        llh_max = np.amax(llh_matrix, axis=1)
        joint_probs = np.exp(llh_matrix - llh_max[:, None])
        pdf = np.sum(joint_probs, axis=1)
        logpdf = np.log(pdf) + llh_max
        llh_current = np.sum(logpdf)
        resp = joint_probs / pdf[:, None]

        # Maximization step
        # Estimate means and variances
        gauss_w = np.sum(resp, 0)
        mean = np.dot(x, resp) / gauss_w
        variance = np.diag(np.dot((x - mean[:, None]) ** 2, resp) / gauss_w)

        # Estimate new weights
        gauss_w /= x_len

        # End of EM
        if verbose:
            print("Step number: " + str(ite))
            print("Mean values:      " + str(mean))
            print("Variance values:  " + str(variance))
            print("Gaussian weights: " + str(gauss_w))
            print("---------------------------------------")

        if np.abs((llh_current - llh_prev) / llh_current) < tol:
            break

    # Evaluate grid points in the estimated pdf
    pdf = get_mixture(grid, mean, variance, gauss_w)

    return grid, pdf


def kde_convolution(x, bw, grid_len, grid_min, grid_max, bound_correction):
    # pylint: disable=too-many-arguments
    """
    1 dimensional Gaussian kernel density estimation via 
    convolution of the binned relative frequencies and a Gaussian filter.
    It does NOT use FFT because there is no real gain for the 
    number of bins used.
    This is an internal function used by `estimate_density()`.
    
    Parameters
    ----------
    x : 1-D numpy array
        1 dimensional array of sample data from the 
        variable for which a density estimate is desired.
    bw: int or float
        Bandwidth parameter, a.k.a. the standard deviation
        of the Gaussian kernel.
    grid_len: int
        The number of intervals used to bin the data points.
    grid_min: float
        Minimum value of the grid
    grid_max: float
        Maximum value of the grid
    bound_correction: boolean
        Whether to perform boundary correction on the bounds of `x` or not.
        
    Returns
    -------
    grid : Gridded numpy array for the x values.
    pdf : Numpy array for the density estimates.
    """
    # Calculate relative frequencies per bin
    f, _ = np.histogram(x, bins=grid_len, range=(grid_min, grid_max), density=True)

    # Bandwidth must consider the bin width
    bin_width = (grid_max - grid_min) / (grid_len - 1)
    bw /= bin_width

    # Instantiate kernel signal
    kernel = gaussian(120, bw)

    if bound_correction:
        npad = int(grid_len / 4)
        f = np.concatenate([f[npad:0:-1], f, f[grid_len : grid_len - npad : -1]])
        pdf = convolve(f, kernel, mode="same", method="direct")[npad : npad + grid_len]
        pdf = pdf / sum(kernel)
    else:
        pdf = convolve(f, kernel, mode="same", method="direct") / sum(kernel)

    grid = np.linspace(grid_min, grid_max, num=grid_len)

    return grid, pdf


def kde_adaptive(x, bw, grid_len, grid_min, grid_max, bound_correction):
    # pylint: disable=too-many-arguments,too-many-locals
    """
    1 dimensional adaptive Gaussian kernel density estimation.
    The implementation uses the binning technique. 
    However, since there is not an unique `bw`, the convolution 
    is not possible.
    The alternative implemented in this function is known as Anderson's method.
    This is an internal function used by `estimate_density()`.
    
    Parameters
    ----------
    x : 1-D numpy array
        1 dimensional array of sample data from the 
        variable for which a density estimate is desired.
    bw: int or float
        Bandwidth parameter, a.k.a. the standard deviation
        of the Gaussian kernel.
        This bandwidth parameter is then modified for each bin.
    grid_len: int
        The number of intervals used to bin the data points.
    grid_min: float
        Minimum value of the grid
    grid_max: float
        Maximum value of the grid
    bound_correction: boolean
        Whether to perform boundary correction on the bounds of `x` or not.
        
    Returns
    -------
    grid : Gridded numpy array for the x values.
    pdf : Numpy array for the density estimates.
    
    """
    # Computations for bandwidth adjustment
    pilot_grid, pilot_pdf = kde_convolution(x, bw, grid_len, grid_min, grid_max, bound_correction)

    # Step 2: Determine the modification factors
    # a: Geometric mean of KDE evaluated at sample points
    # EXTREMELY important to calculate geom_mean with interpolated points
    # and `adj_factor` with `pilot_pdf`.
    pdf_interp = np.interp(x, pilot_grid, pilot_pdf)
    geom_mean = np.exp(np.mean(np.log(pdf_interp)))

    # b: Compute modification factors
    # Power of c = 0.5 -> Anderson's method
    adj_factor = (geom_mean / pilot_pdf) ** 0.5
    bw_adj = bw * adj_factor

    # Estimation of Gaussian KDE via binned method (convolution not possible)
    grid_count, grid = np.histogram(x, bins=grid_len, range=(grid_min, grid_max))
    grid = (grid[1:] + grid[:-1]) / 2

    if bound_correction:

        x_pad_min = (2 * grid_min) - grid_max
        x_pad_max = (2 * grid_max) - grid_min
        grid_pad_len = 3 * grid_len

        grid = np.linspace(x_pad_min, x_pad_max, num=grid_pad_len)
        grid = (grid[1:] + grid[:-1]) / 2

        grid_count = np.concatenate(
            [
                grid_count[grid_pad_len:0:-1],
                grid_count,
                grid_count[grid_len : grid_len - grid_pad_len : -1],
            ]
        )

        bw_adj = np.concatenate(
            [bw_adj[grid_pad_len:0:-1], bw_adj, bw_adj[grid_len : grid_len - grid_pad_len : -1]]
        )

        pdf_mat_num = (
            np.exp(-0.5 * ((grid - grid[:, None]) / bw_adj[:, None]) ** 2) * grid_count[:, None]
        )
        pdf_mat_den = (2 * np.pi) ** 0.5 * bw_adj[:, None]
        pdf_mat = pdf_mat_num / pdf_mat_den
        pdf = np.sum(pdf_mat[:, grid_len : (2 * grid_len)], axis=0) / len(x)
        grid = grid[grid_len : (2 * grid_len)]

    else:
        pdf_mat_num = (
            np.exp(-0.5 * ((grid - grid[:, None]) / bw_adj[:, None]) ** 2) * grid_count[:, None]
        )
        pdf_mat_den = (2 * np.pi) ** 0.5 * bw_adj[:, None]
        pdf_mat = pdf_mat_num / pdf_mat_den
        pdf = np.sum(pdf_mat, axis=0) / len(x)

    return grid, pdf
