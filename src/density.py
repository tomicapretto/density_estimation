"""
Functions to compute kernel density estimates
"""

import numpy as np
from scipy.signal import gaussian, convolve  # pylint: disable=no-name-in-module
from fast_histogram import histogram1d
from warnings import warn

# Own
from .density_utils import check_type, len_warning, get_grid, gaussian_mixture
from .bandwidth import get_bw

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
    x : 1D numpy array
        Data used to calculate the density estimation.
        Theoritically it is a random sample obtained from $f$, 
        the true probability density function we aim to estimate.
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
    x_min = x.min()
    x_max = x.max()
    x_std = (((x ** 2).sum() / x_len) - (x.sum() / x_len) ** 2) ** 0.5
    x_range = x_max - x_min

    # Length warning. Not completely sure if it is necessary
    len_warning(x_len)
    
    # Determine grid
    grid_min, grid_max, grid_len = get_grid(
        x_min, x_max, x_std, extend_fct, grid_len, custom_lims, extend, bound_correction
    )
    
    grid_counts = histogram1d(x, bins=grid_len, range=(grid_min, grid_max))
    grid_edges = np.linspace(grid_min, grid_max, num=grid_len + 1)  

    # Bandwidth estimation
    bw = bw_fct * get_bw(x, bw, grid_counts=grid_counts, x_std=x_std, x_range=x_range)

    # Density estimation
    if adaptive:
        grid, pdf = kde_adaptive(x, bw, grid_edges, grid_counts, grid_len, bound_correction)
    else:
        grid, pdf = kde_convolution(x, bw, grid_edges, grid_counts, grid_len, bound_correction)
    
    if bw_return:
        return grid, pdf, bw
    else:
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
    tol=0.001,
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
    # Check `x` is from appropiate type
    x = check_type(x)

    # Preliminary calculations
    x_len = len(x)
    x_min = x.min()
    x_max = x.max()
    x_std = (((x ** 2).sum() / x_len) - (x.sum() / x_len) ** 2) ** 0.5    
    
    # Length warning:
    # Not completely sure whether it is necessary
    len_warning(x_len)

    grid_min, grid_max, grid_len = get_grid(
        x_min, x_max, x_std, extend_fct, grid_len, custom_lims, extend
    )
    grid = np.linspace(grid_min, grid_max, num=grid_len)

    # Set up number of (Gaussian) kernels - heuristic
    if gauss_n is None:
        gauss_n = int(np.ceil(x_len ** 0.33)) + 10
    if gauss_n > 50:
        gauss_n = 50
    gauss_n = int(gauss_n)

    # Set up initial values for EM
    gauss_w = np.full((gauss_n), 1 / gauss_n)
    mean = x[np.linspace(0, x_len - 1, gauss_n, dtype="int32")]
    variance = np.full((gauss_n), (x_std ** 2) / (gauss_n * 0.5))  # heuristic

    llh_matrix = np.zeros((x_len, gauss_n))
    llh_current = float("-inf")
    converged = 0
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
        
        # Add eps to avoid numerical problems
        mean += 1e-9
        variance.flags.writeable = True
        variance += 1e-9
        gauss_w += 1e-9

        # End of EM
        if verbose:
            print("Step number: " + str(ite))
            print("Mean values:      " + str(mean))
            print("Variance values:  " + str(variance))
            print("Gaussian weights: " + str(gauss_w))
            print("---------------------------------------")

        if np.abs((llh_current - llh_prev) / llh_current) < tol:
            converged = 1
            break
    
    if converged == 0:
        warn("The EM procedure failed to converge. Try increasing `iter` and/or `tol`.", Warning)
    # Evaluate grid points in the estimated pdf
    pdf = gaussian_mixture(grid, mean, variance, gauss_w)

    return grid, pdf


def kde_convolution(x, bw, grid_edges, grid_counts, grid_len, bound_correction):
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
    bin_width = grid_edges[1] - grid_edges[0]
    f = grid_counts / bin_width / len(x) 

    # Bandwidth must consider the bin/grid width
    bw /= bin_width

    # Instantiate kernel signal. 
    # `kernel_n` works perfectly, didn't know why until:
    # Read something that said ~3 times standard deviation on each tail,
    # which is roughly similar to 2 * pi = 6.28 for two tails.
    # See: https://stackoverflow.com/questions/2773606/gaussian-filter-in-matlab
    # Makes sense since almost all density is between \pm 3 SDs
    kernel_n = int(bw * 2 * np.pi)
    kernel = gaussian(kernel_n, bw)

    if bound_correction:
        npad = int(grid_len / 5)
        f = np.concatenate([f[npad - 1:: -1], f, f[grid_len : grid_len - npad - 1: -1]])
        pdf = convolve(f, kernel, mode="same", method="direct")[npad : npad + grid_len]
        pdf /= bw * (2 * np.pi) ** 0.5
    else:
        pdf = convolve(f, kernel, mode="same", method="direct") / (bw * (2 * np.pi) ** 0.5) 
         
    grid = (grid_edges[1:] + grid_edges[:-1]) / 2 
    return grid , pdf


def kde_adaptive(x, bw, grid_edges, grid_counts, grid_len, bound_correction):
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
    pilot_grid, pilot_pdf = kde_convolution(x, bw, grid_edges, grid_counts, grid_len, bound_correction)
    
    # Adds to avoid np.log(0) and zero division
    pilot_pdf += 1e-9
    
    # Determine the modification factors
    # Geometric mean of KDE evaluated at sample points
    # EXTREMELY important to calculate geom_mean with interpolated points
    # and `adj_factor` with `pilot_pdf`.
    pdf_interp = np.interp(x, pilot_grid, pilot_pdf)
    geom_mean = np.exp(np.mean(np.log(pdf_interp)))

    # b: Compute modification factors
    # Power of c = 0.5 -> Anderson's method
    adj_factor = (geom_mean / pilot_pdf) ** 0.5
    bw_adj = bw * adj_factor

    # Estimation of Gaussian KDE via binned method (convolution not possible)
    grid = pilot_grid

    if bound_correction:
        grid_npad = int(grid_len / 5)
        grid_width = grid_edges[1] - grid_edges[0]
        grid_pad = grid_npad * grid_width
        grid_padded = np.linspace(
            grid_edges[0] - grid_pad, 
            grid_edges[grid_len - 1] + grid_pad, 
            num = grid_len + 2 * grid_npad
        )
        grid_counts = np.concatenate([
            grid_counts[grid_npad - 1:: -1],
            grid_counts, 
            grid_counts[grid_len : grid_len - grid_npad - 1: -1]]
        )
        bw_adj = np.concatenate([
            bw_adj[grid_npad - 1:: -1], 
            bw_adj, 
            bw_adj[grid_len : grid_len - grid_npad - 1: -1]]
        )
        
        pdf_mat = np.exp(-0.5 * ((grid_padded - grid_padded[:, None]) / bw_adj[:, None]) ** 2) * grid_counts[:, None]
        pdf_mat /= ((2 * np.pi) ** 0.5 * bw_adj[:, None]) 
        pdf = np.sum(pdf_mat[:, grid_npad : grid_npad + grid_len], axis=0) / len(x)

    else:
        pdf_mat = np.exp(-0.5 * ((grid - grid[:, None]) / bw_adj[:, None]) ** 2) * grid_counts[:, None]
        pdf_mat /= ((2 * np.pi) ** 0.5 * bw_adj[:, None]) 
        pdf = np.sum(pdf_mat, axis=0) / len(x)

    return grid, pdf
