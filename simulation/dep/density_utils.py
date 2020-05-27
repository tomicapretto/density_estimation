import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.signal import gaussian, convolve
from scipy.fftpack import fft, ifft
from scipy.optimize import minimize, fsolve
from scipy.integrate import quad
from fast_histogram import histogram1d
from warnings import warn

# Scott's rule
def bw_scott(x):
    a = np.std(x)
    bw = 1.06 * a * len(x) ** (-0.2)
    return bw

# Silverman's rule
def bw_silverman(x):
    a = min(np.std(x), stats.iqr(x) / 1.34)
    bw = 0.9 * a * len(x) ** (-0.2)
    return bw

# Leave-one-out Cross-Validation
def _get_ise_loocv(h, x, x_min, x_max):
    """
    Computes the Integrated Squared Error (ISE) via Leave-One-Out Cross-Validation.

    Parameters
    ----------
    x : numpy array
        1 dimensional array of sample data from the variable for which a
        density estimate is desired.
    h : float
        Bandwidth (standard deviation of each Gaussian component)
    x_min : float
        Lower limit for the domain of the variable
    x_max : float
        Upper limit for the domain of the variable

    Returns
    -------
    lscv_error : Float, estimation of the Least Squares Cross-Validation Error.
    """

    x_len = len(x)
    h = h[0]
    density_x, density_y = estimate_density(x, bw=h)
    def f_squared(y): return _density_evaluate(y, x, h) ** 2

    # Compute first term of LSCV(h)
    f_sq_twice_area = 2 * quad(f_squared, x_min, x_max)[0]

    # Compute second term of LSCV(h)
    f_loocv_sum = 0
    for i in range(x_len):
        aux_x, aux_y = estimate_density(np.delete(x, i), bw=h)
        f_loocv_sum += _density_evaluate(x[i], x, h)
    f_loocv_sum *= (2 / x_len)

    # LSCV(h)
    lscv_error = np.abs(f_sq_twice_area - f_loocv_sum)

    return lscv_error

def bw_lscv(x):
    """
    Computes Least Squares Cross-Validation bandwidth for a Gaussian KDE

    Parameters
    ----------
    x : 1-D numpy array
        1 dimensional array of sample data from the variable for which a
        density estimate is desired.

    Returns
    -------
    h : float
        Bandwidth estimated via Least Squares Cross-Validation
    """

    x_len = len(x)
    x_std = np.std(x)
    x_min = np.min(x) - 0.5 * x_std
    x_max = np.max(x) + 0.5 * x_std

    # Silverman's rule as initial value for h
    s = min(x_std, stats.iqr(x) / 1.34)
    h0 = 0.9 * s * x_len ** (-0.2)

    # h is constrained to be larger than 10**(-8)
    constraint = ({'type': 'ineq', 'fun': lambda x: x - 10 ** (-8)})
    result = minimize(_get_ise_loocv, h0, args=(x, x_min, x_max), constraints=constraint)
    h = result.x[0]

    return h

# Sheather-Jones plug-in method

def _gaussian_pdf(x):
    """
    Computes standard gaussian density function evaluated at `x`
    """
    return 0.3989422804 * np.exp(-0.5 * x ** 2)

def _density_evaluate(x, x_obs, bw):
    """
    Evaluates a Gaussian KDE in `x`
    """
    return (1. / (bw * len(x_obs))) * np.sum(_gaussian_pdf((x - x_obs) / bw), axis=0)

def _phi6(x):
    """
    Computes the 6th derivative of a standard Gaussian density function
    """
    return (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15) * _gaussian_pdf(x)

def _phi4(x):
    """
    Computes the 4th derivative of a standard Gaussian density function
    """
    return (x ** 4 - 6 * x ** 2 + 3) * _gaussian_pdf(x)

def _sj_double_sum(x, fun, den):
    """
    Auxiliary function used to compute the double sum 
    in \hat{S}_D(\alpha) and \hat{T}_D{b} in p. 689 in [1]
    
    Since a vectorized computation requires storing all parwise differences
    between the elements in `x` it becomes unfeasible as `len(x)` gets larger.
    Consequently, when `len(x)` < 1000 the `x` passed is the 2d array of 
    the pairwise differences. 
    Otherwise, `x` is the 1d array of observed values.
    If the 2d array is received, the sum is computed in a vectorized fashion.
    Otherwise, it loops through `x`.
    
    Parameters
    ----------
    x : numpy array
        1 dimensional array of sample data from the variable for which a
        density estimate is desired OR 
        2 dimensional array of pairwise difference between the 
        mentioned sample points.
    fun: function
         A function that is applied to `x / den`. 
         In this context it will be either `_phi4` or `_phi6`.
    den: float
         Can be any of the bandwidths `a` or `b` in  in p. 689 in [1]
         
    Returns
    -------
    out : float
          Value of the double sum.
    """
    
    if x.ndim == 2:
        out = np.sum(np.sum(fun(x / den), 0))
    else:
        out = 0
        for x_i in x:
            out += np.sum(fun((x_i - x) / den))
    return out

def _sj_constants(x, f1, f2, c1, c2, mult):
    """
    Auxiliary function to compute
    \hat{S}_D(a) and \hat{T}_D(b) in p. 689 in [1]
    """
    t_b = _sj_double_sum(x, f1, c1)
    t_b *= - mult * c1 ** -7
    s_a = _sj_double_sum(x, f2, c2)
    s_a *= mult * c2 ** -5
    return t_b, s_a

def _sj_optim_fun(h, s_a, t_b, x, x_len, x_len_mult):  # pylint: disable=too-many-arguments
    """
    Equation 12 of Sheather and Jones [1]

    References
    ----------
    .. [1] A reliable data-based bandwidth selection method for kernel
       density estimation. Simon J. Sheather and Michael C. Jones.
       Journal of the Royal Statistical Society, Series B. 1991
    """
    numerator = 0.375 * np.pi ** -0.5
    g_h = 1.357 * np.abs(s_a / t_b) ** (1 / 7) * h ** (5 / 7)
    
    s_g = _sj_double_sum(x, _phi4, g_h)
    s_g *= x_len_mult * g_h ** -5

    output = (numerator / np.abs(s_g * x_len)) ** 0.2 - h

    return output

def bw_sj(x):
    """
    Computes Sheather-Jones bandwidth for Gaussian KDE 
    presented in [1]

    Parameters
    ----------
    x : numpy array
        1 dimensional array of sample data from the variable for which a
        density estimate is desired.
        
    Returns
    -------
    h : float
        Bandwidth estimated via the Sheather-Jones method.
        
    References
    ----------
    .. [1] A reliable data-based bandwidth selection method for kernel
       density estimation. Simon J. Sheather and Michael C. Jones.
       Journal of the Royal Statistical Society, Series B. 1991
    """

    x_len = len(x)
    x_std = np.std(x)
    x_iqr = stats.iqr(x)
    h0 = 0.9 * x_std * x_len ** (-0.2)

    a = 0.92 * x_iqr * x_len ** (-1 / 7)
    b = 0.912 * x_iqr * x_len ** (-1 / 9)

    x_len_mult = 1 / (x_len * (x_len - 1))
    vectorized = True if x_len <= 1000 else False    
    
    if vectorized:
        x_pairwise_diff = x - x[:, None]
        s_a, t_b = _sj_constants(x_pairwise_diff, _phi6, _phi4, b, a, x_len_mult)
        result = fsolve(_sj_optim_fun, h0, args=(s_a, t_b, x_pairwise_diff, x_len, x_len_mult))
    else:
        s_a, t_b = _sj_constants(x, _phi6, _phi4, b, a, x_len_mult)
        result = fsolve(_sj_optim_fun, h0, args=(s_a, t_b, x, x_len, x_len_mult))
        
    return result[0]

# Improved Sheather-Jones plug-in method

def _dct1d(x):
    """
    Discrete Cosine Transform in 1 Dimension
    
    Parameters
    ----------
    x : numpy array
        1 dimensional array of values for which the 
        DCT is desired
        
    Returns
    -------
    output : DTC transformed values
    """

    x_len = len(x)

    even_increasing = np.arange(0, x_len, 2)
    odd_decreasing = np.arange(x_len - 1, 0, -2)

    x = np.concatenate((x[even_increasing], x[odd_decreasing]))
    
    w_1k = np.r_[1, (2 * np.exp(-(0 + 1j) * (np.arange(1, x_len)) * np.pi / (2 * x_len)))]
    output = np.real(w_1k * fft(x))
    
    return output

def _fixed_point(t, N, k_sq, a_sq):
    # This implements the function t-zeta*gamma^[l](t) in 3.23
    # To avoid prevent powers from overflowing.
    k_sq = np.asfarray(k_sq, dtype='float')
    a_sq = np.asfarray(a_sq, dtype='float')

    l = 7
    f = 0.5 * np.pi ** (2.0 * l) * np.sum(k_sq ** l * a_sq * np.exp(-k_sq * np.pi ** 2.0 * t))

    for j in reversed(range(2, l)):
        c1  = (1 + 0.5**(j + 0.5)) / 3.0
        c2  = np.product(np.arange(1., 2. * j + 1., 2., dtype = 'float')) / (np.pi / 2) ** 0.5
        t_j = np.power((c1 * c2 / (N * f)), (2 / (3 + 2 * j)))
        f   = 0.5 * np.pi ** (2. * j) * np.sum(k_sq ** j * a_sq * np.exp(-k_sq * np.pi ** 2. * t_j) )

    out = np.abs(t - (2. * N * np.pi ** 0.5 * f) ** (-0.4))
    return out


def bw_isj(x):
    """
    Improved Sheather Jones method for interactive usage
    """
    return _bw_isj(x, grid_counts=None)

def _bw_isj(x, grid_counts=None):
    """
    Improved Sheather and Jones method as explained in [1]
    This is an internal version pretended to be used by the KDE estimator
    
    References
    ----------
    .. [1] Kernel density estimation via diffusion.
       Z. I. Botev, J. F. Grotowski, and D. P. Kroese.
       Ann. Statist. 38 (2010), no. 5, 2916--2957.
    """

    x_len = len(x)
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min

    # Relative frequency per bin
    if grid_counts is None:
        x_std = np.std(x)
        grid_len = 256
        grid_min = x_min - 0.5 * x_std
        grid_max = x_max + 0.5 * x_std
        grid_counts = histogram1d(x, bins=grid_len, range=(grid_min, grid_max)) 
    else:
        grid_len = len(grid_counts) - 1
        
    grid_relfreq = grid_counts / x_len

    # Discrete cosine transform of the data
    a_k = _dct1d(grid_relfreq)

    k_sq = np.arange(1, grid_len) ** 2
    a_sq = a_k[range(1, grid_len)] ** 2

    t = fsolve(_fixed_point, 0.02, args=(x_len, k_sq, a_sq))
    h = t[0] ** 0.5 * x_range

    return h

# Experimental method
def bw_experimental(x, grid_counts=None):
    return 0.5 * (bw_silverman(x) + bw_isj(x, grid_counts))
	
_bw_methods = {
    "scott": bw_scott,
    "silverman": bw_silverman,
    "lscv": bw_lscv,
    "sj" : bw_sj,
    "isj" : _bw_isj,
    "experimental" : bw_experimental
}

def _select_bw_method(x, method="isj", grid_counts=None):
    method_lower = method.lower()

    if method_lower not in _bw_methods.keys():
        error_string = "Unrecognized bandwidth method.\n"
        error_string += f"Input is: {method}.\n"
        error_string += f"Expected one of: {list(_bw_methods.keys())}."
        raise ValueError(error_string)
    
    if method_lower in ["isj", "experimental"]:
        bw = _bw_methods[method_lower](x, grid_counts)
    else:
        bw = _bw_methods[method_lower](x)
    return bw

def get_bw(x, bw, bins=None):
    if isinstance(bw, (int, float)):
        if bw > 0:
            return bw
        else:
            error_string = "Numeric `bw` must be positive.\n"
            error_string += f"Input: {bw:.4f}."
            raise ValueError(error_string)

    elif isinstance(bw, str):
        return _select_bw_method(x, bw, grid_counts=None)
    else:
        error_string = "Unrecognized `bw` argument.\n"
        error_string += f"Input {bw} is of type {type(bw)}.\n"
        error_string += f"Expected a positive numeric or one of the following strings: {list(_bw_methods.keys())}." 
        raise ValueError(error_string)

def _check_type(x):
    
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

def _len_warning(x):
    if x < 50: 
        warn_str = f"The estimation may be unstable for such a few data points.\n"
        warn_str += f"Try a histogram or dotplot instead."
        warn(warn_str, Warning)

def kde_convolution(x, bw, grid_counts, grid_edges, grid_len, bound_correction):
    
    # Calculate relative frequencies per bin. Assumes equal width.
    grid_width = grid_edges[1] - grid_edges[0]
    f = grid_counts / grid_width / len(x) 

    # Bandwidth must consider the grid width (a.k.a. bin width)
    bw /= grid_width
    
    # Instantiate kernel signal. 50 points seems to be enough.
    kernel = gaussian(50, bw)
    
    if bound_correction:
        npad = int(grid_len / 5)
        f = np.concatenate([f[npad - 1:: -1], f, f[grid_len : grid_len - npad - 1: -1]])
        pdf = convolve(f, kernel, mode="same", method="direct")[npad : npad + grid_len]
        pdf /= np.sum(kernel)
    else:
        pdf = convolve(f, kernel, mode="same", method="direct") / np.sum(kernel)
    
    grid = (grid_edges[1:] + grid_edges[:-1]) / 2 
    return grid , pdf 


def kde_adaptive(x, bw, grid_counts, grid_edges, grid_len, bound_correction):

    # Computations for bandwidth adjustment
    pilot_grid, pilot_pdf = kde_convolution(x, bw, grid_counts, grid_edges, grid_len, bound_correction)
    
    # Determine the modification factors
    # Geometric mean of KDE evaluated at sample points
    # EXTREMELY important to calculate geom_mean with interpolated points
    # and `adj_factor` with `pilot_pdf`.
    pdf_interp = np.interp(x, pilot_grid, pilot_pdf)
    geom_mean = np.exp(np.mean(np.log(pdf_interp)))
    
    # Compute modification factors
    # Power of c = 0.5 -> Anderson's method
    # Adds 1e-6 to avoid zero division
    adj_factor = (geom_mean / (pilot_pdf + 1e-6)) ** 0.5 
    bw_adj = bw * adj_factor
    
    # Estimation of Gaussian KDE via binned method (convolution not possible)
    grid = pilot_grid
    
    if bound_correction:
        
        grid_npad = int(grid_len / 5)
        grid_width = grid[1] - grid[0]
        grid_pad = grid_npad * grid_width
        grid_padded = np.linspace(grid[0] - grid_pad, grid[grid_len - 1] + grid_pad, num = grid_len + 2 * grid_npad)
        
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
        
        pdf_mat_num = np.exp(-0.5 * ((grid_padded - grid_padded[:, None]) / bw_adj[:, None]) ** 2) * grid_counts[:, None]
        pdf_mat_den = ((2 * np.pi) ** 0.5 * bw_adj[:, None]) 
        pdf_mat = pdf_mat_num / pdf_mat_den
        pdf = np.sum(pdf_mat[:, grid_npad : grid_npad + grid_len], axis=0) / len(x)
        
    else:
        pdf_mat_num = np.exp(-0.5 * ((grid - grid[:, None]) / bw_adj[:, None]) ** 2) * grid_counts[:, None]
        pdf_mat_den = ((2 * np.pi) ** 0.5 * bw_adj[:, None]) 
        pdf_mat = pdf_mat_num / pdf_mat_den
        pdf = np.sum(pdf_mat, axis=0) / len(x)
        
    return grid, pdf

def estimate_density(
    x,
    bw="silverman",
    grid_len=256, 
    extend=True, 
    bound_correction=False, 
    adaptive=False,
    extend_fct=0.5, 
    bw_fct=1,
    bw_return=False,
    custom_lims=None
):
    
    # Check `x` is from appropiate type
    x = _check_type(x)
    
    # Preliminary calculations
    x_len = len(x)
    x_min = np.min(x)
    x_max = np.max(x)
    
    # Length warning:
    # Not completely sure whether it is necessary
    _len_warning(x_len)
    
    # Set up number of bins
    # Should I enable larger grids?
    if grid_len > 512:
        grid_len = 512
    if grid_len < 100:
        grid_len = 100
    grid_len = int(grid_len)
        
    # Set up domain
    # `custom_limits` overrides `extend`
    # `bound_correction` overrides `extend`
    if custom_lims is not None:
        assert isinstance(custom_lims, (list, tuple))
        assert len(custom_lims) == 2
        assert custom_lims[0] < custom_lims[1]
        grid_min = custom_lims[0]
        grid_max = custom_lims[1]
    elif extend and not bound_correction:
        assert isinstance(extend_fct, (int, float))
        grid_extend = extend_fct * np.std(x)
        grid_min = x_min - grid_extend
        grid_max = x_max + grid_extend
    else:
        grid_min = x_min
        grid_max = x_max
        
    grid_counts = histogram1d(x, bins=grid_len, range=(grid_min, grid_max))
    grid_edges = np.linspace(grid_min, grid_max, num=grid_len + 1)    
    
    # Bandwidth estimation
    # TODO: Take grid_counts to bw estimation. Useful in ISJ.
    assert isinstance(bw_fct, (int, float))
    bw = bw_fct * get_bw(x, bw, grid_counts)
    
    # Density estimation
    if adaptive:
        grid, pdf = kde_adaptive(x, bw, grid_counts, grid_edges, grid_len, bound_correction)
    else:
        grid, pdf = kde_convolution(x, bw, grid_counts, grid_edges, grid_len, bound_correction)
    
    if bw_return:
        return grid, pdf, bw
    else:
        return grid, pdf    
    
def _get_mixture(grid, mu, var, weight):
    out = np.sum(list((map(lambda m, v, w: _norm_pdf(grid, m, v, w), mu, var, weight))), axis=0)
    return out

def _norm_pdf(grid, mu, var, weight):
    # 1 / np.sqrt(2 * np.pi) = 0.3989423
    out = np.exp(-0.5 * ((grid - mu)) ** 2 / var) * 0.3989423 * (var ** -0.50) * weight
    return out   
	
def estimate_density_em(
    x, 
    gauss_n=None, 
    grid_len=256, 
    extend=True,
    extend_fct=0.5,
    custom_lims=None,
    iter_max=200, 
    tol=0.002, 
    verbose=False
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
    x = _check_type(x)
    
    # Preliminary calculations
    x_len = len(x)
    x_min = np.min(x)
    x_max = np.max(x)
    x_std = np.std(x)
    
    # Length warning:
    # Not completely sure whether it is necessary
    _len_warning(x_len)
    
    # Set up number of bins
    # Should I enable larger grids?
    if grid_len > 512:
        grid_len = 512
    if grid_len < 100:
        grid_len = 100
    grid_len = int(grid_len)
    
    # Set up domain
    # `custom_limits` overrides `extend`
    if custom_lims is not None:
        assert isinstance(custom_lims, (list, tuple))
        assert len(custom_lims) == 2
        assert custom_lims[0] < custom_lims[1]
        grid_min = custom_lims[0]
        grid_max = custom_lims[1]
    elif extend:
        grid_extend = extend_fct * x_std
        grid_min = x_min - grid_extend
        grid_max = x_max + grid_extend
    else:
        grid_min = x_min
        grid_max = x_max
    
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
    variance = np.full((gauss_n), (x_std ** 2) / (gauss_n * 0.5)) # heuristic
    
    llh_matrix = np.zeros((x_len, gauss_n))
    llh_current = float('-inf')
    
    for iter in range(0, iter_max):

        llh_prev = llh_current
        
    # Expectation step 
        z_sq = ((x - mean[:, None]) ** 2) / variance[:, None]
        llh_matrix = -0.5 * (np.log(2 * np.pi) + z_sq) - np.log(np.sqrt(variance[:, None])) + np.log(gauss_w[:, None])
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
            print("Step number: " + str(iter))
            print("Mean values:      " + str(mean))
            print("Variance values:  " + str(variance))
            print("Gaussian weights: " + str(gauss_w))
            print("---------------------------------------")
 
        if np.abs((llh_current - llh_prev) / llh_current) < tol:
            break

    # Evaluate grid points in the estimated pdf
    pdf = _get_mixture(grid, mean, variance, gauss_w)
        
    return grid, pdf