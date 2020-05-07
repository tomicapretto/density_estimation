"""
Functions to compute the bandwidth, including auxiliary functions
"""

import numpy as np
from scipy import stats
from scipy.fftpack import fft, ifft
from scipy.optimize import minimize, fsolve
from scipy.integrate import quad

# Own
# Circular dependcy... how to avoid?
import src.density as own

__all__ = ["bw_scott", "bw_silverman", "bw_lscv", "bw_sj", "bw_isj", "bw_experimental"]


def bw_scott(x):
    """
    Scott Rule
    """
    a = min(np.std(x), stats.iqr(x) / 1.34)
    bw = 1.06 * a * len(x) ** (-0.2)
    return bw


def bw_silverman(x):
    """
    Silverman Rule
    """
    a = min(np.std(x), stats.iqr(x) / 1.34)
    bw = 0.9 * a * len(x) ** (-0.2)
    return bw


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
    density_x, density_y = own.estimate_density(x, bw=h)
    def f_squared(y): return _density_evaluate(y, x, h) ** 2

    # Compute first term of LSCV(h)
    f_sq_twice_area = 2 * quad(f_squared, x_min, x_max)[0]

    # Compute second term of LSCV(h)
    f_loocv_sum = 0
    for i in range(x_len):
        aux_x, aux_y = own.estimate_density(np.delete(x, i), bw=h)
        f_loocv_sum += _density_evaluate(x[i], x, h)
    f_loocv_sum *= (2 / x_len)

    # LSCV(h)
    lscv_error = np.abs(f_sq_twice_area - f_loocv_sum)

    return lscv_error

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


def bw_isj(x):  # pylint: disable= too-many-locals
    """
    Improved Sheather and Jones method as explained in [1]
    
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
    x_std = np.std(x)

    grid_len = 256
    grid_min = x_min - 0.5 * x_std
    grid_max = x_max + 0.5 * x_std

    # Relative frequency per bin
    f, _ = np.histogram(x, bins=grid_len, range=(grid_min, grid_max))
    f = f / x_len

    # Discrete cosine transform of the data
    a_k = _dct1d(f)

    k_sq = np.arange(1, grid_len) ** 2
    a_sq = a_k[range(1, grid_len)] ** 2

    t = fsolve(_fixed_point, 0.02, args=(x_len, k_sq, a_sq))
    h = t[0] ** 0.5 * x_range

    return h


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
    """
    Implementation of the function t-zeta*gamma^[l](t) derived
    from equation (30) in [1]
    
    References
    ----------
    .. [1] Kernel density estimation via diffusion.
       Z. I. Botev, J. F. Grotowski, and D. P. Kroese.
       Ann. Statist. 38 (2010), no. 5, 2916--2957.
    """
    # To avoid prevent powers from overflowing.
    k_sq = np.asfarray(k_sq, dtype="float")
    a_sq = np.asfarray(a_sq, dtype="float")

    l = 7
    f = 0.5 * np.pi ** (2.0 * l) * np.sum(k_sq ** l * a_sq * np.exp(-k_sq * np.pi ** 2.0 * t))

    for j in reversed(range(2, l)):
        c1 = (1 + 0.5 ** (j + 0.5)) / 3.0
        c2 = np.product(np.arange(1.0, 2.0 * j + 1.0, 2.0, dtype="float")) / (np.pi / 2) ** 0.5
        t_j = np.power((c1 * c2 / (N * f)), (2 / (3 + 2 * j)))
        f = 0.5 * np.pi ** (2.0 * j) * np.sum(k_sq ** j * a_sq * np.exp(-k_sq * np.pi ** 2.0 * t_j))

    out = np.abs(t - (2.0 * N * np.pi ** 0.5 * f) ** (-0.4))
    return out


# unsed function?
def _idct1d(x):
    """
    Inverse Discrete Cosine Transform in 1 dimension

    Parameters
    ----------
    x : numpy array
        1 dimensional array of values for which the
        IDCT is desired

    Returns
    -------
    output : IDCT transformed values
    """

    x_len = len(x)

    w_2k = x * np.exp((0 + 1j) * np.arange(0, x_len) * np.pi / (2 * x_len))
    x = np.real(ifft(w_2k))

    output = np.zeros(x_len)
    output[np.arange(0, x_len, 2, dtype=int)] = x[np.arange(0, x_len / 2, dtype=int)]
    output[np.arange(1, x_len, 2, dtype=int)] = x[np.arange(x_len - 1, (x_len / 2) - 1, -1, dtype=int)]

    return output


def bw_experimental(x):
    """
    Experimental bandwidth estimator.
    """
    return 0.5 * (bw_silverman(x) + bw_isj(x))
