import numpy as np
from scipy import stats
from scipy.signal import gaussian, convolve
from scipy.fftpack import fft, ifft

def gaussian_kde(x, h=None, grid_len=500, extend=True):
    """
    Naive, inefficient, but straightforward Gaussian KDE

    Parameters
    ----------
    x : numpy array
        1 dimensional array of sample data from the variable for which a 
        density estimate is desired.
    h : float, optional
        Bandwidth (standard deviation of each Gaussian component)
        Defaults to None, which uses Gaussian robust rule of thumb.
    grid_len : int, optional
        Number of points where the kernel is evaluated. 
        Defaults to 500.
    extend: boolean, optional
        Whether to extend the domain of the observed data or not. 
        Defaults to True.

    Returns
    -------
    grid : Gridded numpy array for the x values.
    pdf : Numpy array for the density estimates.

    """

    x_std = x.std()
    x_len = len(x)

    if extend:
        grid_min = np.min(x) - 0.5 * x_std
        grid_max = np.max(x) + 0.5 * x_std
    else:
        grid_min = x_min
        grid_max = x_max

    grid = np.linspace(grid_min, grid_max, num=grid_len)

    if h is None:
        s = min(x_std, stats.iqr(x) / 1.34)
        h = 0.9 * s * x_len ** (-0.2)

    pdf_mat = np.exp(-0.5 * ((grid - x[:, None]) / h) ** 2) / ((2 * np.pi) ** 0.5 * h)
    pdf = np.mean(pdf_mat, axis=0)

    return grid, pdf 

def gaussian_kde_binned(x, h=None, grid_len=256, extend=True):
    
    # Calculate preliminary values
    x_len = len(x)
    x_max = np.max(x)
    x_min = np.min(x)
    x_std = np.std(x)
    
    if grid_len > 500:
        grid_len = 500
    
    grid_len = int(grid_len)
    
    # Set up grid length
    if extend:
        grid_min = x_min - 0.5 * x_std
        grid_max = x_max + 0.5 * x_std
    else:
        grid_min = x_min
        grid_max = x_max
       
    # Calculate relative frequencies per bin
    grid_counts, grid = np.histogram(x, bins=grid_len, range=(grid_min, grid_max))
    grid = (grid[1:]  + grid[:-1]) / 2
    
    if h is None:
        s = min(x_std, stats.iqr(x) / 1.34)
        h = 0.9 * s * x_len ** (-0.2) 

    pdf_mat = np.exp(-0.5 * ((grid - grid[:, None]) / h) ** 2) / ((2 * np.pi) ** 0.5 * h) * grid_counts[:, None]
    pdf = np.sum(pdf_mat, axis=0) / x_len

    return grid, pdf

def convolution_kde(x, h=None, grid_len=256, extend=True):
    
    """
    Gaussian KDE via convolution of empirical density with Gaussian signal
    
    Parameters
    ----------
    x : numpy array
        1 dimensional array of sample data from the variable for which a 
        density estimate is desired.
    h : float, optional
        Bandwidth (standard deviation of each Gaussian component)
        Defaults to None, which uses Gaussian rule of thumb.
    grid_len : int, optional
        Number of points where the kernel is evaluated. 
        Defaults to 256 grid points.
    extend: boolean, optional
        Whether to extend the domain of the observed data
        or not. Defaults to True.
    
    Returns
    -------
    grid : Gridded numpy array for the x values.
    pdf : Numpy array for the density estimates.
    
    """
    
    # Calculate preliminary values
    x_len = len(x)
    x_max = np.max(x)
    x_min = np.min(x)
    x_range = x_max - x_min
    x_std = np.std(x)
    
    # Set up number of bins
    if grid_len > 512:
        grid_len = 512
    else:
        grid_len = 2 ** np.ceil(np.log2(grid_len))
    
    grid_len = int(grid_len)
    
    # Set up grid length
    if extend:
        grid_min = x_min - 0.5 * x_std
        grid_max = x_max + 0.5 * x_std
    else:
        grid_min = x_min
        grid_max = x_max
       
    # Calculate relative frequencies per bin
    f, edges = np.histogram(x, bins=grid_len, range=(grid_min, grid_max), density=True)
    
    # Bin width
    bin_width = (grid_max - grid_min) / (grid_len - 1)
    
    if h is None:
        x_std = x_std
        s = min(x_std, stats.iqr(x) / 1.34)
        h = 0.9 * s * x_len ** (-0.2) 
        
    # Bandwidth must consider the bin width
    h /= bin_width
    
    kernel = gaussian(120, h)
    pdf = convolve(f, kernel, mode="same", method="direct") / sum(kernel) # "direct" better than "fft" for n < ~ 500.
    
    grid = np.linspace(grid_min, grid_max, num=grid_len)

    return grid, pdf

def dct1d(x):
    
    """
    Discrete Cosine Transform in 1 Dimension
    
    Parameters
    ----------
    x : array-like
        1 dimensional array of values for which the 
        DCT is desired
        
    Returns
    -------
    output : Transformed values
    """

    x_len = len(x)
    even_increasing = np.arange(0, x_len, 2)
    odd_decreasing = np.arange(x_len - 1, 0, -2)

    x = np.concatenate((x[even_increasing], x[odd_decreasing]))
    
    w_1k = np.r_[1, (2 * np.exp(-(0 + 1j) * (np.arange(1, x_len)) * np.pi / (2 * x_len)))]
    output = np.real(w_1k * fft(x))
    
    return output

def idct1d(x):
    
    """
    Inverse Discrete Cosine Transform in 1 Dimension
    
    Parameters
    ----------
    x : array-like
        1 dimensional array of values for which the 
        IDCT is desired
        
    Returns
    -------
    output : Transformed values
    """
    
    x_len = len(x)

    w_2k = x * np.exp((0 + 1j) * np.arange(0, x_len) * np.pi / (2 * x_len))
    x = np.real(ifft(w_2k))

    output = np.zeros(x_len)
    output[np.arange(0, x_len, 2, dtype=int)] = x[np.arange(0, x_len / 2, dtype=int)]
    output[np.arange(1, x_len, 2, dtype=int)] = x[np.arange(x_len - 1, (x_len / 2) - 1, -1, dtype=int)]

    return output

# -------------------------------------------------------------------------------------
def theta_kde(x, h=None, grid_len=256, extend=True):
    
    """
    Approximation to Gaussian KDE via Theta kernel
    
    Parameters
    ----------
    x : numpy array
        1 dimensional array of the observed data for which a 
        density estimate is desired.
    h : float, optional
        Bandwidth (standard deviation of each Gaussian component)
        Defaults to None, which uses gaussian rule of thumb.
    grid_len : int, optional
        Number of points where the kernel is evaluated. 
        Defaults to 256 grid points.
    extend: bool, optional
        Whether to extend the domain of the observed data
        or not. Defaults to True.
    
    Returns
    -------
    grid : Gridded numpy array for the x values.
    pdf : Numpy array for the density estimates.
    
    """

    # Calculate preliminary values
    x_len = len(x)
    x_max = np.max(x)
    x_min = np.min(x)
    x_range = x_max - x_min
    x_std = np.std(x)
    
    # Set up number of bins
    if grid_len > 512:
        grid_len = 512
    else:
        grid_len = 2 ** np.ceil(np.log2(grid_len))
    
    grid_len = int(grid_len)
    
    # Set up grid range
    if extend:
        grid_min = x_min - 0.5 * x_std
        grid_max = x_max + 0.5 * x_std
    else:
        grid_min = x_min
        grid_max = x_max
          
    # Relative frequency per bin
    f, edges = np.histogram(x, bins=grid_len, range=(grid_min, grid_max), density=True)

    # Discrete cosine transform of the data
    a_k = dct1d(f)

    # Bandwidth selection
    if h is None:
        s = min(x_std, stats.iqr(x) / 1.34)
        t = 1.12 * s ** 2 * x_len ** (-0.4) # 1.12 could be replaced with 0.9^2 according to Silverman.
    else:
        t = float(h) ** 2
    
    t = t / x_range ** 2

    # Smooth values obtained with the DCT
    a_k = a_k * np.exp(-np.arange(0, grid_len) ** 2 * np.pi ** 2 * t * 0.5)

    # Inverse discrete cosine transform
    density = idct1d(a_k)

    grid = np.linspace(grid_min, grid_max, num=grid_len)
    return grid, density

# Bandwidth selectors ==========================================================
from scipy.optimize import fsolve

# Improved sheather jones
def _fixed_point(t, N, k_sq, a_sq):

    # To avoid prevent powers from overflowing.
    k_sq = np.asfarray(k_sq, dtype='float')
    a_sq = np.asfarray(a_sq, dtype='float')

    l = 7
    f = 0.5 * np.pi ** (2.0 * l) * sum(k_sq ** l * a_sq * np.exp(-k_sq * np.pi ** 2.0 * t))

    for j in reversed(range(2, l)):
        c1  = (1 + 0.5**(j + 0.5)) / 3.0
        c2  = np.product(np.arange(1., 2. * j + 1., 2., dtype = 'float')) / np.sqrt(np.pi / 2)
        t_j = np.power((c1 * c2 / (N * f)), (2 / (3 + 2 * j)))
        f   = 0.5 * np.pi ** (2. * j) * sum(k_sq ** j * a_sq * np.exp(-k_sq * np.pi ** 2. * t_j) )

    out = t - (2. * N * np.sqrt(np.pi) * f) ** (-0.4)
    return out

def h_isj(x, grid_len = 256):
    
    x_len = len(x)
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    x_std = np.std(x)
    
    grid_min = x_min - 0.5 * x_std
    grid_max = x_max + 0.5 * x_std
       
    # Relative frequency per bin
    f, edges = np.histogram(x, bins=grid_len, range=(grid_min, grid_max))
    f = f / x_len

    # Discrete cosine transform of the data
    a_k = dct1d(f)
    
    k_sq = np.arange(1, grid_len) ** 2
    a_sq = a_k[range(1, grid_len)] ** 2
    
    t = fsolve(_fixed_point, 0.02, args=(x_len, k_sq, a_sq))
    h = np.sqrt(t[0]) * x_range
    
    return h

# Silverman's rule of thumb
def h_sil(x):
    h = 0.9 * min(np.std(x), stats.iqr(x) / 1.34) * len(x) ** (-0.2)
    return h
	
from scipy.optimize import fsolve

# The n-th derivative of a gaussian pdf is given by the pdf times 
# the n-th probabilists' Hermite polynomial.
# ref: https://math.stackexchange.com/questions/2287895/are-derivatives-of-the-standard-gaussian-density-bounded
def phi6(x):
    return (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15) * stats.norm.pdf(x)

def phi4(x):
    return (x ** 4 - 6 * x ** 2 + 3) * stats.norm.pdf(x)

def sj_helper(h, s_a, t_b, x_len, x_len_mult, x_pairwise_diff):
    """
    Equation 12 of Sheather and Jones [1]
    Equation 3.9 in this work.
    
    References
    ----------
    .. [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991
    """
    
    numerator = 0.375 * np.pi ** -0.5  
    g_h = 1.357 * np.abs(s_a / t_b) ** (1 / 7) * h ** (5 / 7)
    s_g = np.sum(np.sum(phi4(x_pairwise_diff / g_h), 0))
    s_g *= x_len_mult * g_h ** -5
    
    output = (numerator / np.abs(s_g * x_len)) ** 0.2 - h
    
    return output

def h_sj(x):
    """
    Computes Sheather-Jones bandwidth for Gaussian KDE
    
    Parameters
    ----------
    x : numpy array
        1 dimensional array of sample data from the variable for which a 
        density estimate is desired.  
    Returns
    -------
    h : float
        Bandwidth estimated via Least Squares Cross-Validation
    """
    
    x_len = len(x)
    x_std = np.std(x)
    x_iqr = stats.iqr(x)
    
    a = 0.92 * x_iqr * x_len ** (-1 / 7)
    b = 0.912 * x_iqr * x_len ** (-1 / 9) 
    
    x_len_mult = 1 / (x_len * (x_len - 1))
    x_matrix = np.tile(x, (x_len, 1))
    x_pairwise_diff = x_matrix - x_matrix.T

    t_b = np.sum(np.sum(phi6(x_pairwise_diff / b), 0))
    t_b *= - x_len_mult * b ** -7

    s_a = np.sum(np.sum(phi4(x_pairwise_diff / a), 0))
    s_a *= x_len_mult * a ** -5
    
    # Silverman's rule as initial value for h
    h0 = 0.9 * x_std * x_len ** (-0.2)
    
    result = fsolve(sj_helper, h0, args=(s_a, t_b, x_len, x_len_mult, x_pairwise_diff))
    return result[0]