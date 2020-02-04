import numpy as np
from scipy import stats
from scipy.signal import gaussian, convolve
from scipy.fftpack import fft, ifft # faster than np.fft.fft and np.fft.ifft

def gaussian_kde(x, h=None, grid_len=500, extend=True):
    
    """
    Naive, inefficient, but straightforward Gaussian KDE
    
    Parameters
    ----------
    x : array-like
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
    
    x_std = np.std(x)
    x_len = len(x)
        
    if extend:
        grid_min = np.min(x) - x_std
        grid_max = np.max(x) + x_std
    else:
        grid_min = x_min
        grid_max = x_max
    
    grid = np.linspace(grid_min, grid_max, num=grid_len)
    
    pdf_mat = np.zeros((x_len, grid_len))
    
    if h is None:
        s = min(x_std, stats.iqr(x) / 1.34)
        h = 0.9 * s * x_len ** (-0.2)    
    
    for i in range(0, x_len):
        mu = x[i]
        pdf_mat[i, :] = np.exp(-0.5 * ((grid - mu) / h) ** 2) / (np.sqrt(2 * np.pi) * h)
     
    pdf = np.mean(pdf_mat, axis=0)
    
    return grid, pdf   

def convolution_kde(x, h=None, grid_len=256, extend=True):
    
    """
    Gaussian KDE via convolution of empirical density with Gaussian signal
    
    Parameters
    ----------
    x : array-like
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
        grid_min = x_min - x_std
        grid_max = x_max + x_std
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
    # alternative
    # grid = 0.5 * (edges[1:] + edges[:-1]) 
    
    return grid, pdf

# -------------------------------------------------------------------------------------
def _dct1d(x):
    
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

#     x = np.asfarray(x, dtype='float')
    x_len = len(x)

    even_increasing = np.arange(0, x_len, 2)
    odd_decreasing = np.arange(x_len - 1, 0, -2)

    x = np.concatenate((x[even_increasing], x[odd_decreasing]))
    
    w_1k = np.r_[1, (2 * np.exp(-(0 + 1j) * (np.arange(1, x_len)) * np.pi / (2 * x_len)))]
    output = np.real(w_1k * fft(x))
    
    return output

# -------------------------------------------------------------------------------------
def _idct1d(x):
    
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

#     x = np.asfarray(x, dtype='float')
    x_len = len(x)

    w_2k = x * np.exp((0 + 1j) * np.arange(0, x_len) * np.pi / (2 * x_len))
    x = np.real(ifft(w_2k))

    output = np.zeros(x_len)
    output[np.arange(0, x_len, 2, dtype=int)] = x[np.arange(0, x_len / 2, dtype=int)]
    output[np.arange(1, x_len, 2, dtype=int)] = x[np.arange(x_len - 1, (x_len / 2) - 1, -1, dtype=int)]

    return output

# --------------------------------------------------------------------------------------------------------
def theta_kde(x, h=None, grid_len=None, extend=True):
    
    """
    Approximation to Gaussian KDE via Theta kernel
    
    Parameters
    ----------
    x : array-like
        1 dimensional array of the observed data for which a 
        density estimate is desired.
    h : float, optional
        Bandwidth (standard deviation of each Gaussian component)
        Defaults to None, which uses gaussian rule of thumb.
    grid_len : int, optional
        Number of points where the kernel is evaluated. 
        Defaults to None, which means 256 grid points.
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
    if grid_len is None:
        grid_len = 256
    elif grid_len > 512:
        grid_len = 512
    else:
        grid_len = 2 ** np.ceil(np.log2(grid_len))
    
    # Set up grid length
    if extend:
        grid_min = x_min - 1 * x_std
        grid_max = x_max + 1 * x_std
    else:
        grid_min = x_min
        grid_max = x_max
          
    # Relative frequency per bin
    f, edges = np.histogram(x, bins=grid_len, range=(grid_min, grid_max), density=True)

    # Discrete cosine transform of the data
    a_k = _dct1d(f)

    # Bandwidth selection
    if h is None:
        s = min(x_std, stats.iqr(x) / 1.34)
        t = 1.12 * s ** 2 * x_len ** (-0.4)
    else:
        t = float(h) ** 2
    
    t = t / x_range ** 2

    # Smooth values obtained with the DCT
    a_k = a_k * np.exp(-np.arange(0, grid_len) ** 2 * np.pi ** 2 * t * 0.5)

    # Inverse discrete cosine transform
    density = _idct1d(a_k)

    grid = np.linspace(grid_min, grid_max, num=grid_len)
    # alternative
    # grid = 0.5 * (edges[1:] + edges[:-1]) 

    return grid, density