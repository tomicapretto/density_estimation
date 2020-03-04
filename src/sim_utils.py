def trapezoid_sum(x, f):
    """
    Takes two arrays and computes the trapezoid sum of f(x) over the interval
    [min(x), max(x)]

    Parameters
    ----------
    x: array
       Subintevals bounds
    f: array
       The value of f(x) at each point in x

    Returns
    -------
    float
        Approximation of the integral given by the Riemann sum.
    """

    if not len(x) == len(f):
        raise ValueError("len(x) and len(f) must match")
    
    x_diff = x[1:] - x[:-1]
    f_midpoint = (f[1:] + f[:-1]) / 2

    return np.sum(f_midpoint * x_diff)

