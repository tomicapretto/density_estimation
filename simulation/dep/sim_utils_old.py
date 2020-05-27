import numpy as np
import pandas as pd
import datetime
from scipy import stats
from timeit import default_timer as timer
# Own
from density_utils import estimate_density, estimate_density_em

def print_status(bw_name):
    print(datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S') + 
      f": Simulating with `bw_name` {bw_name}")

def trapezoid_sum(x, f):
    if not len(x) == len(f):
        raise ValueError("len(x) and len(f) must match")
    
    x_diff = x[1:] - x[:-1]
    f_midpoint = (f[1:] + f[:-1]) / 2

    return np.sum(f_midpoint * x_diff)

# This function is a wrapper to select an estimator within the `simulate` function
def get_estimator_func(
    bw="silverman",
    grid_len=256, 
    extend=True, 
    bound_correction=False, 
    adaptive=False,
    extend_fct=0.5, 
    bw_fct=1,
    bw_return=False,
    custom_lims=None,
    mixture=False
):
    
    def func(x):
        if mixture:
             return estimate_density_em(x=x, grid_len=grid_len,
                                    extend=extend, extend_fct=extend_fct,
                                    custom_lims=custom_lims)
        else:
            return estimate_density(x=x, bw=bw, grid_len=grid_len, extend=extend,
                       bound_correction=bound_correction,
                       adaptive=adaptive, extend_fct=extend_fct,
                       bw_fct=bw_fct, bw_return=bw_return,
                       custom_lims=custom_lims)
    return func

# Functions to generate random values

def get_gmixture_rvs(size, mean, sd, wt = None): 
    if wt is None:
        wt = np.repeat((1 / len(mean)), len(mean))
    assert len(mean) == len(sd) == len(wt)
    assert np.sum(wt) == 1
    x = np.concatenate((
        list(map(lambda m, s, w: stats.norm.rvs(m, s, int(np.round(size * w))), mean, sd, wt))
    ))
    return x

def get_gamma_rvs(size, shape, scale=1):
    return stats.gamma.rvs(a=shape, scale=scale, size=size)

def get_logn_rvs(size, scale):
    return stats.lognorm.rvs(s=scale, size=size)

def get_beta_rvs(size, a, b):
    return stats.beta.rvs(a=b, b=b, size=size)

# Function to generate true pdfs

def get_gmixture_pdf(grid, mean, sd, wt=None):
    if wt is None:
        wt = np.repeat((1 / len(mean)), len(mean))
    assert len(mean) == len(sd) == len(wt)
    assert np.sum(wt) == 1
    pdf = np.average(list((map(lambda m, s: stats.norm.pdf(grid, m, s), mean, sd))), axis=0, weights=wt)
    return pdf

def get_gamma_pdf(grid, shape, scale=1):
    return stats.gamma.pdf(grid, a=shape, scale=scale)

def get_logn_pdf(grid, scale):
    return stats.lognorm.pdf(grid, s=scale)

def get_beta_pdf(grid, a, b):
    return stats.beta.pdf(grid, a=a, b=b)

# Store rvs as well as pdf functions in dictionaries

pdf_funcs = {
    "gaussian": get_gmixture_pdf,
    "gamma": get_gamma_pdf,
    "logn": get_logn_pdf,
    "beta": get_beta_pdf
}

rvs_funcs = {
    "gaussian": get_gmixture_rvs,
    "gamma": get_gamma_rvs,
    "logn": get_logn_rvs,
    "beta": get_beta_rvs
}

# Given an identifier and key arguments, returns rvs and pdf generators.

def get_funcs(identifier, **kwargs):
 
    def rvs_func(size):
        return rvs_funcs[identifier](size=size, **kwargs)
    
    def pdf_func(grid):
        return pdf_funcs[identifier](grid=grid, **kwargs)
    
    return rvs_func, pdf_func

# Performs simulation for a given density, estimator and sizes.

def simulate_single(rvs_func, pdf_func, pdf_name, 
                    estimator_func, estimator_name, 
                    bw_name, sizes, niter=120):
    
    colums = ["iter", "pdf", "estimator", "bw", "size", "time", "error"]
    df = pd.DataFrame(columns=colums)
    loc = 0
    
    for size in sizes:
        for i in range(niter):
            
            # Generate sample
            rvs = rvs_func(size)

            # Estimate density and measure time
            start = timer()
            grid, pdf = estimator_func(rvs)
            end = timer()
            time = end - start

            # Estimate error
            squared_diff = (pdf - pdf_func(grid)) ** 2
            ise = trapezoid_sum(grid, squared_diff)
            
            # Append to data frame
            df.loc[loc] = [i + 1, pdf_name, estimator_name, bw_name, size, time, ise]
            loc += 1
        
    return df

# This is the main function.
# Simulates observations from every density specified in `pdf_kwargs`.
# It does it for every sample size in `sizes`.
# It is estimated with an estimator given by `estimator_name` `bw_name` and `mixture`.
# When the estimator uses boundary correction or custom limits, they are passed
# within `pdf_kwargs`. It is not the most beatiful approach, but it was a last minute fix.

def simulate(estimator_name, bw_name, sizes, pdf_kwargs, mixture=False, niter=120):
    
    pdf_names = list(pdf_kwargs.keys())
    columns = ["iter", "pdf", "estimator", "bw", "size", "time", "error"]
    df = pd.DataFrame(columns=columns)
     
    if estimator_name == "fixed_gaussian":
        adaptive=False
    else:
        adaptive=True
        
    for name in pdf_names:
        params = pdf_kwargs[name]["params"]
        func_key = pdf_kwargs[name]["func_key"]
        rvs_func, pdf_func = get_funcs(func_key, **params)
        
        try:
            bound_correction = pdf_kwargs[name]["bc"]
            custom_lims = pdf_kwargs[name]["lims"]
            estimator_func = get_estimator_func(bw=bw_name, 
                                                bound_correction=bound_correction, 
                                                custom_lims=custom_lims,
                                                adaptive=adaptive,
                                                mixture=mixture)
        except KeyError:
            estimator_func = get_estimator_func(bw=bw_name, adaptive=adaptive, mixture=mixture)
        
        df2 = simulate_single(rvs_func, pdf_func, name, 
                              estimator_func, estimator_name, 
                              bw_name, sizes, niter)
        
        df = df.append(df2, ignore_index = True)
    
    return df