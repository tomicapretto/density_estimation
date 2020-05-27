import numpy as np
import pandas as pd
import datetime
from scipy import stats
from timeit import default_timer as timer

import sys
sys.path.append("..") 
from src import estimate_density, estimate_density_em

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

distributions_dict = {
    "norm": stats.norm,
    "gamma": stats.gamma,
    "lognorm": stats.lognorm,
    "beta": stats.beta
}

def component_rvs(distribution, params, size):
    f = distributions_dict[distribution](**params)
    return f.rvs(size)

def mixture_rvs(distributions, params, size, wts=None):
    if wts is None:
        wts = np.repeat(1 / len(distributions), len(distributions))
    assert len(distributions) == len(params) == len(wts)
    assert np.allclose(np.sum(wts), 1)
    sizes = np.round(np.array(wts) * size).astype(int)
    map_obj = map(lambda d, p, s: component_rvs(d, p, s), distributions, params, sizes)
    return np.concatenate(list(map_obj))

def distribution_bounds(distribution, params):
    bounds = {
    "norm" : lambda p: [p["loc"] - 3.5 * p["scale"], p["loc"] + 3.5 * p["scale"]],
    "gamma": lambda p: [0, stats.gamma.ppf(0.995, a=p["a"], scale=p["scale"])],
    "lognorm": lambda p: [0, stats.lognorm.ppf(0.995, s=p["s"], scale=p["scale"])],
    "beta": lambda p: [0, 1]}
    
    return bounds[distribution](params)

def mixture_bounds(distributions, params):
    map_obj = map(lambda d, p: distribution_bounds(d, p), distributions, params)
    vals = np.concatenate(list(map_obj))
    return [np.min(vals), np.max(vals)]

def mixture_grid(distributions, params):
    bounds = mixture_bounds(distributions, params)
    return np.linspace(bounds[0], bounds[1], num=500)

def component_pdf(distribution, params, grid):
    f = distributions_dict[distribution](**params)
    return f.pdf(grid)

def mixture_pdf(distributions, params, wts = None, grid = None, return_grid=False):
    if wts is None:
        wts = np.repeat(1 / len(distributions), len(distributions))
    assert len(distributions) == len(params) == len(wts)
    assert np.allclose(np.sum(wts), 1)
    
    if grid is None:
        grid = mixture_grid(distributions, params)
    
    pdf = np.average(list((map(lambda d, p: component_pdf(d, p, grid), distributions, params))), 
                     axis=0, weights=wts)
    if return_grid:
        return grid, pdf
    else:
        return pdf

# Performs simulation for a given density, estimator and sizes.
def simulate_single(pdf_kwargs, estimator_func, names, sizes, niter=120):
    
    distribution = pdf_kwargs["distribution"]
    params = pdf_kwargs["params"]
    wts = pdf_kwargs["wts"]
    
    time_list, ise_list = [], []
    
    for size in sizes:
        print(datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S') + ": Size " + str(size))
        for i in range(niter):
            # Generate sample
            rvs = mixture_rvs(distribution, params, size, wts)

            # Estimate density and measure time
            start = timer()
            grid, pdf = estimator_func(rvs)
            end = timer()
            time = end - start
            
            # Estimate error
            pdf_true = mixture_pdf(distribution, params, wts, grid)
            squared_diff = (pdf - pdf_true) ** 2
            ise = trapezoid_sum(grid, squared_diff)
            
            # Append elements to lists
            time_list.append(time)
            ise_list.append(ise)
    
    n_rows = int(len(sizes) * niter)
    
    df_dict = {
        "iter": [i + 1 for i in range(n_rows)],
        "pdf": [names["pdf"]] * n_rows,
        "estimator": [names["estimator"]] * n_rows,
        "bw": [names["bw"]] * n_rows,
        "size": [s for s in sizes for iter in range(niter)],
        "time": time_list,
        "error": ise_list
    }
    df = pd.DataFrame.from_dict(df_dict)
    return df

# This is the main function.
# Simulates observations from every density specified in `pdf_kwargs`.
# It does it for every sample size in `sizes`.
# It is estimated with an estimator given by `estimator_name` `bw_name` and `mixture`.
# When the estimator uses boundary correction or custom limits, they are passed
# within `pdf_kwargs`. It is not the most beatiful approach, but it was a last minute fix.

def simulate(pdf_kwargs, estimator_name, bw_name, sizes, mixture=False, niter=120):
    
    columns = ["iter", "pdf", "estimator", "bw", "size", "time", "error"]
    df = pd.DataFrame(columns=columns)
    
    pdf_names = list(pdf_kwargs.keys())
    adaptive = True if estimator_name == "adaptive_gaussian" else False
    
    for pdf_name in pdf_names:
        names = {"pdf": pdf_name, "estimator": estimator_name, "bw": bw_name}
        distribution = pdf_kwargs[pdf_name]["distribution"]
        params = pdf_kwargs[pdf_name]["params"]
        custom_lims = mixture_bounds(distribution, params)
        
        estimator_func = get_estimator_func(
                bw=bw_name, 
                bound_correction=pdf_kwargs[pdf_name]["bc"], 
                custom_lims=custom_lims,
                adaptive=adaptive,
                mixture=mixture)

        df2 = simulate_single(pdf_kwargs[pdf_name], estimator_func, names, sizes, niter)
        df = df.append(df2, ignore_index = True)
        print(datetime.datetime.now().strftime('%m-%d-%y %H:%M:%S') + ": Finshed " + pdf_name)
    
    return df


# Not simulation functions but used to generate plots
import matplotlib.pyplot as plt

def plot_single(pdf_kwargs, adaptive, bw, size, mixture=False):
    
    pdf_names = list(pdf_kwargs.keys())
    idx = 0
    fig, axes = plt.subplots(6, 2)
    
    for row in range(6):
        for col in range(2):
            pdf_name = pdf_names[idx]
            idx += 1
            
            distribution = pdf_kwargs[pdf_name]["distribution"]
            params = pdf_kwargs[pdf_name]["params"]
            wts = pdf_kwargs[pdf_name]["wts"]

            # Generate sample
            rvs = mixture_rvs(distribution, params, size, wts)

            # Get estimation function  
            custom_lims = mixture_bounds(distribution, params)
            estimator_func = get_estimator_func(bw=bw, bound_correction=pdf_kwargs[pdf_name]["bc"], 
                                              custom_lims=custom_lims, adaptive=adaptive, mixture=mixture)
            # Estimate density          
            grid, pdf = estimator_func(rvs)

            # Get true density function
            pdf_true = mixture_pdf(distribution, params, wts, grid)

            axes[row, col].hist(rvs, bins=50, density=True)
            axes[row, col].set_xlim([min(grid), max(grid)])
            axes[row, col].plot(grid, pdf_true, ls="--", lw=3, color="black")
            axes[row, col].plot(grid, pdf, lw=3, color="#c0392b")
            axes[row, col].set_title(pdf_name)
            

  
