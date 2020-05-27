import numpy as np
import pandas as pd
from sim_utils import print_status, simulate 

pdf_kwargs = {
    "gaussian_1": {
        "distribution": ["norm"],
        "params": [{"loc" : 0, "scale": 1}], 
        "wts": None,
        "bc": False 
    },
    "gaussian_2": {
        "distribution": ["norm"],
        "params": [{"loc" : 0, "scale": 2}], 
        "wts": None,
        "bc": False
    },
    "gmixture_1": {
        "distribution": ["norm", "norm"],
        "params": [{"loc": 0, "scale": 1}, {"loc": 0, "scale": 0.1}], 
        "wts": [0.667, 0.333],
        "bc": False
    },
    "gmixture_2": {
        "distribution": ["norm", "norm"],
        "params": [{"loc": -12, "scale": 0.5}, {"loc": 12, "scale": 0.5}], 
        "wts": None,
        "bc": False
    },
    "gmixture_3": {
        "distribution": ["norm", "norm"],
        "params": [{"loc": 0, "scale": 0.1}, {"loc": 5, "scale": 1}], 
        "wts": None,
        "bc": False
    },
    "gmixture_4": {
        "distribution": ["norm", "norm"],
        "params": [{"loc": 0, "scale": 1}, {"loc": 1.5, "scale": 0.33}], 
        "wts": [0.75, 0.25],
        "bc": False
    },
    "gmixture_5": {
        "distribution": ["norm", "norm"],
        "params": [{"loc": 3.5, "scale": 0.5}, {"loc": 9, "scale": 1.5}],
        "wts": [0.6, 0.4],
        "bc": False
    },
    "gamma_1": {
        "distribution": ["gamma"], 
        "params": [{"a": 1, "scale": 1}], 
        "wts": None,
        "bc": True
    },
    "gamma_2": {
        "distribution": ["gamma"], 
        "params": [{"a": 2, "scale": 1}], 
        "wts": None,
        "bc": True
    },
    "beta_1": {
        "distribution": ["beta"], 
        "params": [{"a" : 2.5, "b" : 1.5}], 
        "wts": None,
        "bc": True
    },
    "logn_1": {
        "distribution": ["lognorm"],         
        "params": [{"s": 1, "scale": 1}], 
        "wts": None,
        "bc": True
    },
    "skwd_mixture1": {
        "distribution": ["gamma", "norm", "norm"],
        "params": [{"a": 1.5, "scale": 1}, {"loc": 5, "scale": 1}, {"loc": 8, "scale": 0.75}],
        "wts": [0.7, 0.2, 0.1],
        "bc": True
    }
}

sizes = [200, 500, 1000, 5000, 10000]
sizes_sj = [200, 500, 1000, 2000]
niter = 500
niter_sj = 300

np.random.seed(1995)

print(f"Sizes: {sizes} and {sizes_sj} for Sheather-Jones")
print(f"Number of iterations: {niter} and {niter_sj} for Sheather-Jones")
print("-------------------------------------------")

# Fixed gaussian ------------------------------------------------------------
estimator_name = "fixed_gaussian"
mixture=False
print(f"Simulation with estimator {estimator_name}")

# Silverman
bw_name = "silverman"
print_status(bw_name)
df1 = simulate(pdf_kwargs, estimator_name, bw_name, sizes, mixture, niter)

bw_name = "scott"
print_status(bw_name)
df2 = simulate(pdf_kwargs, estimator_name, bw_name, sizes, mixture, niter)

# SJ
bw_name = "sj"
print_status(bw_name)
df3 = simulate(pdf_kwargs, estimator_name, bw_name, sizes_sj, mixture, niter_sj)

# ISJ
bw_name = "isj"
print_status(bw_name)
df4 = simulate(pdf_kwargs, estimator_name, bw_name, sizes, mixture, niter)

# Experimental
bw_name = "experimental"
print_status(bw_name)
df5 = simulate(pdf_kwargs, estimator_name, bw_name, sizes, mixture, niter)

df_fixed_gaussian = pd.concat([df1, df2, df3, df4, df5])
df_fixed_gaussian.to_csv("output/fixed_gaussian.csv", index=False)

# Adaptive gaussian ----------------------------------------------------------
estimator_name = "adaptive_gaussian"
print("-------------------------------------------")
print(f"Simulation with estimator {estimator_name}")

# Silverman
bw_name = "silverman"
print_status(bw_name)
df1 = simulate(pdf_kwargs, estimator_name, bw_name, sizes, mixture, niter)

# Scott
bw_name = "scott"
print_status(bw_name)
df2 = simulate(pdf_kwargs, estimator_name, bw_name, sizes, mixture, niter)

# SJ
bw_name = "sj"
print_status(bw_name)
df3 = simulate(pdf_kwargs, estimator_name, bw_name, sizes_sj, mixture, niter_sj)

# ISJ
bw_name = "isj"
print_status(bw_name)
df4 = simulate(pdf_kwargs, estimator_name, bw_name, sizes, mixture, niter)

# Experimental
bw_name = "experimental"
print_status(bw_name)
df5 = simulate(pdf_kwargs, estimator_name, bw_name, sizes, mixture, niter)

df_adaptive_gaussian = pd.concat([df1, df2, df3, df4, df5])
df_adaptive_gaussian.to_csv("output/adaptive_gaussian.csv", index=False)

# Mixture estimator ----------------------------------------------------------
estimator_name = "mixture"
bw_name = "mixture"
mixture=True
print("-------------------------------------------")
print(f"Simulation with estimator {estimator_name}")

print_status(bw_name)
df_mixture = simulate(pdf_kwargs, estimator_name, bw_name, sizes, mixture, niter)
df_mixture.to_csv("output/mixture.csv", index=False)
