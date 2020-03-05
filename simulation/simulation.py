import numpy as np
import pandas as pd
from sim_utils import get_funcs, get_estimator_func, simulate_single, simulate

pdf_kwargs = {
    "gaussian_1": {"params": {"mean" : [0], "sd": [1]}, "func_key": "gaussian"},
    "gaussian_2": {"params": {"mean" : [0], "sd": [2]}, "func_key": "gaussian"},
    "gmixture_1": {"params": {"mean" : [-12, 12], "sd": [0.5, 0.5]}, "func_key": "gaussian"},
    "gmixture_2": {"params": {"mean" : [0, 5], "sd": [0.1, 1]}, "func_key": "gaussian"},
    "gmixture_3": {"params": {"mean" : [0, 0], "sd": [1, 0.1], "wt": [0.667, 0.333]}, "func_key": "gaussian"},
    "gmixture_4": {"params": {"mean" : [0, 1.5], "sd": [1, 0.33], "wt": [0.75, 0.25]}, "func_key": "gaussian"},
    "gmixture_5": {"params": {"mean" : [3.5, 9], "sd": [0.5, 1.5], "wt": [0.6, 0.4]}, "func_key": "gaussian"},
    "gamma_1": {"params": {"shape" : 1}, "func_key": "gamma"},
    "gamma_2": {"params": {"shape" : 2}, "func_key": "gamma"},
    "beta_1": {"params": {"a" : 2.4, "b" : 1.4}, "func_key": "beta"},
    "logn_1": {"params": {"scale" : 1}, "func_key": "logn"}
}

sizes = [200, 1000, 5000, 10000]
niter = 5

# Silverman bandwidth ---------------------------------------
bw_name = "silverman"

# Fixed gaussian
estimator_name = "fixed_gaussian"
estimator = get_estimator_func(bw = bw_name)
df = simulate(estimator, estimator_name, bw_name, sizes, pdf_kwargs, niter)

# Adaptive gaussian
estimator_name = "adaptive_gaussian"
estimator = get_estimator_func(bw=bw_name, adaptive=True)
df2 = simulate(estimator, estimator_name, bw_name, sizes, pdf_kwargs, niter)