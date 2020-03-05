import pandas as pd
from sim_utils import print_status, simulate 

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
    "beta_1": {"params": {"a" : 2.5, "b" : 1.5}, "func_key": "beta"},
    "logn_1": {"params": {"scale" : 1}, "func_key": "logn"}
}

sizes = [200, 500, 1000, 5000, 10000]
sizes_sj = [200, 500, 1000]
niter = 300
niter_sj = 100

# Fixed gaussian ------------------------------------------------------------
estimator_name = "fixed_gaussian"
mixture=False
print(f"Simulation with estimator {estimator_name}")

# Silverman
bw_name = "silverman"
print_status(bw_name)
df1 = simulate(estimator_name, bw_name, sizes, pdf_kwargs, mixture, niter)

bw_name = "scott"
print_status(bw_name)
df2 = simulate(estimator_name, bw_name, sizes, pdf_kwargs, mixture, niter)

# SJ
bw_name = "sj"
print_status(bw_name)
df3 = simulate(estimator_name, bw_name, sizes_sj, pdf_kwargs, mixture, niter_sj)

# ISJ
bw_name = "isj"
print_status(bw_name)
df4 = simulate(estimator_name, bw_name, sizes, pdf_kwargs, mixture, niter)

# Experimental
bw_name = "experimental"
print_status(bw_name)
df5 = simulate(estimator_name, bw_name, sizes, pdf_kwargs, mixture, niter)

df_fixed_gaussian = pd.concat([df1, df2, df3, df4, df5])
df_fixed_gaussian.to_csv("output/fixed_gaussian.csv", index=False)

# Adaptive gaussian ----------------------------------------------------------
estimator_name = "adaptive_gaussian"
print("-------------------------------------------")
print(f"Simulation with estimator {estimator_name}")

# Silverman
bw_name = "silverman"
print_status(bw_name)
df1 = simulate(estimator_name, bw_name, sizes, pdf_kwargs, mixture, niter)

# Scott
bw_name = "scott"
print_status(bw_name)
df2 = simulate(estimator_name, bw_name, sizes, pdf_kwargs, mixture, niter)

# SJ
bw_name = "sj"
print_status(bw_name)
df3 = simulate(estimator_name, bw_name, sizes_sj, pdf_kwargs, mixture, niter_sj)

# ISJ
bw_name = "isj"
print_status(bw_name)
df4 = simulate(estimator_name, bw_name, sizes, pdf_kwargs, mixture, niter)

# Experimental
bw_name = "experimental"
print_status(bw_name)
df5 = simulate(estimator_name, bw_name, sizes, pdf_kwargs, mixture, niter)

df_adaptive_gaussian = pd.concat([df1, df2, df3, df4, df5])
df_adaptive_gaussian.to_csv("output/adaptive_gaussian.csv", index=False)

# Mixture estimator ----------------------------------------------------------
estimator_name = "mixture"
bw_name = "mixture"
mixture=True
print("-------------------------------------------")
print(f"Simulation with estimator {estimator_name}")

print_status(bw_name)
df_mixture = simulate(estimator_name, bw_name, sizes, pdf_kwargs, mixture, niter)
df_mixture.to_csv("output/mixture.csv", index=False)
