# Density estimators

The aim of this repository is to recopilate, introduce, implement, and compare several density estimators.

## Notebooks

The folder `notebooks` contain the following set of Jupyter Notebooks:

* `01_gaussian_kde`: Introduces the concept of Kerndel Density Estimator (KDE) as well as the classic bandwidth estimator for the Gaussian KDE. A naive implementation and two fast implementations are included. Finally, a time comparison is done.

* `02_boundary_issues`: Explains why the Gaussian KDE has to be modified when the domain of the variable is bounded to an interval of the real line. Introduces the *boundary reflection method* as a default alternative to treat bounded variables. Compares times between the same three implementations than in `01_gaussian_kde`.

* `03_more_bandwidth_selectors`: Discusses and introduce cases where the Gaussian rules of thumb to estimate the bandwidth for the Gaussian KDE fail. Presents some alternatives in chronological order of appearence and implements the most relevant ones. Finally a short graphical comparison between the methods is done. No time comparison is performed because the differences are extreme and easily to note with usage.

* `04_adaptive_bandwidth_kde`: Starts with a motivational example showing why a constant bandwidth is not appropiate for some cases. Introduces two variable bandwidth density estimators, sample point KDE and an adaptive density estimator based on the EM algorithm. Implements both of the estimators and show how they work in a couple of cases.

* [TODO] `05_method_comparison`: Here I am going to explain what methods I am going to compare (estimators, distributions, sample sizes, etc.)

* [TODO] `06_misc`: I still don't know what is going to be here.

## Simulation

The folder `simulation` contains the programs used to carry out simulations to compare different density estimators under different circumstances in terms of error and time. 

* `density_utils.py`: Contains all the required functions to perform the density estimation (both bandwidth and density estimators).

* `sim_utils.py`: Contains all the required functions to carry out the simulation. There are functions to generate random values, generate true density functions, and some wrappers that given some parameters perform the entire simulation and return a pandas data frame with the results.

* `simulation.py`: Script where the simulation is setted up. It determines the probability distributions and its parameters, the sample sizes, and the location of the output, among others.

* `run.ipynb`: Simply a notebook that runs `simulation.py`.

* `output/*.csv`: The result of the simulations for each density estimator. They contain the following fields:
  + iter: The iteration number.
  + pdf: An identifier of the probability distribution from which values where simulated.
  + estimator: The name of the density estimator used. Same as file name.
  + bw: The name of the bandwidth estimator.
  + size: Sample size.
  + time: The time it took to compute the estimation, in seconds.
  + error: The difference between the true pdf and the estimated pdf in terms of the Integrated Squared Error.

## Shiny explorer application

It would have been cumbersome to generate graphics for each possible combination of the results in the simulation. That's why a [Shiny](https://shiny.rstudio.com/) application has been created to create visualizations interactively. It lives under `simulation/R/app`. 

It is also possible, and simple, to run the application locally with 

```
# install.packages("shiny")
shiny::runGitHub("density_estimation", username = "tomicapretto", subdir = "R/app/")
```

A good place to start once you're running the application is the **About** tab.
