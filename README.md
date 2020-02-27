# Density estimators

The aim of this repository is to recopilate, introduce, implement, and compare several density estimators.

The folder `notebooks` contain the following set of Jupyter Notebooks:

* `01_gaussian_kde`: Introduces the concept of Kerndel Density Estimator (KDE) as well as the classic bandwidth estimator for the Gaussian KDE. A naive implementation and two fast implementations are included. Finally, a time comparison is done.

* `02_boundary_issues`: Explains why the Gaussian KDE has to be modified when the domain of the variable is bounded to an interval of the real line. Introduces the *boundary reflection method* as a default alternative to treat bounded variables. Compares times between the same three implementations than in `01_gaussian_kde`.

* `03_more_bandwidth_selectors`: Discusses and introduce cases where the Gaussian rules of thumb to estimate the bandwidth for the Gaussian KDE fail. Presents some alternatives in chronological order of appearence and implements the most relevant ones. Finally a short graphical comparison between the methods is done. Not time comparison is performed because the differences are extreme and easily to note with usage.

* `04_adaptive_bandwidth_kde`: Starts with a motivational example showing why a constant bandwidth is not appropiate for some cases. Introduces two variable bandwidth density estimators, sample point KDE and an adaptive density estimator based on the EM algorithm. Implements both of the estimators and show how they work in a couple of cses.

* [TODO] `05_method_comparison`
* [TODO] `06_misc`

