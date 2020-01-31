# Density estimators

The aim of this repository is to recopilate, introduce, implement, and compare several density estimators.

The folder `notebooks` contain the following set of Jupyter Notebooks:

* `01_gaussian_kde`: Introduces the concept of Kerndel Density Estimator (KDE) as well as the classic bandwidth estimator for the Gaussian KDE. A naive implementation and two fast implementations are included. Finally, a time comparison is done.

* `02_boundary_issues`: Explains why the Gaussian KDE has to be modified when the domain of the variable is bounded to an interval of the real line. Introduces the *boundary reflection method* as a default alternative to treat bounded variables. Compares times between the same three implementations than in `01_gaussian_kde`.

* TODO: `03_more_bandwidth_selectors`
* TODO: `04_adaptive_bandwidth_kde`
* TODO: `05_method_comparison`
* TODO: `06_misc`

