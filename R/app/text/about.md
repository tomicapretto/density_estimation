## Quick overview

This shiny application is a compliment of the work presented in **[todo]**. 
The objective of this app is to enable the interactive exploration of simulation results 
as well as testing different density estimation methods with samples 
coming from a wide range of probability distributions.

### Boxplots

#### Objective

Visualize and compare the performance of the different estimators under 
the different scenarios  in terms of computational time and 
estimation error measured via the Integrated Squared Error.

#### How to use

Set the options on the left panel and see the results after clicking on the **Plot** button. 

1. Choose the **Metric** you want to visualize.
1. Specify one or more probability distributions from which random samples were
simulated and used to estimate the underlying probability density function
in the **Density** field.
1. Select one or more **Estimators(s)**. There are three available options.
The classic Gaussian KDE, an adaptive Gaussian KDE and a method 
that fits a Gaussian Mixture via the EM algorithm.
1. Select one or more **Bandwidth(s)** estimation methods. 
Note the Gaussian mixture via EM only uses the method we decided to call "Mixture".
1. By default all the sample sizes used in the simulation are selected. 
**Sample size(s)** lets you select a subset of these.
1. Since some distributions tend to be highly right-skewed, it is possible to trim
this tail between 0% and 10% in **Right tail trimming**.
1. **Facetting variables** enables you to create a grid of multiple boxplots. 
You can choose one or two facetting variables.  
It is **highly recommended** to include those variables for which multiple levels
where selected as facetting variables, such as in the case of 
**Density**, **Estimator** and **Bandwidth**.
1. Finally the **Log scale** and **Free scale** enables logarithm scale and 
different (free) scales for every row in the grid, respectively.

Once you generated a plot you will be able to add and customize a title. 
You can also change the plot size with the sliders that are shown
when you click on the top-left gear. 
Finally you can save the generated plot. If you specified a title, 
that will be used to name the file. 
Otherwise, a name based on a counter will be generated automatically.

#### Examples

* Example 1 **[todo]**
* Example 2 **[todo]**

### Heatmaps

#### Objective

The objective is similar to the one in Boxplots tab. 
However, here the information is presented in summarized manner 
(instead of the whole empirical distributions).

#### How to use

Set the options on the left panel and see the results after clicking on the **Plot** button. 

1. Choose the **Metric** you want to visualize.
1. Select a **Density** from which values where simulated. 
Here you can select only one density at a time.
1. You can also **exclude the Sheather-Jones** bandwidth estimator. 
This method is by far the slowest. 
Excluding it helps to make a better comparison of the other methods.

#### How to read

Each box contains 4 numbers in two rows. 
The first number in the first row corresponds to a 2% right-trimmed mean of the metric selected. 
Time is measured in seconds and error is the integated squared error.
The second number, within parenthesis, is the ratio between the mean
of the case in the box and the best result for the selected density (the minimum in both cases). 
Thus, a ratio of 1 represents the case with the best performance.
Finally, the second row represents the bounds of the central 94% density of the empirical distribution.


### Density

#### Objective

Play with different strategies to estimate density functions. 
In this tab you can generate random samples from a variety of probability distributions and
estimate the density function combining density and bandwidth estimators. 
A notable feature is that you can generate arbitrary mixtures of probability distributions. 
This is not only useful to see the strengths and weaknesses of each strategy, 
but also funny to play with.

#### How to use

Set the options on the left panel and see the results after clicking on the **Go!!** button.

* Choose the **distribution type**. 
  + **Fixed** lets you choose one of several popular distribution families. 
  When you select an option it generates one or two input parameters corresponding to the distribution.
  + **Mixture** lets you create a distribution as a result of up to three fixed distributions.
  When Mixture is selected a Settings button is enabled. Click on it and move the slider
  to specify the number of components. 
  Each distribution will be followed by inputs for its parameters as well as a Weight input.
  The weights are empty by default, which means all distributions are weighted equally.
  If you decide to specify weights, make sure they add up to 1. 
  Otherwise, they will be weighted equally and a warning will be raised.
  A text output is placed next to the setting button. It tells you which mixture you have created.
* Specify a **Sample Size**. The minimum value is 50 and the maximum is 100,000.
* Select an **Estimator**. 
You can choose any of the three methods used in the simulation. 
The classic Gaussian KDE, the adaptive Gaussian KDE and the Gaussian mixture fitted via EM.
* Select a **Bandwidth estimator**. Experimental averages the result of Improved Sheather-Jones and Silverman's rule.
* Finally there are two other features you can specify.
  + The first one is the **extension of variable domain**, 
  which extends the domain by 0.5 standard deviations. 
  This one is the default in most of the implementation. 
  However it can be disables, as you would expect when the range of the variable
  is known to be bounded.
  + The second feature is the **boundary correction**. 
  Since the implemented methods are based on Gaussian distributions, 
  a real domain is implicitly assumed. 
  When you bound the domain, you have to perform a boundary correction.

When you click on **Go!!** a sample of the required size is generated and
the density is estimated according the specifications. 
The plot shows a histogram of the sample, the true density function, and
the estimation. Then you can save the plot.

#### Examples

* Example 1 **[todo]**
* Example 2 **[todo]**
