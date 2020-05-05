tabPanel(
  title = "About",
  class = "fade in",
  icon  = icon("info-circle"),
  
  h2("Quick overview"),
  p("This shiny application is a compliment of the work presented in", strong("[todo]."),
    "The objective of this app is to enable the interactive exploration of simulation results 
     as well as testing different density estimation methods with samples 
     coming from a wide range of probability distributions."),
  
  h3("Boxplots"),
  h4("Objective"),
  p("Visualize and compare the performance of the different estimators under 
     the different scenarios  in terms of computational time and 
     estimation error measured via the Integrated Squared Error."),
  h4("How to use"),
  p("Set the options on the left panel and see the results after clicking on the", 
     strong("Plot"), "button."),
  
  tags$ul(
    tags$li(HTML("Choose the <strong>Metric</strong> you want to visualize.")),
    tags$li(HTML("Specify one or more probability distributions from which random samples were
             simulated and used to estimate the underlying probability density function
             in the <strong>Density</strong> field.")),
    tags$li(HTML("Select one or more <strong>Estimators(s)</strong>. There are three available options.
             The classic Gaussian KDE, an adaptive Gaussian KDE and a method 
             that fits a Gaussian Mixture via the EM algorithm.")),
    tags$li(HTML('Select one or more <strong>Bandwidth(s)</strong> estimation methods. 
             Note the Gaussian mixture via EM only uses the method we decided to call "Mixture".')),
    tags$li(HTML("By default all the sample sizes used in the simulation are selected 
            <strong>Sample size(s)</strong> lets you select a subset of these.")),
    tags$li(HTML("Since some distributions tend to be highly right-skewed, it is possible to trim
             this tail between 0% and 10% in <strong>Right tail trimming</strong>.")),
    tags$li(HTML("<strong>Facetting variables</strong> enables you to create a grid of multiple boxplots.
            You can choose one or two facetting variables. <br/>
            It is <strong>highly recommended</strong> to include those variables for which
            multiple levels where selectes as facetting variables, such as in the case of
            <strong>Density</strong>, <strong>Estimator</strong>, and <strong>Bandwidth</strong>.")),
    tags$li(HTML("Finally the <strong>Log scale</strong> and <strong>Free scale</strong>
                  enables logarithm scale and different (free) scales for every row in the grid, 
                  respectively."))
  ),
  
  p("Once you generated a plot you will be able to add and customize a title. 
     You can also change the plot size with the sliders that are shown
     when you click on the top-left gear. 
     Finally you can save the generated plot. If you specified a title, 
     that will be used to name the file. 
     Otherwise, a name based on a counter will be generated automatically."),
  
  h4("Examples"),
  tags$ul(
    tags$li(
      actionLink(
        inputId = "boxplots_example_1",
        label = "Example 1: Time comparison of Classic and Adaptive Gaussian KDE for some bandwidths."
      )
    ),
    tags$li(
      actionLink(
        inputId = "boxplots_example_2", 
        label = "Example 2: Error comparison when estimating Gamma 
                 distributions with Adaptive Gaussian KDE and different bandwidths."
      )
    )
  ),
  
  h3("Heatmaps"),
  h4("Objective"),
  p("The objective is similar to the one in Boxplots tab. 
     However, here the information is presented in summarized manner 
     (instead of the whole empirical distributions)."),
  
  h4("How to use"),
  p(HTML("Set the options on the left panel and see the results after 
          clicking on the <strong>Plot</strong> button.")),
  tags$ul(
    HTML(
      "<li>Choose the <strong>Metric</strong> you want to visualize.</li>
       <li>Select a <strong>Density</strong> from which values where simulated. 
           Here you can select only one density at a time.</li>
       <li>You can also <strong>exclude the Sheather-Jones</strong> bandwidth estimator. 
           This method is by far the slowest. 
           Excluding it helps to make a better comparison of the other methods.</li>"
    )
  ),
  
  h4("How to read"),
  p("Each box contains 4 numbers in two rows. 
     The first number in the first row corresponds to a 2% right-trimmed mean of the metric selected. 
     Time is measured in seconds and error is the integated squared error.
     The second number, within parenthesis, is the ratio between the mean
     of the case in the box and the best result for the selected density (the minimum in both cases). 
     Thus, a ratio of 1 represents the case with the best performance.
     Finally, the second row represents the bounds of the central 94% density 
     of the empirical distribution."),
  
  h3("Density"),
  h4("Objective"),
  p("Play with different strategies to estimate density functions. 
     In this tab you can generate random samples from a variety of probability distributions and
     estimate the density function combining density and bandwidth estimators. 
     A notable feature is that you can generate arbitrary mixtures of probability distributions. 
     This is not only useful to see the strengths and weaknesses of each strategy, 
     but also funny to play with."),
  
  h4("How to use"),
  p("Set the options on the left panel and see the results after clicking on the", 
    strong("Go!!"), "button."),
  tags$ul(
    HTML(
      "<li>Choose the <strong>Distribution type.</strong></li>
      <ul>
      <li><strong>Fixed</strong> lets you choose one of 
      several popular distribution families.</br> 
      When you select an option it generates one or two input 
      parameters corresponding to the distribution.
      </li>
      <li><strong>Mixture</strong> lets you create a distribution as 
      a result of up to three fixed distributions.
      When Mixture is selected a Settings button is enabled. 
      Click on it and move the slider to specify the number of components. 
      Each distribution will be followed by inputs for 
      its parameters as well as a Weight input.
      The weights are empty by default, which means all 
      distributions are weighted equally.
      If you decide to specify weights, make sure they add up to 1. 
      Otherwise, they will be weighted equally and a warning will be raised.
      A text output is placed next to the setting button. 
      It tells you which mixture you have created.
      </li>
      </ul>
      <li>
      Specify a <strong>Sample Size</strong>. 
      The minimum value is 50 and the maximum is 100,000.
      </li>
      <li>
      Select an <strong>Estimator</strong></br>
      You can choose any of the three methods used in the simulation. 
      The classic Gaussian KDE, the adaptive Gaussian KDE 
      and the Gaussian mixture fitted via EM.
      </li>
      <li>
      Select a <strong>Bandwidth estimator</strong>. Experimental averages
      the result of Improved Sheather-Jones and Silverman.
      </li>
      <li>
      Finally there are two other features you can specify.
      </li>
      <ul>
      <li>
      The first one is the <strong>extension of variable domain</strong>, 
      which extends the domain by 0.5 standard deviations. 
      This one is the default in most of the implementation. 
      However it can be disables, as you would expect when the range of the variable
      is known to be bounded.
      </li>
      <li>
      The second feature is the <strong>boundary correction</strong>. 
      Since the implemented methods are based on Gaussian distributions, 
      a real domain is implicitly assumed. 
      When you bound the domain, you have to perform a boundary correction.
      </li>
      </ul>"
    )
  ),
  p("When you click on", strong("Go!!"), "a sample of the required size is generated and
     the density is estimated according the specifications. 
     The plot shows a histogram of the sample, the true density function, and
     the estimation. Then you can save the plot."),
  
  h4("Examples"),
  uiOutput("density_examples_links")
)

