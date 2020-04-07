tabPanel(
  title = "Density plots",
  class = "fade in",
  sidebarLayout(
    sidebarPanel(
      selectInput(
        inputId = "contDist", 
        label = "Select a distribution",
        choices = c("Normal" = "norm",
                    "T-Student" = "t",
                    "Gamma" = "gamma",
                    "Exponential" = "exp",
                    "Beta" = "beta",
                    "Log Normal" = "lnorm",
                    "Cauchy" = "cauchy",
                    "Weibull" = "weibull",
                    "Uniform" = "unif")
      ),
      uiOutput("contParamsUI"),
      numericInput(
        inputId = "contSampleSize", 
        label = "Sample size", 
        value = 200, 
        min = 50, 
        max = 10000, 
        step = 1
      ),
      hr(),
      selectInput(
        inputId = "densityEstimator",
        label = "Select estimator",
        choices = c("Gaussian KDE" = "gaussian_kde",
                    "Adaptive Gaussian KDE" = "adaptive_kde",
                    "Gaussian mixture via EM" = "mixture_kde")
      ),
      uiOutput("bwMethodUI"),
      checkboxInput(
        inputId = "extendLimits",
        label = "Extend variable domain", 
        value = TRUE
      ),
      checkboxInput(
        inputId = "boundCorrection",
        label = "Perform boundary correction", 
        value = FALSE
      ),
      actionButton(
        "button3",
        label = "Go!!"
      ),
      width = 3
    ),
    mainPanel(
      uiOutput(
        "plotPanel3UI"
      ),
      uiOutput(
        "downloadPlotPanel3UI"
      ),
      width = 9
    )
  )
)
