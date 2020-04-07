# Sidebar panel ----------------------------------------------------------------

mySidebarPanel <- function() {
  
  sidebarPanel(
    
    # Boxplot tab 
    conditionalPanel(
      "input.tabSelected == 1",

      radioGroupButtons(
        inputId = "metric",
        label = "Select a metric",
        choices = c("Time" = "time", "Error" = "error"),
        justified = TRUE
      ),
      
      selectizeInput(
        inputId = "pdf",
        label = "Choose a density", 
        multiple = TRUE,
        choices = NULL
      ),
      
      selectInput(
        inputId = "estimator", 
        label = "Estimator(s)",
        choices = c("Gaussian KDE" = "fixed_gaussian", 
                    "Adaptive Gaussian KDE" = "adaptive_gaussian", 
                    "Gaussian mixture" = "mixture"), 
        selected = "fixed_gaussian",
        multiple = TRUE
      ),
      
      selectInput(
        inputId = "bw",
        label = "Bandwidth(s)",
        choices = choices_bw_classic,
        selected = choices_bw_classic[[1]],
        multiple = TRUE
      ),
      
      checkboxGroupButtons(
        inputId = "size",
        label = "Sample size(s)",
        choices = choices_size_default,
        selected = choices_size_default,
        justified = TRUE, 
        checkIcon = list(
          yes = tags$i(class = "fa fa-circle", 
                       style = "color: steelblue"),
          no = tags$i(class = "fa fa-circle-o", 
                      style = "color: steelblue"))
      ),
      
      sliderInput(
        inputId = "trimPct",
        label = "Trim percent",
        min = 0,
        max = 10,
        value = 5,
        step = 0.5,
        post = "%" 
      ),
      
      h4(strong("Optional")),
      
      # Only first two variables are going to be selected
      # The order will be first row, then column.
      selectInput(
        inputId = "facetVars",
        label = "Facetting variables",
        choices = c("Bandwidth" = "bw", 
                    "Estimator" = "estimator",
                    "Density" = "pdf"),
        multiple = TRUE
      ),
      
      fluidRow(
        column(
          width = 6,
          checkboxInput(
            inputId = "log10",
            label = strong("Log-scale"),
            value = FALSE
          ),
          
          checkboxInput(
            inputId = "freeScale",
            label = strong("Free y-scale"),
            value = FALSE
          )
        ),
        column(
          width = 6,
          actionButton(
            inputId = "getPlot", 
            label = "Plot"
          )
        )
      )
    ),
    
    conditionalPanel(
      "input.tabSelected == 2",
      h4(strong("Mandatory")),
      
      radioGroupButtons(
        inputId = "metricPanel2",
        label = "Select a metric",
        choices = c("Time" = "time", "Error" = "error"), 
        justified = TRUE
      ),
      
      selectizeInput(
        inputId = "pdfPanel2",
        label = "Choose a density", 
        choices = NULL
      ),
      
      hr(),
      
      checkboxInput(
        inputId = "excludeSJ",
        label = "Exclude Sheather-Jones",
        value = FALSE
      ),
      
      actionButton(
        inputId = "getPlotPanel2",
        label = "Plot"
      )
    ),
    
    conditionalPanel(
      "input.tabSelected == 3",
      
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
        step = 1),
      
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
      )
    ),
    width = 3
  )
}


# Main panel -------------------------------------------------------------------

# Window size captures script
# The first part takes size when the app is connected
# The second part takes size when a resize is performed
tabPanel1 <- function() {
  tabPanel(
    title = "Boxplots",
    value = 1,
    uiOutput("plotSizeUI"),
    uiOutput("plotUI"),
    uiOutput("plotTitleUI"),
    uiOutput("downloadPlotUI")
  )
}

tabPanel2 <- function() {
  tabPanel(
    title = "Heatmaps",
    value = 2,
    imageOutput("plotPanel2", inline = TRUE),
    uiOutput("downloadPlotPanel2UI")
  )
}

tabPanel3 <- function() {
  tabPanel(
    title = "Estimators",
    value = 3,
    uiOutput("plotPanel3UI")
  )
}

myMainPanel <- function() {
  mainPanel(
    # Capture window size, used to save plots.
    tags$head(
      tags$script(window_capture_script),
    ),  
    # Render Latex
    tags$head(
      tags$link(
        rel = "stylesheet", 
        href = "https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.css", 
        integrity = "sha384-9tPv11A+glH/on/wEu99NVwDPwkMQESOocs/ZGXPoIiLE8MU/qkqUcZ3zzL+6DuH", 
        crossorigin = "anonymous"),
      tags$script(
        src = "https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.js", 
        integrity = "sha384-U8Vrjwb8fuHMt6ewaCy8uqeUXv4oitYACKdB0VziCerzt011iQ/0TqlSlv8MReCm", 
        crossorigin = "anonymous")
    ),
    tabsetPanel(
      tabPanel1(),
      tabPanel2(),
      tabPanel3(),
      id = "tabSelected"
    ),
    width = 9
  )
}