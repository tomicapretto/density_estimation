tabPanel(
  title = "Boxplots",
  class = "fade in",
  sidebarLayout(
    sidebarPanel(
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
        size = "sm"
      ),
      
      sliderInput(
        inputId = "trimPct",
        label = "Right tail trimming",
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
            label = strong("Log scale"),
            value = FALSE
          )
        ),
        column(
          width = 6,
          checkboxInput(
            inputId = "freeScale",
            label = strong("Free scale"),
            value = FALSE
          )
        )
      ),
      actionButton(
        inputId = "getPlot", 
        label = "Plot"
      ),
      width = 3
    ),
    mainPanel(
      uiOutput("plotSizeUI"),
      uiOutput("plotUI"),
      uiOutput("plotTitleUI"),
      uiOutput("downloadPlotUI"),
      width = 9
    )
  )
)

