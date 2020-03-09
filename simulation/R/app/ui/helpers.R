library(shinyWidgets)

# Sidebar panel ----------------------------------------------------------------
mySidebarPanel <- function() {
  sidebarPanel(
    
    radioGroupButtons(
      inputId = "metric",
      label = "Select a metric",
      choices = c("Time", "Error"),
      justified = TRUE
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
    
    # TODO: Add update input
    selectInput(
      inputId = "bw",
      label = "Bandwidth(s)",
      choices = c("Silverman's rule" = "silverman",
                  "Scott's rule" = "scott",
                  "Sheather-Jones" = "sj",
                  "Improved Sheather-Jones" = "isj",
                  "Experimental" = "experimental"),
      selected = "silverman",
      multiple = TRUE
    ),
    
    # TODO: Add update input
    checkboxGroupButtons(
      inputId = "size",
      label = "Sample size(s)",
      choices = c(200, 500, 1000, 5000, 10000),
      justified = TRUE, 
      selected = c(200, 500, 1000, 5000, 10000)
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
    
    hr(),
    
    h4("Optional"),
    
    # Only first two variables are going to be selected
    # The order will be first row, then column.
    selectInput(
      inputId = "facetVars",
      label = "Facetting variables",
      choices = c("Bandwidth" = "bw", 
                  "Estimator" = "estimator"),
      multiple = TRUE
    ),
    
    hr(),
    
    actionButton(
      inputId = "getPlot", 
      label = "Plot"
    )
  )
}


# Main panel -------------------------------------------------------------------
myMainPanel <- function() {
  mainPanel(
  uiOutput("mytext"),
  dataTableOutput("table")
  )
}