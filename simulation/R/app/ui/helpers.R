source("utils.R")

# Sidebar panel ----------------------------------------------------------------
mySidebarPanel <- function() {
  sidebarPanel(
    
    radioGroupButtons(
      inputId = "metric",
      label = "Select a metric",
      choices = c("Time" = "time", "Error" = "error"),
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
      justified = TRUE, 
      selected = choices_size_default
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
    
    h4(strong("Optional")),
    
    # Only first two variables are going to be selected
    # The order will be first row, then column.
    selectInput(
      inputId = "facetVars",
      label = "Facetting variables",
      choices = c("Bandwidth" = "bw", 
                  "Estimator" = "estimator"),
      multiple = TRUE
    ),
    
    checkboxInput(
      inputId = "log10",
      label = strong("Log-scale"),
      value = FALSE
    ),
    
    checkboxInput(
      inputId = "freeScale",
      label = strong("Free y-scale"),
      value = FALSE
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
    # Capture window size, used to save plots.
    tags$head(
      tags$script('
                  var dimension = [0, 0];
                  $(document).on("shiny:connected", function(e) {
                    dimension[0] = window.innerWidth;
                    dimension[1] = window.innerHeight;
                    Shiny.onInputChange("dimension", dimension);
                  });
                  $(window).resize(function(e) {
                    dimension[0] = window.innerWidth;
                    dimension[1] = window.innerHeight;
                    Shiny.onInputChange("dimension", dimension);
                  });'
                  )
      ),  

  uiOutput("plotSizeUI"),
  uiOutput("plotUI"),
  uiOutput("plotTitleUI"),
  uiOutput("downloadPlotUI")
  )
}