tabPanel(
  title = "Density plots",
  class = "fade in",
  sidebarLayout(
    sidebarPanel(
      
      radioGroupButtons(
        inputId = "density_dist_type",
        label = "Distribution type",
        choices = c("Fixed" = "fixed", "Mixture" = "mixture"), 
        justified = TRUE
      ),
      
      uiOutput("density_distribution_ui"),
      
      numericInput(
        inputId = "density_sample_size", 
        label = "Sample size", 
        value = 200, 
        min = 50, 
        max = 10000, 
        step = 1
      ),
      hr(),
      selectInput(
        inputId = "density_estimator",
        label = "Select estimator",
        choices = c("Gaussian KDE" = "gaussian_kde",
                    "Adaptive Gaussian KDE" = "adaptive_kde",
                    "Gaussian mixture via EM" = "mixture_kde")
      ),
      uiOutput("density_bw_method_ui"),
      checkboxInput(
        inputId = "density_extend_limits",
        label = "Extend variable domain", 
        value = TRUE
      ),
      checkboxInput(
        inputId = "density_bound_correction",
        label = "Perform boundary correction", 
        value = FALSE
      ),
      actionButton(
        "density_plot_btn",
        label = "Go!!"
      ),
      width = 3
    ),
    mainPanel(
      uiOutput(
        "density_plot_ui"
      ),
      uiOutput(
        "density_download_plot_ui"
      ),
      width = 9
    )
  )
)
