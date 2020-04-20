tabPanel(
  title = "Heatmaps",
  class = "fade in",
  sidebarLayout(
    sidebarPanel(
      radioGroupButtons(
        inputId = "heatmaps_metric",
        label = "Select a metric",
        choices = c("Time" = "time", "Error" = "error"), 
        justified = TRUE
      ),
      selectizeInput(
        inputId = "heatmaps_pdf",
        label = "Choose a density", 
        choices = NULL
      ),
      hr(),
      checkboxInput(
        inputId = "heatmaps_exclude_sj",
        label = "Exclude Sheather-Jones",
        value = FALSE
      ),
      actionButton(
        inputId = "heatmaps_plot_btn",
        label = "Plot"
      ),
      width = 3
    ),
    mainPanel(
      value = 2,
      imageOutput("heatmaps_plot", inline = TRUE),
      uiOutput("heatmaps_download_plot_ui"),
      width = 9
    )
  )
)

