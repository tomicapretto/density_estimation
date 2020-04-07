tabPanel(
  title = "Heatmaps",
  class = "fade in",
  sidebarLayout(
    sidebarPanel(
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
      ),
      width = 3
    ),
    mainPanel(
      value = 2,
      imageOutput("plotPanel2", inline = TRUE),
      uiOutput("downloadPlotPanel2UI"),
      width = 9
    )
  )
)

