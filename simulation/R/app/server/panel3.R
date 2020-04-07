output$contParamsUI <- renderUI({
  tagList(
    distribution_parameters_cont(input$contDist)
  )
})

output$bwMethodUI <- renderUI({
  if (input$densityEstimator %in% c("gaussian_kde", "adaptive_kde")) {
    selectInput(
      inputId = "bwMethod",
      label = "Bandwidth method",
      choices = c(
        "Silverman's rule" = "silverman",
        "Scott's rule" = "scott",
        "Least Squares Cross Validation" = "lscv",
        "Sheather-Jones" = "sj",
        "Improved Sheather-Jones" = "isj",
        "Experimental" = "experimental"
      )
    )
  } else {
    selectInput(
      inputId = "bwMethod",
      label = "Bandwidth method",
      choices = c("Mixture" = "mixture")
    )
  }
})

observeEvent(input$button3, {

  if (input$contSampleSize <= 100000) {
    if (input$bwMethod %in% c("lscv", "sj")) {
      if (input$contSampleSize >= 5000) {
        showNotification(
          ui = HTML(paste(c("Sample size is too large for the chosen method.",
                            "Computation is not performed."), collapse = "<br/>")),
          type = "error"
        )
      } else if (input$contSampleSize >= 2000) {
        confirmSweetAlert(
          session = session,
          inputId = "confirm_computation",
          type = "warning",
          title = "The computation may take a while, confirm?",
          danger_mode = TRUE
        )
        observeEvent(input$confirm_computation, {
          if (input$confirm_computation) {
            show_spinner()
            density_plot_params <- get_density_params(input)
            store$density_plot <- density_plot_generator(density_plot_params)
            hide_spinner()
          }
        }, ignoreNULL = TRUE)
      } else {
        show_spinner()
        density_plot_params <- get_density_params(input)
        store$density_plot <- density_plot_generator(density_plot_params)
        hide_spinner()
      }
    } else {
      show_spinner()
      density_plot_params <- get_density_params(input)
      store$density_plot <- density_plot_generator(density_plot_params)
      hide_spinner()
    }
  } else {
    showNotification(
      ui = HTML(paste(c("The maximum sample size is 100,000.",
                        "Computation is not performed."), collapse = "<br/>")),
      type = "error"
    )
  }
})


output$plotPanel3 <- renderPlot({
  req(store$density_plot)
  store$density_plot()
})

output$plotPanel3UI <- renderUI({
  plotOutput(
    "plotPanel3",
    height = "520px",
    width = "95%"
  )
})

output$downloadPlotPanel3UI <- renderUI({
  req(store$density_plot)
  tagList(
    h4("File name"),
    fluidRow(
      column(
        width = 3, 
        textInput(
          inputId = "filenamePlotPanel3", 
          label = NULL, 
          value = NULL,
          placeholder = "Optional"
        )
      ),
      column(
        width = 3,
        downloadButton("downloadPlotPanel3", "Save plot"))
    )
  )
})

output$downloadPlotPanel3 <- downloadHandler(
  filename = function() {
    if (is.null(input$filenamePlotPanel3)) {
      name <- paste0("plot", count_plot_dens())
    } else if (input$filenamePlotPanel3 == "") {
      name <- paste0("plot", count_plot_dens())
    } else {
      name <- stringr::str_replace_all(input$filenamePlotPanel3, " ", "_")
    }
    paste0(name, '.png')
  },
  
  content = function(file) {
    png(
      file, 
      width = 3 * (input$dimension[1] * 0.70), 
      height = 3 * 520, 
      res = 240)
    store$density_plot()
    dev.off()
  }
)


