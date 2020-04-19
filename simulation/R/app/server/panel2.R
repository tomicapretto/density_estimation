updateSelectizeInput(
  session = session, 
  inputId = "heatmaps_pdf",
  choices = pdf_choices,
  selected = pdf_choices[[1]],
  options = list(render = I(latex_input_script))
)

observeEvent(input$heatmaps_plot_btn, {
  store$heatmaps_plot_path <- paste0(
    "data/heatmaps/", input$heatmaps_metric, "_", input$heatmaps_pdf)
  
  if (input$heatmaps_exclude_sj) {
    store$heatmaps_plot_path <- paste0(store$heatmaps_plot_path, "_no_sj")
  }
  store$heatmaps_plot_path <- paste0(store$heatmaps_plot_path, ".png")
  
  output$heatmaps_plot <- renderImage({
    return(
      list(
        src = store$heatmaps_plot_path,
        contentType = "image/png",
        width = input$dimension[1] * 0.74
      )
    )
  }, deleteFile = FALSE)
  
  output$heatmaps_download_plot_ui <- renderUI({
    downloadButton("heatmaps_download_plot", "Save plot")
  })
})

output$heatmaps_download_plot <- downloadHandler(
  filename = function() {
    basename(store$heatmaps_plot_path)
  },
  content = function(file) {
    file.copy(store$heatmaps_plot_path, file)
  }
)
