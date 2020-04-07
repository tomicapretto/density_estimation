observeEvent(input$getPlotPanel2, {
  # be careful with this global assignment
  plot2Path <<- paste0("data/heatmaps/", 
                       input$metricPanel2, "_",
                       input$pdfPanel2)
  
  if (input$excludeSJ)  plot2Path <<- paste0(plot2Path, "_no_sj")
  plot2Path <<- paste0(plot2Path, ".png")
  
  output$plotPanel2 <- renderImage({
    return(
      list(
        src = plot2Path,
        contentType = "image/png",
        width = input$dimension[1] * 0.74 # Sidebar occupies 25% of width
      )
    )
  }, deleteFile = FALSE)
  
  output$downloadPlotPanel2UI <- renderUI({
    downloadButton("downloadPlotPanel2", "Save plot")
  })
  
})

output$downloadPlotPanel2 <- downloadHandler(
  filename = function() {
    basename(plot2Path)
  },
  
  content = function(file) {
    file.copy(plot2Path, file)
  }
)
