# Produce latex-like output to input simulated `pdf``
updateSelectizeInput(
  session = session, 
  inputId = "pdf",
  choices = pdf_choices,
  selected = pdf_choices[[1]],
  options = list(render = I(latex_input_script))
)

updateSelectizeInput(
  session = session, 
  inputId = "pdfPanel2",
  choices = pdf_choices,
  selected = pdf_choices[[1]],
  options = list(render = I(latex_input_script))
)

# Update `bw` input according to `estimator`.
observeEvent(
  input$estimator, {
    updateSelectInput(
      session = session,
      inputId = "bw",
      choices = get_bw_choices(input$estimator),
      selected = get_bw_choices(input$estimator)[[1]]
    )
  })

# Update `size` input according to `bw`.
observeEvent(
  input$bw, {
    choices <- as.numeric(get_size_choices(input$bw))
    updateCheckboxGroupButtons(
      session = session,
      inputId = "size",
      choices = choices,
      selected = choices
    )
  }, ignoreInit = TRUE # fixes only start, but does not select all when update
)

# Produce plot
observeEvent({
  input$getPlot
}, {
  
  # Plot data and settings
  # Filter
  args <- list(store$df_static, input$pdf, 
               input$estimator, input$bw, as.numeric(input$size))
  store$df_filtered <- do.call(df_filter, args)
  
  # Trim
  group_vars <- c("estimator", "bw", "size")
  quantile <- (100 - input$trimPct) / 100
  args <- list(store$df_filtered, group_vars, isolate(input$metric), quantile)
  store$df_trimmed <- do.call(df_trim, args)
  
  # Deduce scale
  scale <- if (input$metric == "time") {
    deduce_scale(store$df_trimmed[[isolate(input$metric)]])
  } else {
    ""
  }
  
  # Get accuracy
  # TODO: Improve this pls! Not always working nice.y
  if (scale == "ms") {
    acc_var <- store$df_trimmed[[input$metric]] * 100
  } else if (scale == "sec") {
    acc_var <- store$df_trimmed[[input$metric]]
  } else {
    acc_var <- store$df_trimmed[[input$metric]] / 10
  }
  
  acc <- precision(acc_var)
  
  # Plot it!
  store$plt <- initialize_plot(store$df_trimmed, isolate(input$metric)) +
    add_boxplot() +
    custom_fill() +
    custom_theme() +
    custom_scale(scale = scale, acc = acc, log10 = isolate(input$log10)) +
    custom_facet(isolate(input$facetVars), free_y = isolate(input$freeScale)) + 
    labs(x = "Size", y = stringr::str_to_sentence(isolate(input$metric)))
  
  output$plot <- renderPlot({
    store$plt
  },
  res = 120)
})

# Add title to plot
observeEvent(input$plotTitleButton, {
  store$plt <- store$plt +
    ggtitle(input$plotTitle) 
  theme(
    plot.title = element_text(size = input$plotTitleSize,
                              hjust = input$plotTitlePos)
  )
  output$plot <- renderPlot({
    store$plt
  },
  res = 120)
})

# Plot size displayed. It is used when saving too.
observeEvent(
  input$getPlot, {
    output$plotSizeUI <- renderUI({
      dropdownButton(
        inputId = "plotSize",
        icon = icon("gear"),
        status = "primary",
        circle = TRUE,
        
        sliderInput(
          inputId = "plotWidth",
          label = "Plot width (%)",
          min = 40, 
          max = 100, 
          value = 90, 
          step = 1
        ),
        
        sliderInput(
          inputId = "plotHeight",
          label = "Plot height (px)",
          min = 400, 
          max = 1200, 
          value = 500, 
          step = 10
        )
      )
    })
  }, once = TRUE)


# Output plot according to size setings
observeEvent(
  input$getPlot, {
    output$plotUI <- renderUI({
      req(input$plotHeight)
      plotOutput(
        "plot", 
        height = paste0(input$plotHeight, "px"),
        width = paste0(input$plotWidth, "%")
      )
    })
  })

# Generate several UI components only after a plot is generated.
# * Download button
# * Title settings
# * Apply title settings button
# * Collpsable panel
observeEvent(
  input$getPlot, {
    output$downloadPlotUI <- renderUI({
      downloadButton("downloadPlot", "Save plot")
    })
    
    output$plotTitleSettings <- renderUI({
      fluidRow(
        column(
          4,
          textInput("plotTitle", "Title")),
        column(
          4,
          numericInput("plotTitleSize", "Title size", value = 14, min = 2)),
        column(
          4,
          selectInput("plotTitlePos", "Title position", 
                      choices = c("Left" = 0, "Center" = 0.5, "Right" = 1),
                      selected = 0)
        )
      )
    })
    
    # 
    output$plotTitleButtonUI <- renderUI({
      actionButton(
        inputId = "plotTitleButton",
        label = "Apply"
      )
    })
    
    output$plotTitleUI <- renderUI({
      bsCollapse(
        id = "Panel 1",
        bsCollapsePanel(
          "Title settings", 
          uiOutput("plotTitleSettings"),
          uiOutput("plotTitleButtonUI")
        )
      )
    })
    
  }, once = TRUE)

# Configure plot download handler.
# It uses mainPanel dimensions to calculate plot width.
output$downloadPlot <- downloadHandler(
  filename = function() {
    if (is.null(input$plotTitle)) {
      name <- paste0("plot", count_plot())
    } else if (input$plotTitle == "") {
      name <- paste0("plot", count_plot())
    } else {
      name <- stringr::str_replace_all(input$plotTitle, " ", "_")
    }
    paste0(name, '.png')
  },
  
  content = function(file) {
    png(
      file, 
      width = as.numeric(input$plotWidth / 100) * (input$dimension[1] * 0.74), # Sidebar occupies 25% of width
      height = input$plotHeight, 
      res = 120)
    print(store$plt)
    dev.off()
  }
)
