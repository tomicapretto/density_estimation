# Produce latex-like output to input simulated `boxplots_pdf``
updateSelectizeInput(
  session = session, 
  inputId = "boxplots_pdf",
  choices = pdf_choices,
  selected = pdf_choices[[1]],
  options = list(render = I(latex_input_script))
)

# Update `boxplots_bw` input according to `boxplots_estimator`.
observeEvent(input$boxplots_estimator, {
  updateSelectInput(
    session = session,
    inputId = "boxplots_bw",
    choices = get_bw_choices(input$boxplots_estimator),
    selected = get_bw_choices(input$boxplots_estimator)[[1]]
  )
})

# Update `size` input according to `boxplots_bw`.
observeEvent(input$boxplots_bw, {
  choices <- as.character(get_size_choices(input$boxplots_bw))
  updateCheckboxGroupButtons(
    session = session,
    inputId = "boxplots_size",
    choices = choices,
    selected = choices
  )
  }, ignoreInit = TRUE # fixes only start, but does not select all when update
)

# Produce plot
observeEvent(input$boxplots_plot_btn, {
  args <- list(store$df_static, input$boxplots_pdf, 
               input$boxplots_estimator, input$boxplots_bw, 
               as.numeric(input$boxplots_size))
  
  if (check_boxplot_args(args)) {
    # Check arguments are not empty
    showNotification(
      paste("At least one required argument is empty"),
      type = "error"
    )
  } else {
    # Plot data and settings
    # Filter
    store$df_filtered <- do.call(df_filter, args)
    
    # Convert facetting variables to factors with order resembling the one in the input
    store$df_filtered <- df_facet_order(store$df_filtered, input)
    
    # Trim
    group_vars <- c("pdf", "estimator", "bw", "size")
    quantile <- (100 - input$boxplots_trim_pct) / 100
    args <- list(store$df_filtered, group_vars, isolate(input$boxplots_metric), quantile)
    store$df_trimmed <- do.call(df_trim, args)
    
    # Get precision for scaling
    prec <- precision(store$df_trimmed[[input$boxplots_metric]])
    time_flag <- if (input$boxplots_metric == "time") TRUE else FALSE
    
    # Plot it!
    store$boxplots_plot <- initialize_plot(
      store$df_trimmed, 
      isolate(input$boxplots_metric)
      ) +
      add_boxplot() +
      custom_fill() +
      custom_theme() +
      custom_scale(log10 = isolate(input$boxplots_log10), prec = prec, time_flag = time_flag) +
      custom_facet(isolate(input$boxplots_facet_vars), free_y = isolate(input$boxplots_free_y)) + 
      labs(x = "Size", y = stringr::str_to_sentence(isolate(input$boxplots_metric)))
    
    output$boxplots_plot <- renderPlot({
      store$boxplots_plot
    },
    res = 120)
  }
})

# Add title to plot
observeEvent(input$boxplots_plot_title_btn, {
  if (trimws(input$boxplots_plot_title) == "") {
    ttl <- NULL
  } else {
    ttl <- input$boxplots_plot_title
  }

  store$boxplots_plot <- store$boxplots_plot +
    labs(
      title = ttl
    ) +
    theme(plot.title = element_text(
      size = input$boxplot_plot_title_size,
      hjust = input$boxplot_plot_title_pos))
  
  output$boxplots_plot <- renderPlot({
    store$boxplots_plot
  },
  res = 120)
})

# Plot size displayed. It is used when saving too.
observeEvent(input$boxplots_plot_btn, {
  output$boxplots_plot_size_UI <- renderUI({
    dropdownButton(
      inputId = "boxplots_plot_size",
      icon = icon("gear"),
      status = "primary",
      circle = TRUE,
      
      sliderInput(
        inputId = "boxplots_plot_width",
        label = "Plot width (%)",
        min = 40, 
        max = 100, 
        value = 90, 
        step = 1
      ),
      
      sliderInput(
        inputId = "boxplots_plot_height",
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
observeEvent(input$boxplots_plot_btn, {
  output$boxplots_plot_UI <- renderUI({
    req(input$boxplots_plot_height)
    plotOutput(
      outputId = "boxplots_plot",
      height = paste0(input$boxplots_plot_height, "px"),
      width = paste0(input$boxplots_plot_width, "%")
    )
  })
})

# Generate several UI components only after a plot is generated.
# * Download button
# * Title settings
# * Apply title settings button
# * Collpsable panel
observeEvent(input$boxplots_plot_btn, {
  output$boxplots_download_UI <- renderUI({
    downloadButton("downloadPlot", "Save plot")
  })
    
  output$boxplots_plot_title_settings_ui <- renderUI({
    fluidRow(
      column(
        width = 4,
        textInput(
          inputId = "boxplots_plot_title", 
          label = "Title"
        )
      ),
      column(
        width = 4,
        numericInput(
          inputId = "boxplot_plot_title_size", 
          label = "Title size", 
          value = 14, 
          min = 2
        )
      ),
      column(
        width = 4,
        selectInput(
          inputId = "boxplot_plot_title_pos", 
          label = "Title position", 
          choices = c("Left" = 0, "Center" = 0.5, "Right" = 1),
          selected = 0
        )
      )
    )
  })
    
  output$boxplots_plot_title_btn_ui <- renderUI({
    actionButton(
      inputId = "boxplots_plot_title_btn",
      label = "Apply"
    )
  })
  
  output$boxplots_plot_title_UI <- renderUI({
    bsCollapse(
      id = "Panel 1",
      bsCollapsePanel(
        "Title settings", 
        uiOutput("boxplots_plot_title_settings_ui"),
        uiOutput("boxplots_plot_title_btn_ui")
      )
    )
  })
}, once = TRUE)

# Configure plot download handler.
# It uses mainPanel dimensions to calculate plot width.
output$downloadPlot <- downloadHandler(
  filename = function() {
    if (is.null(input$boxplots_plot_title)) {
      name <- paste0("plot", count_plot())
    } else if (input$boxplots_plot_title == "") {
      name <- paste0("plot", count_plot())
    } else {
      name <- gsub(" ", "_", input$boxplots_plot_title)
    }
    paste0(name, '.png')
  },
  
  content = function(file) {
    png(
      filename = file, 
      width = as.numeric(input$boxplots_plot_width / 100) * (input$dimension[1] * 0.72),
      height = input$boxplots_plot_height, 
      res = 120)
    print(store$boxplots_plot)
    dev.off()
  }
)
