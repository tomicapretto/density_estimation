# Load libraries ---------------------------------------------------------------

library(shiny)
library(shinyWidgets)
library(shinyBS)
source("utils.R")
source("server/helpers.R")

count_plot <- plot_counter()

# Load data --------------------------------------------------------------------
df_static <- readRDS("data/data.rds")

# Create server function -------------------------------------------------------

function(input, output, session) {
  session$onSessionEnded(stopApp)
  
  store <- reactiveValues()
  store$df_static <- df_static
  
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
    args <- list(store$df_static, input$estimator, input$bw, as.numeric(input$size))
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

    # Plot it!
    store$plt <- initialize_plot(store$df_trimmed, isolate(input$metric)) +
      add_boxplot() +
      custom_fill() +
      custom_theme() +
      custom_scale(scale = scale, log10 = isolate(input$log10)) +
      custom_facet(isolate(input$facetVars), free_y = isolate(input$freeScale)) + 
      labs(x = "Size", y = stringr::str_to_sentence(isolate(input$metric)))
  
    output$plot <- renderPlot({
      store$plt
    },
    res = 120)
  })
  
  observeEvent(input$plotTitleButton, {
    store$plt <- store$plt +
      ggtitle(input$plotTitle) +
      theme(
        plot.title = element_text(size = input$plotTitleSize,
                                  hjust = input$plotTitlePos)
      )
    
    output$plot <- renderPlot({
      store$plt
    },
    res = 120)
  })
  
  # Output plot
  
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
            value = 600, 
            step = 10
          )
        )
      })
    }, once = TRUE)
  
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
  
  output$downloadPlot <- downloadHandler(
    filename = function() {
      name <- paste0("plot", count_plot())
      paste0(name, '.png')
    },

    content = function(file) {
      png(
        file, 
        width = as.numeric(input$plotWidth / 100) * input$dimension[1],
        height = input$plotHeight, 
        res = 120)
      print(store$plt)
      dev.off()
    }
  )
  

}
