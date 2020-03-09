# Load libraries ---------------------------------------------------------------

library(shiny)
library(shinyWidgets)
source("utils.R")

choices_bw_classic <- c("Silverman's rule" = "silverman",
                     "Scott's rule" = "scott",
                     "Sheather-Jones" = "sj",
                     "Improved Sheather-Jones" = "isj",
                     "Experimental" = "experimental")
choices_bw_mixture <- c("Default" = "mixture")

get_bw_choices <- function(x) {
  switch(x, "mixture" = choices_bw_mixture, choices_bw_classic)
}
  
choices_size_sj <- c(200, 500, 1000)
choices_size_default <- c(200, 500, 1000, 5000, 10000)
# TODO: this has to work with more than one bw
get_size_choices <- function(x) {
  switch(x, "sj" = choices_size_sj, choices_size_default)
}

# Load data --------------------------------------------------------------------
df_static <- readRDS("data/data.rds")

# Create server function -------------------------------------------------------

function(input, output, session) {
  
  store <- reactiveValues()
  store$df_static <- df_static

  observeEvent(
    input$estimator, {
    # req(input$estimator) # not needed, built in observeEvent
      updateSelectInput(
        session = session,
        inputId = "bw",
        choices = get_bw_choices(input$estimator),
        selected = get_bw_choices(input$estimator)[[1]]
      )
    })
  
  observeEvent(
    input$bw, {
      choices <- as.numeric(get_size_choices(input$bw))
      updateCheckboxGroupButtons(
        session = session,
        inputId = "size",
        choices = choices,
        selected = choices
      )
    }, ignoreNULL = TRUE, ignoreInit = TRUE #fixes only start, but does not select all when update
  )
  

  observeEvent({
    input$getPlot
  }, {
    store$df_filtered <- df_filter(store$df_static, input$estimator, 
                                   input$bw, as.numeric(input$size))
    
  })
  
  
  
  # Playground
  # store$df_head <- head(df_static)
  output$table <- renderDataTable(head(store$df_filtered))
  
}