showModal(
  modalDialog(
  title = "Specify your Python 3 directory",
  h4(strong("Select a directory")),
  fluidRow(
    column(
      width = 10,
      textInput(
        inputId = "python_path",
        label = NULL,
        width = "100%"
      )
    ),
    column(
      width = 2,
      actionButton(
        inputId = "python_look_path_btn",
        label = "Browse"
      )
    )
  ),
  verbatimTextOutput(
    "modal_text"
  ),
  footer = tagList(
    p("The density estimator panel is not available if you do not have Python 3.x"),
    p("Numpy, scipy and statsmodels are", strong("required")),
    actionButton("python_cancel_btn", "Dismiss"),
    actionButton("python_add_path_btn", "OK")
  ),
  easyClose = FALSE
))

observeEvent(input$python_look_path_btn, {
  PYTHON_DIR <- choose_directory()
  updateTextInput(
    session = session, 
    inputId = "python_path",
    value = gsub("\\\\", "/", PYTHON_DIR)
    )
  output$modal_text <- renderText(
    input$python_path
  )
})

observeEvent(input$python_add_path_btn, {
  if (!is_valid_path(input$python_path)) {
    showNotification("Please input a valid path", type = "error")
  } else if (!dir.exists(input$python_path)) {
    showNotification("Path not found.", type = "error")
  } else {
    show("mainLayout")
    removeModal()
    store$PYTHON_PATH <- init_python_custom(input)
    source("server/panel1.R", local = TRUE)$value
    source("server/panel2.R", local = TRUE)$value
    source("server/panel3.R", local = TRUE)$value
  }
})

observeEvent(input$python_cancel_btn, {
  show("mainLayout")
  removeModal()
  hideTab(
    inputId = "tabs", 
    target = "Density plots"
  )
  source("server/panel1.R", local = TRUE)$value
  source("server/panel2.R", local = TRUE)$value
})