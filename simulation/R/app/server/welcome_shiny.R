showModal(
  modalDialog(
    title = "Create a Python environment?",
    p("The process usually takes about 30 seconds."),
    footer = tagList(
      p("The density estimator panel is not available if you do not enable this."),
      actionButton("python_cancel_btn", "Dismiss"),
      actionButton("python_add_btn", "OK")
    ),
    easyClose = FALSE
  ))

observeEvent(input$python_add_btn, {
  show("mainLayout")
  removeModal()
  PYTHON_STATUS <- init_python_shiny(input)
  if (PYTHON_STATUS) {
    source("server/panel1.R", local = TRUE)$value
    source("server/panel2.R", local = TRUE)$value
    source("server/panel3.R", local = TRUE)$value
  } else {
    show("mainLayout")
    hideTab(
      inputId = "tabs", 
      target = "Density plots"
    )
    source("server/panel1.R", local = TRUE)$value
    source("server/panel2.R", local = TRUE)$value
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