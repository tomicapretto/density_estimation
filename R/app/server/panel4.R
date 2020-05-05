# Boxplot examples -------------------------------------------------------------
# Example 1
boxplots_example1_rctv <- reactive(input$boxplots_example_1)
boxplots_example1_d1 <- debounce(boxplots_example1_rctv, 100)
boxplots_example1_d2 <- debounce(boxplots_example1_rctv, 300)
boxplots_example1_d3 <- debounce(boxplots_example1_rctv, 500)

observeEvent(input$boxplots_example_1, {
  updateTabsetPanel(
    session = session,
    inputId = "tabs",
    selected = "Boxplots"
  )
})

observeEvent(boxplots_example1_d1(), {
  updateRadioGroupButtons(
    session = session,
    inputId = "boxplots_metric", 
    selected = "time"
  )
  updateSelectizeInput(
    session = session,
    inputId = "boxplots_pdf",
    selected = "gaussian_1"
  )
  updateSelectInput(
    session = session,
    inputId = "boxplots_estimator",
    selected = c("fixed_gaussian", "adaptive_gaussian")
  )
  updateSelectInput(
    session = session,
    inputId = "boxplots_facet_vars",
    selected = c("bw", "estimator")
  )
  updateCheckboxInput(
    session = session,
    inputId = "boxplots_log10",
    value = FALSE
  )
  updateCheckboxInput(
    session = session,
    inputId = "boxplots_free_y",
    value = FALSE
  )
  updateSliderInput(
    session = session,
    inputId = "boxplots_trim_pct",
    value = 5
  )
  # Not necessary, triggered when you update `boxplots_bw`
  # updateCheckboxGroupButtons( 
  #   session = session,
  #   inputId = "boxplots_size",
  #   selected = as.character(choices_size_default)
  # )
})

observeEvent(boxplots_example1_d2(), {
  updateSelectInput(
    session = session,
    inputId = "boxplots_bw",
    selected = c("silverman", "isj")
  )
})

observeEvent(boxplots_example1_d3(), {
  click("boxplots_plot_btn")
})

# Example 2
boxplots_example2_rctv <- reactive(input$boxplots_example_2)
boxplots_example2_d1 <- debounce(boxplots_example2_rctv, 100)
boxplots_example2_d2 <- debounce(boxplots_example2_rctv, 300)
boxplots_example2_d3 <- debounce(boxplots_example2_rctv, 450)
boxplots_example2_d4 <- debounce(boxplots_example2_rctv, 550)

observeEvent(input$boxplots_example_2, {
  updateTabsetPanel(
    session = session,
    inputId = "tabs",
    selected = "Boxplots"
  )
})

observeEvent(boxplots_example2_d1(), {
  updateRadioGroupButtons(
    session = session,
    inputId = "boxplots_metric", 
    selected = "error"
  )
  updateSelectizeInput(
    session = session,
    inputId = "boxplots_pdf",
    selected = c("gamma_1", "gamma_2")
  )
  updateSelectInput(
    session = session,
    inputId = "boxplots_estimator",
    selected = "adaptive_gaussian"
  )
  updateSelectInput(
    session = session,
    inputId = "boxplots_facet_vars",
    selected = c("density", "bw")
  )
  updateCheckboxInput(
    session = session,
    inputId = "boxplots_log10",
    value = FALSE
  )
  updateCheckboxInput(
    session = session,
    inputId = "boxplots_free_y",
    value = TRUE
  )
  updateSliderInput(
    session = session,
    inputId = "boxplots_trim_pct",
    value = 5
  )
})

observeEvent(boxplots_example2_d2(), {
  updateSelectInput(
    session = session,
    inputId = "boxplots_bw",
    selected = c("silverman", "isj", "experimental")
  )
})

observeEvent(boxplots_example2_d3(), {
  updateCheckboxGroupButtons(
    session = session,
    inputId = "boxplots_size",
    selected = as.character(choices_size_default[-1])
  )
})

observeEvent(boxplots_example2_d4(), {
  click("boxplots_plot_btn")
})


# Density examples -------------------------------------------------------------

# Disable links if density tab is not available

output$density_examples_links <- renderUI({
  if (store$PYTHON_LOADED) {
    tagList(
      tags$ul(
        tags$li(
          actionLink(
            inputId = "density_example_1", 
            label = "Example 1: Gamma distribution estimated via Gaussian KDE."),
        ),
        tags$li(
          actionLink(
            inputId = "density_example_2", 
            label = "Example 2: Mixture of Beta and Gaussian estimated via Adaptive Gaussian KDE.")
          )
      )
    )
  } else {
    p("Density tab not available.")
  }
})

# Example 1
density_example1_rctv <- reactive(input$density_example_1)
density_example1_d1 <- debounce(density_example1_rctv, 400)
density_example1_d2 <- debounce(density_example1_rctv, 800)
density_example1_d3 <- debounce(density_example1_rctv, 1200)
density_example1_d4 <- debounce(density_example1_rctv, 1500)


observeEvent(input$density_example_1, {
  updateTabsetPanel(
    session = session,
    inputId = "tabs",
    selected = "Density plots"
  )
})

observeEvent(density_example1_d1(), {
  updateRadioGroupButtons(
    session = session,
    inputId = "density_dist_type", 
    selected = "fixed"
  )
})

observeEvent(density_example1_d2(), {
  show_spinner()
  updateSelectInput(
    session = session,
    inputId = "density_fixed_distribution", 
    selected = "gamma"
  )
  updateNumericInput(
    session = session,
    inputId = "density_sample_size",
    value = 600,
  )
  updateSelectInput(
    session = session,
    inputId = "density_estimator",
    selected = "gaussian_kde",
  )
  updateSelectInput(
    session = session,
    inputId = "density_bw_method",
    selected = "scott",
  )
  updateCheckboxInput(
    session = session,
    inputId = "density_extend_limits",
    value = FALSE
  )
  updateCheckboxInput(
    session = session,
    inputId = "density_bound_correction",
    value = TRUE
  )
})

observeEvent(density_example1_d3(), {
  updateNumericInput(
    session = session,
    inputId = "fixed_params_input1",
    value = 2,
  )
  updateNumericInput(
    session = session,
    inputId = "fixed_params_input2",
    value = 1.5,
  )
})

observeEvent(density_example1_d4(), {
  click("density_plot_btn")
})

# Example 2
density_example2_rctv <- reactive(input$density_example_2)
density_example2_d1 <- debounce(density_example2_rctv, 300)
density_example2_d2 <- debounce(density_example2_rctv, 700)
density_example2_d3 <- debounce(density_example2_rctv, 1100)
density_example2_d4 <- debounce(density_example2_rctv, 1500)
density_example2_d5 <- debounce(density_example2_rctv, 1800)
density_example2_d6 <- debounce(density_example2_rctv, 2100)

observeEvent(input$density_example_2, {
  updateTabsetPanel(
    session = session,
    inputId = "tabs",
    selected = "Density plots"
  )
})

observeEvent(density_example2_d1(), {
  show_spinner()
  updateRadioGroupButtons(
    session = session,
    inputId = "density_dist_type", 
    selected = "mixture"
  )
  
  updateNumericInput(
    session = session,
    inputId = "density_sample_size",
    value = 1500,
  )
  updateSelectInput(
    session = session,
    inputId = "density_estimator",
    selected = "adaptive_kde",
  )

  updateCheckboxInput(
    session = session,
    inputId = "density_extend_limits",
    value = FALSE
  )
  updateCheckboxInput(
    session = session,
    inputId = "density_bound_correction",
    value = FALSE
  )
  click("density_open_settings")
})

observeEvent(density_example2_d2(), {
  updateSliderInput(
    session = session,
    inputId = "density_mixture_n",
    value = 2
  )
})

observeEvent(density_example2_d3(), {
  updateSliderInput(
    session = session,
    inputId = "density_mixture_n",
    value = 3
  )
})


observeEvent(density_example2_d4(), {
  updateSelectInput(
    session = session, 
    inputId = "mixture_component1",
    selected = "beta"
  )
  updateSelectInput(
    session = session, 
    inputId = "mixture_component2",
    selected = "beta"
  )
  updateSelectInput(
    session = session, 
    inputId = "mixture_component3",
    selected = "norm"
  )
  updateSelectInput(
    session = session,
    inputId = "density_bw_method",
    selected = "experimental",
  )
})

observeEvent(density_example2_d5(), {
  updateNumericInput(
    session = session, 
    inputId = "mixture_component1_input1",
    value = 8
  )
  updateNumericInput(
    session = session, 
    inputId = "mixture_component1_input2",
    value = 1.5
  )
  updateNumericInput(
    session = session, 
    inputId = "mixture_component1_input_wt",
    value = 0.45
  )
  
  updateNumericInput(
    session = session, 
    inputId = "mixture_component2_input1",
    value = 1.5
  )
  updateNumericInput(
    session = session, 
    inputId = "mixture_component2_input2",
    value = 8
  )
  updateNumericInput(
    session = session, 
    inputId = "mixture_component2_input_wt",
    value = 0.45
  )
  
  updateNumericInput(
    session = session, 
    inputId = "mixture_component3_input1",
    value = 0.5
  )
  updateNumericInput(
    session = session, 
    inputId = "mixture_component3_input2",
    value = 0.05
  )
  updateNumericInput(
    session = session, 
    inputId = "mixture_component3_input_wt",
    value = 0.10
  )
})

observeEvent(density_example2_d6(), {
  click("density_plot_btn")
})
