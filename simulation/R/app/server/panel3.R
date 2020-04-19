# Fixed and mixture densities specifications -----------------------------------
update_mixture_ui <- update_mixture_ui_gen()

observeEvent(input$density_dist_type, {
  if (input$density_dist_type == "mixture") {
    hide(id = "density_distribution_fixed_ui")
    show(id = "density_distribution_mix_ui")
  } else {
    hide(id = "density_distribution_mix_ui")
    show(id = "density_distribution_fixed_ui")
  }
})

output$density_distribution_ui <- renderUI({
  tagList(
    div(
      id = "density_distribution_fixed_ui",
      selectInput(
        inputId = "density_fixed_distribution", 
        label = "Select a distribution",
        choices = c("Normal" = "norm",
                    "T-Student" = "t",
                    "Gamma" = "gamma",
                    "Exponential" = "exp",
                    "Beta" = "beta",
                    "Log Normal" = "lnorm",
                    "Weibull" = "weibull",
                    "Uniform" = "unif")
      ),
      uiOutput("density_fixed_params_ui")
    ),
    hidden(
      div(
        id = "density_distribution_mix_ui",
        dropdownButton(
          inputId = "density_open_settings",
          label = "Settings",
          icon = icon("sliders"),
          status = "primary",
          circle = FALSE,
          width = "500px",
          
          sliderInput(
            inputId = "density_mixture_n",
            label = "Number of components",
            min = 1, max = 3, value = 1, step = 1 
          )
        ),
        br(),
        verbatimTextOutput("mix_msg")
      )
    )
  )
})

output$density_fixed_params_ui <- renderUI({
  tagList(
    distribution_parameters_cont(input$density_fixed_distribution)
  )
})

observeEvent(input$density_mixture_n, {
  update_mixture_ui(
    input = input,
    output = output,
    n_new = input$density_mixture_n
  )
})

output$mix_msg <- renderText(
  mixture_message(input)
)

# Density estimator specifications ---------------------------------------------
output$density_bw_method_ui <- renderUI({
  if (input$density_estimator %in% c("gaussian_kde", "adaptive_kde")) {
    selectInput(
      inputId = "density_bw_method",
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
      inputId = "density_bw_method",
      label = "Bandwidth method",
      choices = c("Mixture" = "mixture")
    )
  }
})

observeEvent(input$density_plot_btn, {
  if (input$density_sample_size <= 100000) {
    if (input$density_bw_method %in% c("lscv", "sj")) {
      if (input$density_sample_size >= 5000) {
        showNotification(
          ui = HTML(paste(c("Sample size is too large for the chosen method.",
                            "Computation is not performed."), collapse = "<br/>")),
          type = "error"
        )
      } else if (input$density_sample_size >= 2000) {
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

# Plotting section -------------------------------------------------------------
output$density_plot <- renderPlot({
  req(store$density_plot)
  store$density_plot()
})

output$density_plot_ui <- renderUI({
  plotOutput(
    "density_plot",
    height = input$dimension[2] * 0.70,
    width = "95%"
  )
})

# Download section -------------------------------------------------------------
output$density_download_plot_ui <- renderUI({
  req(store$density_plot)
  tagList(
    h4("File name"),
    fluidRow(
      column(
        width = 3, 
        textInput(
          inputId = "density_plot_filename", 
          label = NULL, 
          value = NULL,
          placeholder = "Optional"
        )
      ),
      column(
        width = 3,
        downloadButton(
          outputId = "density_download_plot", 
          label = "Save plot")
      )
    )
  )
})

output$density_download_plot <- downloadHandler(
  filename = function() {
    if (is.null(input$density_plot_filename)) {
      name <- paste0("plot", count_plot_dens())
    } else if (input$density_plot_filename == "") {
      name <- paste0("plot", count_plot_dens())
    } else {
      name <- gsub(" ", "_", input$density_plot_filename)
    }
    paste0(name, '.png')
  },
  
  content = function(file) {
    png(
      file, 
      width = 3 * input$dimension[1] * 0.70, 
      height = 3 * input$dimension[2] * 0.70, 
      res = 240)
    store$density_plot()
    dev.off()
  }
)


