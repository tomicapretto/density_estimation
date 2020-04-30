source("server/helpers_mixture_modal.R")

# Initialize python ------------------------------------------------------------
use_python_custom = function(path) {
  tryCatch({
    reticulate::use_python(
      python = path
    )
    suppressWarnings(invisible(reticulate::py_config()))
    path_found = dirname(reticulate:::.globals$py_config$python)
    if (path != path_found) {
      showNotification(
        paste(
          stringi::stri_wrap(
            paste(
              "Python was found in", path_found, "which differs from your input."
            ), 
            width = 30),
          collapse = "\n"
        ),
        type = "warning", 
        duration = 7
      )
    }
    showNotification(
      paste("Python version:", reticulate:::.globals$py_config$version),
      type = "message",
      duration = 7
    )
    return(path_found)
  }, 
  error = function(c) {
    showNotification(
      paste("The following error was thrown when trying to load Python<\br>", 
            c),
      type = "error",
      duration = 7
    )
  })
}

init_python_custom <- function(input) {
  withProgress(
    message = "Getting Python ready", 
    expr = {
      incProgress(0, detail = "Configuring Python...")
      PYTHON_PATH <- use_python_custom(input$python_path)
      incProgress(0.33, detail = "Checking packages...")
      
      tryCatch({
        msg <- reticulate::py_capture_output(
          reticulate::source_python("python/check_pkgs.py"))
        showNotification(
          HTML(gsub("\n","<br/>", msg)),
          type = "message",
          duration = 7)
      },
      error = function(c) {
        showNotification(
          HTML(paste("Not all required packages have been found<br/>", c)),
          type = "error",
          duration = 7
        )
        return(invisible(NULL))
      })
      incProgress(0.33, detail = "Loading functions...")
      reticulate::source_python("python/density_utils.py", envir = globalenv())
      incProgress(0.34, detail = "Done!")
    }
  )
  return(invisible(PYTHON_PATH))
}

init_python_shiny <- function(input) {
  tryCatch({
    withProgress(
      message = "Getting Python ready", 
      expr = {
        
        incProgress(0, detail = "Creating virtual environment...")
        reticulate::virtualenv_create(
          envname = "python35_env", 
          python = "/usr/bin/python3")
        
        incProgress(0.2, detail = "Installing packages...")
        reticulate::virtualenv_install(
          envname = "python35_env", 
          packages = c("numpy", "scipy"))
        
        incProgress(0.2, detail = "Loading virtual environment...")
        reticulate::use_virtualenv(
          virtualenv = "python35_env", 
          required = TRUE)
        
        incProgress(0.2, detail = "Loading functions...")
        reticulate::source_python(
          file = "python/density_utils.py", 
          envir = globalenv())
        incProgress(0.2, detail = "Done!")
      }
    )
    return(TRUE)
  },
  error = function(c){
    showNotification(
      HTML(paste(
        stringi::stri_wrap(paste("Python could not be loaded.<br/>Error:", c), 
                           width = 30), collapse = "<br/>")),
      type = "error", 
      duration = 7
    )
    return(FALSE)
  })
}

# Restart session --------------------------------------------------------------
restart_r <- function() if (tolower(.Platform$GUI) == "rstudio") {
  # rm(list = ls())
  # .rs.restartR()
}

# Data wrangling functions -----------------------------------------------------
df_filter <- function(df, .pdf, .estimator, .bw, .size) {
  df %>%
    filter(
      pdf %in% .pdf,
      estimator %in% .estimator,
      bw %in% .bw,
      size %in% .size
    )
}

df_trim <- function(df, .group_vars, .trim_var, .quantile) {
  
  .group_vars <- purrr::map(.group_vars, as.name)
  .trim_var <- as.name(.trim_var)
  
  df %>%
    group_by(!!!.group_vars) %>%
    mutate(
      upper_bound = quantile(!!.trim_var, .quantile, na.rm = TRUE)
    ) %>%
    mutate(
      !!.trim_var := ifelse(!!.trim_var >= upper_bound, NA, !!.trim_var)
    ) %>%
    select(-upper_bound)
}

deduce_scale <- function(x) {
  if ((quantile(x, 0.75, na.rm = TRUE)) > 1) {
    return("sec") 
  } else {
    return("ms")
  }
}

# Specify accuracy by default
precision <- scales:::precision

# Plotting functions -----------------------------------------------------------
initialize_plot <- function(df, .metric) {
  .metric <- as.name(.metric)
  ggplot(
    df,
    aes(
        x = factor(size),
        y = !! .metric, 
        fill = factor(size)
    )
  )
}

add_boxplot <- function(outlier.shape = 19) {
  geom_boxplot(
    outlier.fill = alpha(DARK_GRAY, 0.3),
    outlier.color = alpha(DARK_GRAY, 0.3),
    outlier.shape = outlier.shape
  )
}

custom_fill <- function(alpha = 0.6) {
  scale_fill_viridis_d(alpha = alpha)
}

custom_theme <- function() {
  theme_gray() + 
    theme(
      panel.grid.minor.x = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.border = element_blank(), 
      legend.position = "none",
      strip.background = element_blank()
    )
}

custom_scale <- function(limits = NULL,  breaks = waiver(),
                         scale = "ms", acc = 0.1, log10 = FALSE) {
  scale_func <- if (log10) scale_y_log10 else scale_y_continuous
  scale_func(
    limits = limits,
    breaks = breaks,
    labels = get_label(scale, acc)
  )
}

custom_facet <- function(..., free_y, nrow = 1) {
  
  args <- as.list(...)
  
  if (!any(args %in% c("bw", "estimator", "pdf"))) return(NULL)
  
  scales <- if (free_y) "free_y" else "fixed"
  
  if (length(args) == 1) {
    col_labs <- lbls_list[[args[[1]]]]
    
    form <- as.formula(paste(".", args[[1]], sep = "~"))
    facet_grid(form, scales = scales, 
               labeller = labeller(.cols = col_labs))
    
  } else if (length(args) == 2) {
    row_labs <- lbls_list[[args[[1]]]]
    col_labs <- lbls_list[[args[[2]]]]
    
    form <- as.formula(paste(args[[1]], args[[2]], sep = "~"))
    facet_grid(form, scales = scales, 
               labeller = labeller(.cols = col_labs, .rows = row_labs))
  }
}

get_bw_choices <- function(x) {
  f <- function (x) switch(x, "mixture" = choices_bw_mixture, choices_bw_classic)
  vec <- unlist(lapply(x, f)) 
  vec[!duplicated(vec)]
}

get_size_choices <- function(x) {
  f <- function (x) switch(x, "sj" = choices_size_sj, choices_size_default)
  l <- lapply(x, f)
  Reduce(union, l)
}

miliseconds <- function(acc = 1) {
  scales::number_format(
    accuracy = acc,
    scale = 1000,
    suffix = " ms")
}

seconds <- function(acc = 1) {
  scales::number_format(
    accuracy = acc,
    scale = 1,
    suffix = " secs")
}

get_label <- function(scale, acc) {
  switch(scale, 
         ms = miliseconds(acc),
         sec = seconds(acc),
         scales::number_format(acc))
}

# Continuous variables arguments -----------------------------------------------
cont_dist_labels <- list(
  norm = c("Mean", "Standard Deviation"),
  t = c("Degrees of freedom", "NCP"),
  gamma = c("Shape", "Scale"),
  exp = c("Rate"),
  beta = c("Shape 1", "Shape 2"),
  lnorm = c("Log mean", "Log SD"),
  weibull = c("Shape", "Scale"),
  unif = c("Lower bound", "Upper bound")
)

cont_dist_values <- list(
  norm = 0:1,
  t = 1:0,
  gamma = c(1, 1),
  exp = 1,
  beta = c(1.5, 1.5),
  lnorm = 0:1,
  weibull = c(0.5, 1),
  unif = c(-1, 1)
)

cont_dist_mins <- list(
  norm = c(-100, 0),
  t = 1:0,
  gamma = c(0, 0),
  exp = 0,
  beta = c(0, 0),
  lnorm = c(0, 0),
  weibull = c(0, 0),
  unif = c(-100, -100)
)

cont_dist_mins_check <- list(
  norm = c(-100, 0.001),
  t = 1:0,
  gamma = c(0, 0.001),
  exp = 0.001,
  beta = c(0.01, 0.01),
  lnorm = c(0.01, 0.01),
  weibull = c(0.01, 0.01),
  unif = c(-100, -100)
)

cont_dist_maxs <- list(
  norm = c(100, 100),
  t = c(200, 50),
  gamma = c(100, 100),
  exp = 100,
  beta = c(100, 100),
  lnorm = c(100, 100),
  weibull = c(100, 100),
  unif = c(100, 100)
)

cont_dist_bounds <- list(
  "norm" = function(.params) {
    width <- 3 * .params[[2]]
    c(.params[[1]] - width, .params[[1]] + width)
  },
  "t" = function(.params) {
    qt(c(0.005, 0.995), .params[[1]], .params[[2]])
  },
  "gamma" = function(.params) {
    c(0, qgamma(0.995, .params[[1]], .params[[2]]))
  },
  "exp" = function(.params) {
    c(0, qexp(0.995, .params[[1]]))
  },
  "beta" = function(.params) {
    c(0, 1)
  },
  "lnorm" = function(.params) {
    c(0, qlnorm(0.995, .params[[1]], .params[[2]]))
  },
  "weibull" = function(.params) {
    c(0, qweibull(0.995, .params[[1]], .params[[2]]))
  },
  "unif" = function(.params) {
    c(.params[[1]], .params[[2]])
  }
)

all_equal_length <- function(...) {
  length(unique(vapply(list(...), length, 1L))) == 1
}

numeric_params <- function(label, value, min = NA, max = NA, step = 0.5, prefix) {
  if (all(is.na(min))) min <- rep(NA, length(label))
  if (all(is.na(max))) max <- rep(NA, length(label))
  step <- rep(step, length(label))
  
  stopifnot(all_equal_length(label, value, min, max, step))
  ids <- paste0(prefix, "_input", seq_along(label))
  out <- purrr::pmap(list(ids, label, value, min, max, step), numericInput)
  purrr::iwalk(out, `[[`)
}

distribution_parameters_cont <- function(distribution, prefix = "fixed_params") {
  numeric_params(
    label = cont_dist_labels[[distribution]],
    value = cont_dist_values[[distribution]],
    min = cont_dist_mins[[distribution]],
    max = cont_dist_maxs[[distribution]],
    prefix = prefix
  ) 
} 

custom_column <- function(id, label, value, min, max, step) {
  column(
    width = 4,
    numericInput(
      inputId = id,
      label = label,
      value = value,
      min = min,
      max = max,
      step = step,
      width = "100%"
    )
  )
}

generate_mixture_ui <- function(label, value, min = NA, max = NA, step = NA, prefix) {
  if (all(is.na(min))) min <- rep(NA, length(label))
  if (all(is.na(max))) max <- rep(NA, length(label))
  if (all(is.na(step))) step <- rep(NA, length(label))
  
  stopifnot(all_equal_length(label, value, min, max, step))
  ids <- paste0(prefix, "_input", seq_along(label))
  out <- purrr::pmap(list(ids, label, value, min, max, step), custom_column)
  purrr::iwalk(out, `[[`)
}

render_mixture_ui <- function() {
  distribution = "rnorm"
  prefix = "mixture1"
  renderUI({
    fluidRow(
      generate_mixture_ui(
        label = cont_dist_labels[[distribution]],
        value = cont_dist_values[[distribution]],
        min = cont_dist_mins[[distribution]],
        max = cont_dist_maxs[[distribution]],
        prefix = prefix 
      )
    )
  })
}

# Plot functions ---------------------------------------------------------------
component_rvs <- function(distribution, .params, size) {
  .f <- paste0("r", distribution)
  .args <- c(list(size), .params)
  do.call(.f, .args)
}

mixture_rvs <- function(distributions, .params, size, wts = NULL) {
  if (is.null(wts)) wts <- rep(1 / length(distributions), length(distributions))
  stopifnot(length(distributions) == length(.params), length(.params) == length(wts))
  stopifnot(all.equal(sum(wts), 1, tolerance = 0.011))
  
  .l <- list(distributions, .params, round(wts * size))
  unlist(purrr::pmap(.l, component_rvs))
}

component_pdf <- function(distribution, .params, x_grid) {
  .f <- paste0("d", distribution)
  .args <- c(list(x_grid), .params)
  do.call(.f, .args)
}

pdf_bounds <- function(distribution, .params) {
  .f <- cont_dist_bounds[[distribution]]
  .f(.params)
}

mixture_grid <- function(distributions, .params) {
  .l <- list(distributions, .params)
  out <- unlist(purrr::pmap(.l, pdf_bounds))
  seq(min(out), max(out), length.out = 250)
}

mixture_pdf <- function(distributions, .params, wts = NULL) {
  if (is.null(wts)) wts <- rep(1 / length(distributions), length(distributions))
  stopifnot(length(distributions) == length(.params), length(.params) == length(wts))
  stopifnot(all.equal(sum(wts), 1, tolerance = 0.011))

  x_grid <- mixture_grid(distributions, .params)
  
  .l <- list(distributions, .params)
  pdf <- unlist(purrr::pmap(.l, component_pdf, x_grid = x_grid))
  pdf <- as.vector(matrix(pdf, ncol = length(wts)) %*% wts)
  pdf[is.infinite(pdf)] <- NA # In some edge cases pdf is `Inf`
  return(list("x" = x_grid, "pdf" = pdf))
}

get_mixture_distributions <- function(input, mixture_n) {
  vapply(
    X = paste0("mixture_component", seq_len(mixture_n)),
    FUN = function(x) input[[x]],
    FUN.VALUE = character(1),
    USE.NAMES = FALSE
  )
}

get_mixture_params <- function(input, mixture_n) {
  # Starts with mixture_component1_input and ends with digit
  pattern <- paste0("(mixture_component", seq_len(mixture_n), "_input)(\\d$)")
  lapply(
    X = pattern,
    FUN = function(x) {
      .nms <- names(input)[grepl(x, names(input))]
      # Names are not sorted in `input`. Not very solid, but worked so far
      .nms <- stringr::str_sort(.nms)
      # Reactive values allow subsetting only one element
      out <- vector("numeric", length(.nms))
      for (i in seq_along(.nms)) {
        out[[i]] <- input[[.nms[[i]]]]
      }
      out
    }
  )
}

get_mixture_wts <- function(input, mixture_n) {
  vapply(
    X = paste0("mixture_component", seq_len(mixture_n), "_input_wt"), 
    FUN = function(x) input[[x]],
    FUN.VALUE = numeric(1), 
    USE.NAMES = FALSE
  )
}

check_mixture_wts_gen <- function() {
  wts_previous <- 1
  f <- function(wts, mixture_n) {
    if (any(is.na(wts))) {
      wts <- round(rep(1 / mixture_n, mixture_n), 3)
    } else if (!((sum(wts) > 1 - 0.0011) && (sum(wts) < 1 + 0.0011))) {
      changed <- length(wts) != length(wts_previous) | !all(wts == wts_previous)
      if (changed) {
        showNotification(
          ui = HTML(paste(c("Sum of weights is not equal to 1.",
                            "Weighting all components equally."), 
                          collapse = "<br/>")),
          type = "warning"
        )
        wts_previous <<- wts
      }
      wts <- round(rep(1 / mixture_n, mixture_n), 3)
    }
    wts
  }
  return(f)
}

check_mixture_wts <- check_mixture_wts_gen()

stop_custom <- function(.subclass, message, call = NULL, ...) {
  .error <- structure(
    list(
      message = message,
      call = call,
      ...
    ),
    class = c(.subclass, "error", "condition")
  )
  stop(.error)
}

check_density_args_inner <- function(.params, mins, maxs) {
  vapply(
    X = seq_along(.params), 
    FUN = function(i) .params[[i]] >= mins[[i]] & .params[[i]] <= maxs[[i]],
    FUN.VALUE = logical(1),
    USE.NAMES = FALSE
  )
}

check_density_args <- function(distributions, .params) {
  mins <- purrr::map(distributions, .f = function(x) cont_dist_mins_check[[x]])
  maxs <- purrr::map(distributions, .f = function(x) cont_dist_maxs[[x]])
  
  lgls <- unlist(purrr::pmap(list(.params, mins, maxs), check_density_args_inner))
  
  if (any(!lgls)) {
    stop_custom(
      .subclass = "bad-parameters", 
      message = paste(
        "At least one parameter value is not supported.",
        "Possible causes:",
        "* Very low/high values.",
        "* Value not in domain of parameter (i.e. negative SD).", 
        sep = "\n")
      )
  }
  return(invisible(NULL))
}

get_density_params <- function(input) {
  if (input$density_dist_type == "mixture") {
    
    if (is.null(input[["mixture_component1_input2"]])) {
      stop_custom(
        .subclass = "missing-component",
        message = "Specify at least one component for the mixture"
      )
    }
    
    mixture_n <- input$density_mixture_n
    distributions <- get_mixture_distributions(input, mixture_n)
    .params <- get_mixture_params(input, mixture_n)
    
    check_density_args(distributions, .params)
  
    wts <- get_mixture_wts(input, mixture_n)
    wts <- check_mixture_wts(wts, mixture_n)
    rvs <- mixture_rvs(distributions, .params, input$density_sample_size, wts)
    pdf <- mixture_pdf(distributions, .params, wts)
    
  } else {
    distribution <- input$density_fixed_distribution
    dist_params <- if (distribution == "exp") {
      list(input$fixed_params_input1)
    } else {
      list(input$fixed_params_input1, input$fixed_params_input2)
    }
    
    check_density_args(distribution, list(dist_params))
    rvs <- mixture_rvs(distribution, list(dist_params), input$density_sample_size, wts = 1)
    pdf <- mixture_pdf(distribution, list(dist_params), wts = 1)
  }
  
  x_true <- pdf$x
  y_true <- pdf$pdf
  x_range <- range(x_true)
  y_range <- c(0, 1.16 * max(y_true, na.rm = TRUE))
  
  if (input$density_estimator == "gaussian_kde") {
    estimation <- estimate_density(
      rvs, 
      bw = input$density_bw_method, 
      extend = input$density_extend_limits,
      bound_correction = input$density_bound_correction
    )
  } else if (input$density_estimator == "adaptive_kde") {
    estimation <- estimate_density(
      rvs, 
      bw = input$density_bw_method, 
      extend = input$density_extend_limits,
      bound_correction = input$density_bound_correction, 
      adaptive = TRUE
    )
  } else {
    estimation <- estimate_density_em(
      rvs, 
      extend = input$density_extend_limits
    )
  }
  
  params = list("rvs" = rvs, "x_range" = x_range, "y_range" = y_range,
                "x_estimation" = estimation[[1]], "y_estimation" = estimation[[2]],
                "x_true" = x_true, "y_true" = y_true)
  return(params)
}

add_hist = function(x, x_range, y_range, breaks_n = 40) {
  hist(x, breaks = breaks_n, prob = TRUE, 
       main = NULL,
       xlab = "X", 
       ylab = "Probabilty density function", 
       xlim = x_range,
       ylim = y_range,
       col = LIGHT_BLUE
  )
}

add_lines = function(x_estimation, y_estimation, x_true, y_true) {
  lines(x_true, y_true, lwd = 4, col = "black", lty = "dashed")
  lines(x_estimation, y_estimation, lwd = 5, col = DARK_RED)
}

density_plot_generator <- function(params) {
  rvs = params[["rvs"]]
  x_range = params[["x_range"]]
  y_range = params[["y_range"]]
  x_estimation = params[["x_estimation"]]
  y_estimation = params[["y_estimation"]]
  x_true = params[["x_true"]]
  y_true = params[["y_true"]]
  
  f <- function() {
    add_hist(rvs, x_range, y_range)
    add_lines(x_estimation, y_estimation, x_true, y_true)
    legend("topright", legend = c("True density", "Estimation"), 
           col = c("black", DARK_RED), lwd = 5, inset = 0.015)
  }
  return(f)
}

get_density_plot <- function(input) {
  show_spinner()
  density_plot_params <- get_density_params(input)
  density_plot <- density_plot_generator(density_plot_params)
  hide_spinner()
  return(density_plot)
}

