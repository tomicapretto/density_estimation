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

# Misc -------------------------------------------------------------------------
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
labels_cont <- list(
  norm = c("Mean", "Standard Deviation"),
  t = "Degrees of freedom",
  gamma = c("Shape", "Scale"),
  exp = c("Rate"),
  beta = c("Shape 1", "Shape 2"),
  lnorm = c("Log mean", "Log SD"),
  cauchy = c("Location", "Scale"),
  weibull = c("Shape", "Scale"),
  unif = c("Lower bound", "Upper bound")
)

values_cont <- list(
  norm = 0:1,
  t = 1,
  gamma = c(1, 1),
  exp = 1,
  beta = c(1.5, 1.5),
  lnorm = 0:1,
  cauchy = 0:1,
  weibull = c(0.5, 1),
  unif = c(-1, 1)
)

mins_cont <- list(
  norm = c(-100, 0.001),
  t = 1,
  gamma = c(0, 0.001),
  exp = 0.001,
  beta = c(0.001, 0.001),
  lnorm = c(0.001, 0.001),
  cauchy = c(-100, 0.001),
  weibull = c(0.001, 0.001),
  unif = c(-100, -100)
)

maxs_cont <- list(
  norm = c(100, 100),
  t = 200,
  gamma = c(100, 100),
  exp = 100,
  beta = c(100, 100),
  lnorm = c(100, 100),
  cauchy = c(100, 100),
  weibull = c(100, 100),
  unif = c(100, 100)
)

all_equal_length <- function(...) {
  length(unique(vapply(list(...), length, 1L))) == 1
}

numeric_params <- function(label, value, min = NA, max = NA, step = NA, prefix) {
  if (all(is.na(min))) min <- rep(NA, length(label))
  if (all(is.na(max))) max <- rep(NA, length(label))
  if (all(is.na(step))) step <- rep(NA, length(label))
  
  stopifnot(all_equal_length(label, value, min, max, step))
  ids <- paste0(prefix, "Input", seq_along(label))
  out <- purrr::pmap(list(ids, label, value, min, max, step), numericInput)
  purrr::iwalk(out, `[[`)
}

distribution_parameters_cont <- function(distribution, prefix = "contDist") {
  numeric_params(
    label = labels_cont[[distribution]],
    value = values_cont[[distribution]],
    min = mins_cont[[distribution]],
    max = maxs_cont[[distribution]],
    prefix = prefix
  ) 
} 

# Plot functions ---------------------------------------------------------------
get_density_params <- function(input) {
  
  dDist<- paste0("d", input$contDist)
  
  distParams <- switch(
    input$contDist,
    "norm"  = list(input$contDistInput1, input$contDistInput2),
    "t"     = list(input$contDistInput1),
    "gamma" = list(input$contDistInput1, input$contDistInput2),
    "exp"   = list(input$contDistInput1),
    "beta"  = list(input$contDistInput1, input$contDistInput2),
    "lnorm" = list(input$contDistInput1, input$contDistInput2),
    "cauchy" = list(input$contDistInput1, input$contDistInput2),
    "weibull" = list(input$contDistInput1, input$contDistInput2),
    "unif"  = list(input$contDistInput1, input$contDistInput2)
    )
  
  what <- paste0("r", input$contDist)
  rvs <- do.call(what, c(list(input$contSampleSize), distParams))
  
  if (input$densityEstimator == "gaussian_kde") {
    estimation <- estimate_density(
      rvs, 
      bw = input$bwMethod, 
      extend = input$extendLimits,
      bound_correction = input$boundCorrection
    )
  } else if (input$densityEstimator == "adaptive_kde") {
    estimation <- estimate_density(
      rvs, 
      bw = input$bwMethod, 
      extend = input$extendLimits,
      bound_correction = input$boundCorrection, 
      adaptive = TRUE
    )
  } else {
    estimation <- estimate_density_em(
      rvs, 
      extend = input$extendLimits
    )
  }
  
  what <- paste0("q", input$contDist)
  percentiles <- c(0.008, 0.992)
  distRange <- do.call(what, c(list(percentiles), distParams))
  
  x_true = seq(distRange[1], distRange[2], length.out = 250)
  y_true = do.call(dDist, c(list(x_true), distParams))
  
  x_range = c(distRange[1], distRange[2])
  y_range = c(0, 1.18 * max(y_true))
  
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
  rm(list = ls())
  .rs.restartR()
}
