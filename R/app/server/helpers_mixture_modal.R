mixture_comp_choices <- c(
  "Normal" = "norm",
  "T-Student" = "t",
  "Gamma" = "gamma",
  "Beta" = "beta",
  "Log Normal" = "lnorm"
)

custom_column <- function(id, label, value, min, max, step) {
  column(
    width = 4,
    numericInput(inputId = id, label = label, value = value,
                 min = min, max = max, step = step, width = "100%")
  )
}

mixture_params_ui <- function(label, value, min = NA, max = NA, step = 0.5, prefix, wt = NA) {
  
  if (all(is.na(min))) min <- rep(NA, length(label))
  if (all(is.na(max))) max <- rep(NA, length(label))
  step <- rep(step, length(label))
  stopifnot(all_equal_length(label, value, min, max, step))
  
  ids <- paste0(prefix, "_input", seq_along(label))
  
  # Append weight field
  label <- c(label, "Weight")
  value <- c(value, wt)
  min <- c(min, 0)
  max <- c(max, 1)
  step <- c(step, 0.1)
  ids  <- c(ids, paste0(prefix, "_input_wt"))
  
  purrr::pmap(list(ids, label, value, min, max, step), custom_column)
}

update_mixture_ui_gen <- function() {
  n_old <- 0
  f <- function(input, output, n_new) {
    # Enabled when we have to INSERT new UI elements.
    if (n_new > n_old) {
      idxs <- setdiff(seq_len(n_new), seq_len(n_old))
      
      for (idx in idxs) {
        helper_ui <- paste0("helper_ui", idx)
        component_id <- paste0("mixture_component", idx)
        last_id <- if (idx == 1) {
          "#density_mixture_n"
        } else {
          paste0("#mixture_component_", idx - 1, "_params_ui")
        }
        
        # selectInput must be rendered/inserted before
        # the options so the latter could be reactive on selectInput
        insertUI(
          selector = last_id,
          where = "afterEnd",
          ui = tags$div(
            id = paste0("mixture_component_", idx, "_distr_ui"),
            fluidRow(
              column(
                width = 4,
                selectInput(
                  inputId = component_id,
                  label = "Distribution",
                  choices = mixture_comp_choices,
                  width = "100%"
                )  
              )
            )
          )
        )
        # First render input UI
        output[[helper_ui]] <- renderUI({
          req(input[[component_id]])
          fluidRow(
            mixture_params_ui(
              label = cont_dist_labels[[input[[component_id]]]],
              value = cont_dist_values[[input[[component_id]]]],
              min = cont_dist_mins[[input[[component_id]]]],
              max = cont_dist_maxs[[input[[component_id]]]],
              prefix = component_id
            )
          )
        })
        # Then insert input UI 
        insertUI(
          selector = paste0("#mixture_component_", idx, "_distr_ui"),
          where = "afterEnd",
          ui = tags$div(
            id = paste0("mixture_component_", idx, "_params_ui"),
            uiOutput(helper_ui)
          )
        )
      }
      n_old <<- n_new
    }
    # Enabled when we DELETE UI elements
    if (n_new < n_old) {
      idxs <- setdiff(seq_len(n_old), seq_len(n_new))
      for (idx in idxs) {
        removeUI(
          selector = paste0("#mixture_component_", idx, "_distr_ui")
        )
        removeUI(
          selector = paste0("#mixture_component_", idx, "_params_ui")
        )
      }
      n_old <<- n_new
    } 
  }
}

# Generate informative message -------------------------------------------------
mixture_message <- function(input) {
  
  mixture_n <- input$density_mixture_n
  req(input[[paste0("mixture_component", mixture_n, "_input2")]])
  
  comp_short <- c(
    "norm" = "N",
    "t" = "T",
    "gamma" = "Ga",
    "beta" = "Be",
    "lnorm" = "LogN"
  )
  
  # This process is being done twice, but the way I could avoid it
  # will make other more important functions look disgusting
  wts <- vapply(
    X = paste0("mixture_component", seq_len(mixture_n), "_input_wt"), 
    FUN = function(x) input[[x]],
    FUN.VALUE = numeric(1), 
    USE.NAMES = FALSE
  )
  
  if (any(is.na(wts))) {
    wts <- rep(1 / mixture_n, mixture_n)
  } else if (!((sum(wts) > 1 - 0.011) && (sum(wts) < 1 + 0.011))) {
    wts <- rep(1 / mixture_n, mixture_n)
  }
  wts <- round(wts, 2)
  .string_vec <- vector("character", mixture_n)
  for (idx in seq_len(mixture_n)) {
    input_name <- paste0("mixture_component", idx)
    req(input[[input_name]])
    comp_name <- comp_short[[input[[input_name]]]]
    param1 <- input[[paste0(input_name, "_input1")]]
    param2 <- input[[paste0(input_name, "_input2")]]
    wt <- wts[[idx]]
    .string_vec[[idx]] <- paste0(wt, "*", comp_name, "(", param1, ", ", param2, ")")
  }
  paste(.string_vec, collapse = " + ")
}


