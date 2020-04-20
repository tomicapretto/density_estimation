source("utils.R")
packages <- c("shiny", "shinyBS", "shinyWidgets", "shinyjs", 
              "shinybusy", "ggplot2", "dplyr", "reticulate", 
              "stringr", "stringi", "purrr")

# Write dependency file to be used by shinyapps.io to detect
# which packages to use.
pkgs <- setdiff(packages, "reticulate")
file_connection <- file("shiny_dependencies.R")
writeLines(paste0("library(", packages, ")"), file_connection)
close(file_connection)

init_packages(packages)

tagList(
  # Capture window size, used to save plots.
  tags$head(
    tags$script(window_capture_script),
  ),  
  # Render Latex
  tags$head(
    tags$link(
      rel = "stylesheet", 
      href = "https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.css", 
      integrity = "sha384-9tPv11A+glH/on/wEu99NVwDPwkMQESOocs/ZGXPoIiLE8MU/qkqUcZ3zzL+6DuH", 
      crossorigin = "anonymous"
    ),
    tags$script(
      src = "https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.js", 
      integrity = "sha384-U8Vrjwb8fuHMt6ewaCy8uqeUXv4oitYACKdB0VziCerzt011iQ/0TqlSlv8MReCm", 
      crossorigin = "anonymous"
    )
  ),
  useShinyjs(),
  use_busy_spinner(spin = "fading-circle"),
  hidden(
    div(
      id = "mainLayout",
      navbarPage(
        "Explorer!",
        id = "tabs",
        source("ui/panel1.R", local = TRUE)$value,
        source("ui/panel2.R", local = TRUE)$value,
        source("ui/panel3.R", local = TRUE)$value
      )
    )
  )
)
