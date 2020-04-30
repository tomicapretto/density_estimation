source("utils.R")
source("write_shiny_dependencies.R")
init_packages(PACKAGES_REQ, PACKAGES_LOAD)

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
