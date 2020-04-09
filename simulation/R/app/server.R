# Load libraries ---------------------------------------------------------------
source("utils.R")
source("server/helpers.R")

# Init plot counter
count_plot <- plot_counter()
count_plot_dens <- plot_counter()

# Load data --------------------------------------------------------------------
df_static <- readRDS("data/data.rds")
# PYTHON_PATH = "C:/Users/Tomi/AppData/Local/Programs/Python/Python37"

# Create server function -------------------------------------------------------
function(input, output, session) {
  session$onSessionEnded(stopApp)
  session$onSessionEnded(restart_r)
  
  # Store object
  store <- reactiveValues()
  store$df_static <- df_static
  store$PYTHON_PATH <- NULL
  sys_info <- Sys.info()
  
  if (sys_info[["user"]] == "shiny" | sys_info[["effective_user"]] == "shiny") {
    source("server/welcome_shiny.R", local = TRUE)$value
  } else {
    source("server/welcome_custom.R", local = TRUE)$value
  }
}


