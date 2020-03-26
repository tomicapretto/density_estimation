# Comment when not deploying on server
# library(shiny)
# library(shinyBS)
# library(shinyWidgets)
# library(ggplot2)
# library(dplyr)

source("utils.R")

# Uncomment when running locally
pkgs <- c("shinyBS", "shinyWidgets", "ggplot2", "dplyr")
check_pkgs(pkgs)

source("ui/helpers.R")

fluidPage(
  titlePanel("Simulation results explorer"),
  sidebarLayout(
    mySidebarPanel(),
    myMainPanel()
  )
)