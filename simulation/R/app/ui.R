source("utils.R")

pkgs <- c("shiny", "shinyBS", "shinyWidgets", "ggplot2", "dplyr")
check_pkgs(pkgs)

source("ui/helpers.R")

fluidPage(

  titlePanel("Simulation results explorer"),
  
  sidebarLayout(
    mySidebarPanel(),
    myMainPanel()
  )
)