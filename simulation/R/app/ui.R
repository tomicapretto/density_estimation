
check_pkgs <- function(pkgs, quietly = FALSE) { 
  
  lapply(pkgs, FUN = function(x) {
    if (!require(x, character.only = TRUE, quietly = quietly)) 
      install.packages(x, dependencies = TRUE, 
                       repos = "http://cran.us.r-project.org")
    library(x, character.only = TRUE, quietly = quietly)
  })
  
}

pkgs <- c("shiny", "shinyBS", "shinyWidgets", "ggplot2", "dplyr")
check_pkgs(pkgs)

source("ui/helpers.R")
# library(shinyWidgets)

fluidPage(

  titlePanel("App title"),
  
  sidebarLayout(
    mySidebarPanel(),
    myMainPanel()
  )
)