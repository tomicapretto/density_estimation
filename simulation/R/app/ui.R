source("ui/helpers.R")
library(shinyWidgets)

fluidPage(

  titlePanel("App title"),
  
  sidebarLayout(
    mySidebarPanel(),
    myMainPanel()
  )
)