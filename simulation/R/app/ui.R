source("ui/helpers.R")

fluidPage(
  
  titlePanel("App title"),
  
  sidebarLayout(
    mySidebarPanel(),
    myMainPanel()
  )
)