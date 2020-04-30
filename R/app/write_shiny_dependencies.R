# Write dependency so shinyapps.io detects which packages to use.
# pkgs <- setdiff(packages, "reticulate")
file_connection <- file("shiny_dependencies.R")
writeLines(paste0("library(", PACKAGES_REQ, ")"), file_connection)
close(file_connection)