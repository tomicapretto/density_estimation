# Colors -----------------------------------------------------------------------
DARK_GRAY <- "#2d3436"
DARK_RED = "#c0392b"
DARK_BLUE = "#2980b9"
LIGHT_BLUE = "#56B4E9"

# Auxiliary scripts ------------------------------------------------------------
window_capture_script <- '
var dimension = [0, 0];
$(document).on("shiny:connected", function(e) {
  dimension[0] = window.innerWidth;
  dimension[1] = window.innerHeight;
  Shiny.onInputChange("dimension", dimension);
});
$(window).resize(function(e) {
  dimension[0] = window.innerWidth;
  dimension[1] = window.innerHeight;
  Shiny.onInputChange("dimension", dimension);
});'

latex_input_script <- "
{
item:   function(item, escape) { 
  var html = katex.renderToString(item.label);
  return '<div>' + html + '</div>'; 
},
option: function(item, escape) { 
  var html = katex.renderToString(item.label);
  return '<div>' + html + '</div>'; 
}
}
                              "
# Static vectors ---------------------------------------------------------------
# Pre-defined choices for input fields.
choices_bw_classic <- c(
  "Silverman's rule" = "silverman",
  "Scott's rule" = "scott",
  "Sheather-Jones" = "sj",
  "Improved Sheather-Jones" = "isj",
  "Experimental" = "experimental")
choices_bw_mixture <- c("Default" = "mixture")

choices_size_sj <- c(200, 500, 1000)
choices_size_default <- c(200, 500, 1000, 5000, 10000)

# Density options
pdfNames <- c(
  "gaussian_1", "gaussian_2", "gmixture_1", "gmixture_2",
  "gmixture_3", "gmixture_4", "gmixture_5", "gamma_1",
  "gamma_2", "beta_1", "logn_1")

pdfCodes <- c(
  "N(0,1)", "N(0, 2)", 
  "\\frac{1}{2}N(-12, \\frac{1}{2}) + \\frac{1}{2}N(12, \\frac{1}{2})", 
  "\\frac{1}{2} N(0, \\frac{1}{10}) + \\frac{1}{2} N(5, 1)",
  "\\frac{2}{3} N(0, 1) + \\frac{1}{3} N(0, 0.1)",
  "\\frac{3}{4} N(0, 1) + \\frac{1}{4} N(1.5, \\frac{1}{3})",
  "\\frac{3}{5} N(3.5, \\frac{1}{2}) + \\frac{2}{5} N(9, 1.5)",
  "\\Gamma (k = 1, \\theta = 1)",
  "\\Gamma (k = 2, \\theta = 1)",
  "\\beta (a = 2.5, b = 1.5)",
  "\\text{Log}N(0, 1)")

pdf_choices <- setNames(pdfNames, pdfCodes)

# Panel grid labels
bw_lbls <- c(
  "silverman" = "Silverman's rule",
  "scott" = "Scott's rule",
  "sj" = "Sheather-Jones",
  "isj" = "Improved Sheather-Jones",
  "experimental" ="Experimental",
  "default" = "Default")

estimator_lbls <- c(
  "fixed_gaussian" = "Gaussian KDE",
  "adaptive_gaussian" = "Adaptive Gaussian KDE",
  "mixture" = "Gaussian mixture via EM")

lbls_list <- list("bw" = bw_lbls, "estimator" = estimator_lbls)

# Misc -------------------------------------------------------------------------
plot_counter <- function() {
  i <- 0
  function() {
    i <<- i + 1
    i
  }
}

load_packages <- function(packages, quietly = FALSE) {
  lapply(packages, library, character.only = TRUE, quietly = quietly)
}

check_packages <- function(packages, load = FALSE, quietly = FALSE) {
  installed_packages <- rownames(installed.packages())
  if (length(setdiff(packages, installed_packages)) > 0) {
    install.packages(setdiff(packages, installed_packages))  
  }
  if (load) {
    load_packages(packages, quietly = quietly)
  }
}

init_packages <- function(packages) {
  sys_info <- Sys.info()
  if (sys_info[["user"]] == "shiny" | sys_info[["effective_user"]] == "shiny") {
    source("shiny_dependencies.R")
  } else {
    check_packages(packages)
    load_packages(setdiff(packages, "reticulate"))
  }
}

is_valid_path <- function(path) {
  if (is.null(path)) return(FALSE)
  if (grepl("^\\s*$", path)) return(FALSE)
  return(TRUE)
}

choose_directory = function(caption = "Select data directory") {
  if (exists('choose.dir', 'package:utils')) {
    choose.dir(caption = caption) 
  } else {
    tcltk::tk_choose.dir(caption = caption)
  }
}