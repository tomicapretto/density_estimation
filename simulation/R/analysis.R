# Import packages and settings -------------------------------------------------
library(readr)
library(dplyr)
library(ggplot2)

BLUE = "#3498db"
DARK_BLUE = "#2980b9"
NIGHT_BLUE = "#2c3e50"
DARK_RED = "#c0392b"
LIGHT_BLUE = "#56B4E9"
LIGHT_GRAY = "#474747"
DARK_GRAY <- "#2d3436"

trace(grDevices:::png, quote({
  if (missing(type) && missing(antialias)) {
    type <- "cairo-png"
    antialias <- "subpixel"
  }
}), print = FALSE)

# Custom functions -------------------------------------------------------------

# Label helper
miliseconds <- function(acc = 1) {
  scales::number_format(
    accuracy = acc,
    scale = 1000,
    suffix = " ms")
}

seconds <- function(acc = 1) {
  scales::number_format(
    accuracy = acc,
    scale = 1,
    suffix = " secs")
}

# Read data --------------------------------------------------------------------

output_files <- paste0("../output/", list.files("../output"))
data_all <- bind_rows(lapply(output_files, read_csv))

# Mutate data ------------------------------------------------------------------
# Create "cleaner" version of the interest variables `time` and `error`.

data_clean <- data_all %>%
  group_by(estimator, bw, size) %>%
  mutate(
    time_p975 = quantile(time, 0.975),
    error_p975 = quantile(error, 0.975, na.rm = TRUE)
  ) %>%
  mutate(
    time_clean = ifelse(time >= time_p975, NA, time),
    error_clean = ifelse(error >= error_p975, NA, error)
  )

# Generate plots ---------------------------------------------------------------

# Function to generate individual boxplots

custom_boxplot <- function(estimator_, bw_, metric, label_func = miliseconds, acc = 1, 
                           limits = NULL, log10 = FALSE, breaks = waiver()) {
  
  metric <- enquo(metric)
  
  plt <- data_clean %>%
    filter(estimator == estimator_ & bw == bw_) %>%
    ggplot(
      aes(
        x = factor(size),
        y = !!metric, 
        fill = factor(size)
      )
    ) +
    geom_boxplot(
      outlier.fill = alpha(DARK_GRAY, 0.3),
      outlier.color = alpha(DARK_GRAY, 0.3)
    ) +
    scale_fill_viridis_d(
      alpha = 0.6
    ) + 
    labs(
      x = "Sample size",
      y = NULL
    ) +
    theme_bw() + 
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.border = element_blank(), 
      legend.position = "none"
    )
  
  plt_scale <- if (log10) {
    scale_y_log10(
      limits = limits,
      breaks = breaks,
      labels = label_func(acc = acc)
    ) 
  } else {
    scale_y_continuous(
      limits = limits,
      breaks = breaks,
      labels = label_func(acc = acc)
    )
  }
  
  plt + plt_scale
}

# Generate individual boxplots -----
# Time 
estimator <- "fixed_gaussian"
bw <- "silverman"
custom_boxplot(estimator, bw, time, 
               acc = 0.1, limits = c(0.0005, 0.0016))

ggsave(glue::glue("plots/{estimator}_{bw}.png"), device = "png", 
       width = 25, height = 15, dpi = 320, units = "cm")

estimator <- "fixed_gaussian"
bw <- "scott"
custom_boxplot(estimator, bw, time, 
               acc = 0.1, limits = c(0.0005, 0.0016))

ggsave(glue::glue("plots/{estimator}_{bw}.png"), device = "png", 
       width = 25, height = 15, dpi = 320, units = "cm")

estimator <- "fixed_gaussian"
bw <- "sj"
custom_boxplot(estimator, bw, time, 
               limits = c(0.01, 4),label_func = seconds, acc = 0.01, 
               log10 = TRUE, breaks = c(0.01, 0.1, 1, 3)) 

ggsave(glue::glue("plots/{estimator}_{bw}.png"), device = "png", 
       width = 25, height = 15, dpi = 320, units = "cm")

estimator <- "fixed_gaussian"
bw <- "isj"
custom_boxplot(estimator, bw, time, 
               acc = 1, limits = c(0.002, 0.016))

ggsave(glue::glue("plots/{estimator}_{bw}.png"), device = "png", 
       width = 25, height = 15, dpi = 320, units = "cm")

estimator <- "fixed_gaussian"
bw <- "experimental"
custom_boxplot(estimator, bw, time,
               acc = 1, limits = c(0.002, 0.016))
ggsave(glue::glue("plots/{estimator}_{bw}.png"), device = "png", 
       width = 25, height = 15, dpi = 320, units = "cm")

estimator <- "adaptive_gaussian"
bw <- "silverman"
custom_boxplot(estimator, bw, time, 
               acc = 0.1, limits = c(0.0005, 0.08))

ggsave(glue::glue("plots/{estimator}_{bw}.png"), device = "png", 
       width = 25, height = 15, dpi = 320, units = "cm")

estimator <- "adaptive_gaussian"
bw <- "scott"
custom_boxplot(estimator, bw, time, 
               acc = 0.1, limits = c(0.0005, 0.08))

ggsave(glue::glue("plots/{estimator}_{bw}.png"), device = "png", 
       width = 25, height = 15, dpi = 320, units = "cm")

estimator <- "adaptive_gaussian"
bw <- "sj"
custom_boxplot(estimator, bw, time, 
               limits = c(0.01, 4),label_func = seconds, acc = 0.01, 
               log10 = TRUE, breaks = c(0.01, 0.1, 1, 3)) 

ggsave(glue::glue("plots/{estimator}_{bw}.png"), device = "png", 
       width = 25, height = 15, dpi = 320, units = "cm")

estimator <- "adaptive_gaussian"
bw <- "isj"
custom_boxplot(estimator, bw, time, 
               acc = 1, limits = c(0.002, 0.4))

ggsave(glue::glue("plots/{estimator}_{bw}.png"), device = "png", 
       width = 25, height = 15, dpi = 320, units = "cm")

estimator <- "adaptive_gaussian"
bw <- "experimental"
custom_boxplot(estimator, bw, 
               time, acc = 1, limits = c(0.002, 0.4))

ggsave(glue::glue("plots/{estimator}_{bw}.png"), device = "png", 
       width = 25, height = 15, dpi = 320, units = "cm")

estimator <- "mixture"
bw <- "mixture"
custom_boxplot(estimator, bw, time, 
               label_func = seconds, acc = 0.01, 
               log10 = TRUE)
ggsave(glue::glue("plots/{estimator}_{bw}.png"), device = "png", 
       width = 25, height = 15, dpi = 320, units = "cm")


# Hipotesis:
# Los boxplots que tienen la mediana tan abajo y luego cajas anchas son síntoma
# de distribuciones bimodales.
# Devuelta tiene que ver con que las distribuciones mas dificiles llevan mas tiempo.

# Ver correlacion entre tiempo  y error.
# Hipótesis: Los algoritmos que ajustan mediante algun metodo numerico demoran
# mas tiempo en ajustar cuando la distribucion es "mas dificil" (mas error).

# Error

# Need to construct new functions because it should be measured 
# on a per function basis.


