 # Import packages --------------------------------------------------------------
library(readr)
library(dplyr)
library(ggplot2)
library(shadowtext)

options(scipen=999)

# Read data --------------------------------------------------------------------
output_files <- paste0("../simulation/output/", list.files("../simulation/output/"))
data_all <- bind_rows(lapply(output_files, read_csv))

# Mutate data ------------------------------------------------------------------
right_trimmed_mean <- function(x, p) {
  upper_bound <- quantile(x, 1 - p, na.rm = TRUE)
  x <- x[x <= upper_bound]
  mean(x, na.rm = TRUE)
}

data_summary <- data_all %>%
  filter(size != 2000) %>% # drop n =2000, only for SJ
  group_by(
    pdf, estimator, bw, size
  ) %>%
  summarise(
    time_mean = right_trimmed_mean(time, 0.02),
    time_top98 = quantile(time, 0.98, na.rm = TRUE),
    time_l94 = quantile(time, 0.03, na.rm = TRUE),
    time_u94 = quantile(time, 0.97, na.rm = TRUE),
    error_mean = right_trimmed_mean(error, 0.02),
    error_top98 = quantile(error, 0.98, na.rm = TRUE),
    error_l94 = quantile(error, 0.03, na.rm = TRUE),
    error_u94 = quantile(error, 0.97, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  group_by(
    pdf, size
  ) %>%
  mutate(
    time_mean_as_prop = round(time_mean / min(time_mean), 2),
    error_mean_as_prop = round(error_mean / min(error_mean), 2),
    time_fill = (time_mean - min(time_mean)) / (max(time_mean) - min(time_mean)),
    error_fill = (error_mean - min(error_mean)) / (max(error_mean) - min(error_mean))
  ) %>%
  ungroup() %>%
  mutate(
    size = as.factor(size),
    estimator = factor(
      estimator,
      levels = c("fixed_gaussian", "adaptive_gaussian", "mixture"),
      labels = c("Fixed Gaussian", "Adaptive Gaussian", "Mixture")
    ),
    bw = factor(
      bw,
      levels = c("scott", "silverman", "isj", "experimental", "sj", "mixture"),
      labels = c("Scott's", "Silverman's", "ISJ", "Experimental", "SJ", "Mixture")
    ),
    time_text = glue::glue("{round(time_mean, 4)} ({round(time_mean_as_prop, 4)})\n[{round(time_l94, 4)}, {round(time_u94, 4)}]"),
    error_text = glue::glue("{round(error_mean, 4)} ({round(error_mean_as_prop, 4)})\n[{round(error_l94, 4)}, {round(error_u94, 4)}]")
  )

data_summary_time <- data_summary %>%
  select(
    pdf,
    estimator,
    bw,
    size,
    time_mean,
    time_fill,
    time_text
  )

data_summary_error <- data_summary %>%
  select(
    pdf,
    estimator,
    bw,
    size,
    error_mean,
    error_fill,
    error_text
  )

# Plot function ----------------------------------------------------------------
heatmap <- function(df, x, y, fill, label, facet, title) {
  x <- enquo(x)
  y <- enquo(y)
  fill <- enquo(fill)
  label <- enquo(label)
  facet <- enquo(facet)
  
  ggplot(df, 
         aes(
           x = !!x, 
           y = !!y
         )
  ) + 
    geom_tile(
      aes(
        fill = !!fill
      ), 
      color = "white",
      size = 1.1
    ) + 
    geom_shadowtext(
      aes(
        label = !!label
      ),
      size = 4.8
    ) +
    labs(
      x = "",
      y = "",
      title = title
    ) +
    facet_grid(
      cols = vars(!!facet), 
      space = "free", 
      scale = "free"
    ) + 
    scale_fill_viridis_c(begin = 0.05, end = 0.95) +
    theme_minimal() + 
    theme(
      panel.grid.major = element_blank(),
      axis.text = element_text(size = 14),
      legend.title = element_blank(),
      legend.position = "none",
      strip.text.x = element_text(size = 16)
    )
}

pdf_list <- list(
  "gaussian_1" = "N(0, 1)",
  "gaussian_2" = "N(0, 2)",
  "gmixture_1" = "0.67 N(0, 1) + 0.33 N(0, 0.1)",
  "gmixture_2" = "0.5 N(-12, 0.5) + 0.5 N(12, 0.5)",
  "gmixture_3" = "0.5 N(0, 0.1) + 0.5 N(5, 1)",
  "gmixture_4" = "0.75 N(0, 1) + 0.25 N(1.5, 0.33)",
  "gmixture_5" = "0.6 N(3.5, 0.5) + 0.4 N(9, 1.5)",
  "gamma_1" = "Ga(k=1, theta=1)",
  "gamma_2" = "Ga(k=2, theta=1)",
  "logn_1" = "LogN(0, 1)",
  "beta_1" = "Beta(a=2.5, b=1.5)",
  "skwd_mixture1" = "0.7Ga(1.5, 1.0) + 0.2N(5,1) + 0.1N(8, 0.75)"
)

get_time_heatmaps <- function(df, pdf_names = pdf_names, exclude_sj = TRUE) {
  plt_list <- list()
  pdf_names <- names(pdf_list)
  for (pdf_name in pdf_names) {
    .df <- filter(df, pdf == pdf_name)
    if (exclude_sj) .df <- filter(.df, bw != "SJ")
    .title <- pdf_list[[pdf_name]]
    plt_list[[pdf_name]] <- heatmap(.df, bw, size, time_fill, 
                                    time_text, estimator, .title)
  }
  plt_list
}

get_error_heatmaps <- function(df, pdf_names = pdf_names, exclude_sj = TRUE) {
  plt_list <- list()
  pdf_names <- names(pdf_list)
  for (pdf_name in pdf_names) {
    .df <- filter(df, pdf == pdf_name)
    if (exclude_sj) .df <- filter(.df, bw != "SJ")
    .title <- pdf_list[[pdf_name]]
    plt_list[[pdf_name]] <- heatmap(.df, bw, size, error_fill, 
                                    error_text, estimator, .title)
  }
  plt_list
}


time_heatmaps_no_sj <- get_time_heatmaps(data_summary_time)
time_heatmaps <- get_time_heatmaps(data_summary_time, exclude_sj = FALSE)

error_heatmaps_no_sj <- get_error_heatmaps(data_summary_error)
error_heatmaps <- get_error_heatmaps(data_summary_error, exclude_sj = FALSE)

filenames <- paste0("app/data/heatmaps/", paste0("time_", names(time_heatmaps)), ".png")
purrr::pwalk(
  list(filenames, time_heatmaps),
  ggsave,
  device = "png",
  width = 48,
  height = 24,
  units = "cm"
)

filenames <- paste0("app/data/heatmaps/", paste0("time_", names(time_heatmaps_no_sj)), "_no_sj.png")
purrr::pwalk(
  list(filenames, time_heatmaps_no_sj),
  ggsave,
  device = "png",
  width = 48,
  height = 24,
  units = "cm"
)

filenames <- paste0("app/data/heatmaps/", paste0("error_", names(error_heatmaps)), ".png")
purrr::pwalk(
  list(filenames, error_heatmaps),
  ggsave,
  device = "png",
  width = 48,
  height = 24,
  units = "cm"
)

filenames <- paste0("app/data/heatmaps/", paste0("error_", names(error_heatmaps_no_sj)), "_no_sj.png")
purrr::pwalk(
  list(filenames, error_heatmaps_no_sj),
  ggsave,
  device = "png",
  width = 48,
  height = 24,
  units = "cm"
)

# Copy files to app directory
# current_folder <- "heatmaps"
# new_folder <- "app/data/heatmaps"
# files_to_copy <- list.files(current_folder)
# file.copy(file.path(current_folder, files_to_copy), new_folder, overwrite = TRUE)

# TODO: De alguna forma incorporar la razon entre tiempos para el mismo tamaño muestral,
#       pero no eliminar la razon actual... o buscarle la vuelta para que se puedan hacer
#       las siguientes comparaciones de manera directa:
#       * Tiempo para tamaño X / Tiemmpo para tamaño mas bajo, dado un método de estimacion
#       * Tiempo para tamaño X, metodo 1 / Tiempo para tamaño X, metodo 2.