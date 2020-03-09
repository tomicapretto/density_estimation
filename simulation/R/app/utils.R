library(dplyr)
library(ggplot2)

# Colors -----------------------------------------------------------------------
DARK_GRAY <- "#2d3436"

# Data wrangling utils ---------------------------------------------------------
df_filter <- function(df, .estimator, .bw, .size) {
  df %>%
    filter(
      estimator %in% .estimator,
      bw %in% .bw,
      size %in% .size
    )
}

df_trim <- function(df, .group_vars, .trim_var, .quantile) {

  .group_vars <- purrr::map(.group_vars, as.name)
  .trim_var <- as.name(.trim_var)
  
  df %>%
    group_by(!!!.group_vars) %>%
    mutate(
      upper_bound = quantile(!!.trim_var, .quantile, na.rm = TRUE)
    ) %>%
    mutate(
      !!.trim_var := ifelse(!!.trim_var >= upper_bound, NA, !!.trim_var)
    ) %>%
    select(-upper_bound)
}

compute_limits <- function(x) {
  x <- x[!is.na(x)]
  x_iqr <- IQR(x)
  lower <- min(x) - 0.01 * x_iqr
  upper <- median(x) + 2.5 * x_iqr
  c(lower, upper)
}

deduce_scale <- function(x) {
  if ((median(x, na.rm = TRUE) / 1000) > 1) {
    return("sec") 
  } else {
    return("ms")
  }
}

# Specify accuracy by default
precision <- scales:::precision

# Plotting uitls ---------------------------------------------------------------
init_plot <- function(df, .metric) {
  .metric <- as.name(.metric)
  ggplot(df,
    aes(
      x = factor(size),
      y = !! .metric, 
      fill = factor(size)
    )
  )
}

add_boxplot <- function(outlier.shape = 19) {
  geom_boxplot(
    outlier.fill = alpha(DARK_GRAY, 0.3),
    outlier.color = alpha(DARK_GRAY, 0.3),
    outlier.shape = outlier.shape
  )
}

custom_fill <- function(alpha = 0.6) {
  scale_fill_viridis_d(alpha = alpha)
}

custom_theme <- function() {
  theme_bw() + 
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.border = element_blank(), 
    legend.position = "none"
  )
}

custom_scale <- function(limits = NULL,  breaks = waiver(),
                         scale = "ms", acc = 1, log10 = FALSE) {
  scale_func <- if (log10) scale_y_log10 else scale_y_continuous
  scale_func(
    limits = limits,
    breaks = breaks,
    labels = get_label(scale, acc)
  )
}

custom_facet <- function(..., nrow = 1) {
  args <- list(...)
  if (length(args) == 0) {
    return(NULL)
  } else if (length(args) == 1) {
    facet_wrap(c(args[[1]]), nrow = nrow)
  } else if (length(args) == 2) {
    facet_wrap(c(args[[1]], args[[2]]))
  }
}

# Helpers ----------------------------------------------------------------------
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

get_label <- function(scale, acc) {
  switch(scale, 
         ms = miliseconds(acc),
         sec = seconds(acc))
}


# Playground -------------------------------------------------------------------

# estimator <- "fixed_gaussian"
# bw <- "silverman"
# toy_data <- data_clean %>%
#   filter(
#     estimator == !!estimator,
#     bw == !! bw
#   )
# 
