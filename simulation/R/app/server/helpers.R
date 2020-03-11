# Data wrangling functions -----------------------------------------------------
df_filter <- function(df, .pdf, .estimator, .bw, .size) {
  df %>%
    filter(
      pdf %in% .pdf,
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

deduce_scale <- function(x) {
  if ((quantile(x, 0.75, na.rm = TRUE)) > 1) {
    return("sec") 
  } else {
    return("ms")
  }
}

# Specify accuracy by default
precision <- scales:::precision

# Plotting functions -----------------------------------------------------------
initialize_plot <- function(df, .metric) {
  .metric <- as.name(.metric)
  ggplot(
    df,
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
  # theme_bw() + 
  theme_gray() + 
    theme(
      # panel.grid.minor = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.border = element_blank(), 
      legend.position = "none",
      strip.background = element_blank()
    )
}

custom_scale <- function(limits = NULL,  breaks = waiver(),
                         scale = "ms", acc = 0.1, log10 = FALSE) {
  scale_func <- if (log10) scale_y_log10 else scale_y_continuous
  scale_func(
    limits = limits,
    breaks = breaks,
    labels = get_label(scale, acc)
  )
}

custom_facet <- function(..., free_y, nrow = 1) {
  
  args <- as.list(...)
  
  
  if (!any(args %in% c("bw", "estimator", "pdf"))) return(NULL)
  
  scales <- if (free_y) "free_y" else "fixed"
  
  if (length(args) == 1) {
    col_labs <- lbls_list[[args[[1]]]]
    
    form <- as.formula(paste(".", args[[1]], sep = "~"))
    facet_grid(form, scales = scales, 
               labeller = labeller(.cols = col_labs))
    
  } else if (length(args) == 2) {
    row_labs <- lbls_list[[args[[1]]]]
    col_labs <- lbls_list[[args[[2]]]]
    
    form <- as.formula(paste(args[[1]], args[[2]], sep = "~"))
    facet_grid(form, scales = scales, 
               labeller = labeller(.cols = col_labs, .rows = row_labs))
  }
}

# Misc -------------------------------------------------------------------------
get_bw_choices <- function(x) {
  
  # Old: nice, but drop names.
  # f <- function (x) switch(x, "mixture" = choices_bw_mixture, choices_bw_classic)
  # l <- lapply(x, f)
  # Reduce(union, l)
  
  # New:
  f <- function (x) switch(x, "mixture" = choices_bw_mixture, choices_bw_classic)
  vec <- unlist(lapply(x, f)) 
  vec[!duplicated(vec)]
}

get_size_choices <- function(x) {
  f <- function (x) switch(x, "sj" = choices_size_sj, choices_size_default)
  l <- lapply(x, f)
  Reduce(union, l)
}

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
         sec = seconds(acc),
         scales::number_format(acc))
}

