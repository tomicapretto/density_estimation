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

