

rnorm_mix <- function(n, mean, sd, wt) {
  .rnorm <- function(m, s, n) rnorm(n, m, s)
  n <- round(n * wt)
  .l <- list(n, mean, sd) 
  unlist(purrr::pmap(.l, rnorm))
}



rfuncs <- list(
  "gaussian_1" = rnorm,
  "gaussian_2" = rnorm,
  "gmixture_1" = rnorm_mix,
  "gmixture_2" = rnorm_mix,
  "gmixture_3" = rnorm_mix,
  "gmixture_4" = rnorm_mix,
  "gmixture_5" = rnorm_mix,
  "gamma_1" = rgamma,
  "gamma_2" = rgamma,
  "beta_1" = rbeta,
  "logn_1" = rlnorm
)



# ggplot2::alpha("#2980b9", 0.4)

