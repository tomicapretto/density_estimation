library("reticulate")
src <- import_from_path(module = "src", path = "../")

# Helps to have Latex math in the html file.
pdf2png <- function(path) {
  # only do the conversion for non-LaTeX output
  if (knitr::is_latex_output()) 
    return(path)
  path2 <- xfun::with_ext(path, "png")
  img <- magick::image_read_pdf(path)
  magick::image_write(img, path2, format = "png")
  path2
}

cont_dist_bounds <- list(
  "norm" = function(.params) {
    width <- 3 * .params[[2]]
    c(.params[[1]] - width, .params[[1]] + width)
  },
  "t" = function(.params) {
    qt(c(0.005, 0.995), .params[[1]], .params[[2]])
  },
  "gamma" = function(.params) {
    c(0, qgamma(0.995, .params[[1]], .params[[2]]))
  },
  "exp" = function(.params) {
    c(0, qexp(0.995, .params[[1]]))
  },
  "beta" = function(.params) {
    c(0, 1)
  },
  "lnorm" = function(.params) {
    c(0, qlnorm(0.995, .params[[1]], .params[[2]]))
  },
  "weibull" = function(.params) {
    c(0, qweibull(0.995, .params[[1]], .params[[2]]))
  },
  "unif" = function(.params) {
    c(.params[[1]], .params[[2]])
  }
)

component_pdf <- function(distribution, .params, x_grid) {
  .f <- paste0("d", distribution)
  .args <- c(list(x_grid), .params)
  do.call(.f, .args)
}

component_rvs <- function(distribution, .params, size) {
  .f <- paste0("r", distribution)
  .args <- c(list(size), .params)
  do.call(.f, .args)
}

mixture_rvs <- function(distributions, .params, size, wts = NULL) {
  if (is.null(wts)) wts <- rep(1 / length(distributions), length(distributions))
  stopifnot(length(distributions) == length(.params), length(.params) == length(wts))
  stopifnot(all.equal(sum(wts), 1, tolerance = 0.011))
  
  .l <- list(distributions, .params, round(wts * size))
  unlist(purrr::pmap(.l, component_rvs))
}

pdf_bounds <- function(distribution, .params) {
  .f <- cont_dist_bounds[[distribution]]
  .f(.params)
}

mixture_grid <- function(distributions, .params) {
  .l <- list(distributions, .params)
  out <- unlist(purrr::pmap(.l, pdf_bounds))
  seq(min(out), max(out), length.out = 500)
}

mixture_pdf <- function(distributions, .params, wts = NULL) {
  x_grid <- mixture_grid(distributions, .params)
  
  .l <- list(distributions, .params)
  pdf <- unlist(purrr::pmap(.l, component_pdf, x_grid = x_grid))
  pdf <- as.vector(matrix(pdf, ncol = length(wts)) %*% wts)
  pdf[is.infinite(pdf)] <- NA # In some edge cases pdf is `Inf`
  return(list("x" = x_grid, "pdf" = pdf))
}

dists <- list(
  "gaussian_1" = "norm",
  "gaussian_2" = "norm",
  "gmixture_1" = c("norm", "norm"),
  "gmixture_2" = c("norm", "norm"),
  "gmixture_3" = c("norm", "norm"),
  "gmixture_4" = c("norm", "norm"),
  "gmixture_5" = c("norm", "norm"),
  "gamma_1" = "gamma",
  "gamma_2" = "gamma",
  "logn_1" = "lnorm",
  "beta_1" = "beta",
  "skwd_mixture1" = c("gamma", "norm", "norm")
)

grids <- list(
  seq(-3, 3, length.out = 1000),
  seq(-6, 6, length.out = 1000),
  seq(-3, 3, length.out = 1000),
  seq(-14, 14, length.out = 1000),
  seq(-1, 9, length.out = 1000),
  seq(-3, 3, length.out = 1000),
  seq(1, 14, length.out = 1000),
  seq(0, 8, length.out = 1000),
  seq(0, 8, length.out = 1000),
  seq(0, 8, length.out = 1000),
  seq(0, 1, length.out = 1000),
  seq(0, 10, length.out = 1000)
)

params <- list(
  list(c(0, 1)),
  list(c(0, 2)),
  list(c(0, 1), c(0, 0.1)),
  list(c(-12, 0.5), c(12, 0.5)),
  list(c(0, 0.1), c(5, 1)),
  list(c(0, 1), c(1.5, 0.33)),
  list(c(3.5, 0.5), c(9, 1.5)),
  list(c(1, 1)),
  list(c(2, 1)),
  list(c(0, 1)),
  list(c(2.5, 1.5)),
  list(c(1.5, 1), c(5, 1), c(8, 0.75))
)

wts <- list(
  1,
  1,
  c(0.667, 0.333),
  c(0.5, 0.5),
  c(0.5, 0.5),
  c(0.75, 0.25),
  c(0.6, 0.4),
  1, 
  1,
  1, 
  1,
  c(0.7, 0.2, 0.1)
)

titles <- c(
  "N(0,1)", "N(0, 2)", 
  "\\frac{2}{3} N(0, 1) + \\frac{1}{3} N(0, 0.1)",
  "\\frac{1}{2} N(-12, \\frac{1}{2}) + \\frac{1}{2}N(12, \\frac{1}{2})", 
  "\\frac{1}{2} N(0, \\frac{1}{10}) + \\frac{1}{2} N(5, 1)",
  "\\frac{3}{4} N(0, 1) + \\frac{1}{4} N(1.5, \\frac{1}{3})",
  "\\frac{3}{5} N(3.5, \\frac{1}{2}) + \\frac{2}{5} N(9, 1.5)",
  "\\Gamma (k = 1, \\theta = 1)",
  "\\Gamma (k = 2, \\theta = 1)",
  "LogN(0, 1)",
  "\\beta (a = 2.5, b = 1.5)",
  "\\frac{7}{10}\\Gamma(k = 1.5, \\theta = 1) + \\frac{2}{10}N(5, 1) + \\frac{1}{10}N(8, \\frac{3}{4})"
)
titles <- paste0("$", titles, "$")

names(titles) <- names(grids) <- names(params) <- names(wts) <- names(dists)

get_plot <- function(name) {
  .dists <- dists[[name]]
  .params <- params[[name]]
  .wts <- wts[[name]]
  .ttl <- titles[[name]]
  
  out <- mixture_pdf(.dists, .params, .wts)
  
  plot(out$x, out$pdf, type = "l", main = .ttl,
       xlab = '$x$', ylab = '$f(x)$', col = "#2980b9", lwd = 3)
}

get_example <- function(name = "gmixture_2", bw = "isj", 
                        adaptive = TRUE, size = 1000, reps = 10) {
  
  .dists <- dists[[name]]
  .params <- params[[name]]
  .wts <- wts[[name]]
  .ttl <- titles[[name]]
  
  out <- mixture_pdf(.dists, .params, .wts)
  
  samples <- replicate(reps, 
                       do.call(mixture_rvs, list(.dists, .params, size, .wts)), 
                       simplify = FALSE)

  estimate <- purrr::map(samples, src$estimate_density, bw = bw, adaptive = adaptive)
  
  max_y <- max(out$pdf, max(sapply(estimate, function(x) max(x[[2]], na.rm = TRUE))))
  
  par(mar = c(4.5, 4, 1.5, .8), mgp=c(1.8, 0.6, 0))
  plot(out$x, out$pdf, type = "l", 
       lwd = 5, lty = "dashed",
       ylim = c(0, max_y * 1.01),
       xlab = '$x$', ylab = '$f(x)$')
  
  for (i in seq_along(estimate)) {
    lines(estimate[[i]][[1]], estimate[[i]][[2]], 
          lwd = 4, col = "#2980B966")
  }
}

#TODO: See why get_example("gmixture_4") fails so BAD.

