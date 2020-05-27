library("reticulate")
src <- reticulate::import_from_path("src", "../")

size <- 100
rvs <- c(rnorm(round(size / 2), -12, 0.5), 
         rnorm(round(size / 2), 12, 0.5))

l <- src$estimate_density(rvs, bw = "isj", bw_return = TRUE)
plot(l[[1]], l[[2]], type = "l")
l[[3]]


l <- src$estimate_density(rnorm(300), bw = "isj", bw_return = TRUE)
plot(l[[1]], l[[2]], type = "l")
l[[3]]

# size <- 50
# rvs <- c(rnorm(round(size / 2), 0, 0.1), 
#          rnorm(round(size / 2), 5, 1))
# 
# l <- src$estimate_density(rvs, bw = "isj", bw_return = TRUE)
# plot(l[[1]], l[[2]], type = "l")
# 
# l[[3]]

