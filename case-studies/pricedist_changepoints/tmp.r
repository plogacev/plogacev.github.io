
library(Rcpp)

sourceCpp("tmp.cpp")

set.seed(123)
histogram <- matrix(sample(0:10, 20, replace = TRUE), nrow = 5)
changepoints <- c(0, 0, 0, 0)  # example changepoint vector

# compute_path_loglik(histogram, changepoints)

locate_next_best_changepoint(histogram, changepoints)

compute_path_loglik(histogram, c(0, 0, 0, 0)) 
compute_path_loglik(histogram, c(1, 0, 0, 0)) 
compute_path_loglik(histogram, c(0, 1, 0, 0)) 
compute_path_loglik(histogram, c(0, 0, 1, 0))
