library(tidyverse)
library(magrittr)


locate_optimal_changepoints <- function( histogram, max_lambda = 0.2 ) { 
  
  histogram_cumulative <- create_histogram_cumulative(histogram)
  
  transform_log_lambda <- function(logit_lambda, max_lambda) {
    p_lambda <- plogis(logit_lambda)*max_lambda
    log(p_lambda)
  }
  
  loglik_changepoints_fn <- function(logit_lambda, max_lambda) {
    log_lambda <- transform_log_lambda(logit_lambda, max_lambda)
    res <- locate_changepoints(histogram_cumulative, log_lambda)
    res$loglik
  }
  
  res <- optimize(loglik_changepoints_fn, lower=-5, upper=5, max_lambda = max_lambda, maximum = TRUE)
  
  log_lambda <- transform_log_lambda(res$maximum, max_lambda = max_lambda)
  res <- locate_changepoints(histogram_cumulative, log_lambda)
  res$lambda <- exp(log_lambda)
  res
}
