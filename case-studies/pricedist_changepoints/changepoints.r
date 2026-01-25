
library(tidyverse)
library(magrittr)

set.seed(123)

# Define segments
segments <- data.frame(
  start_day = c(1, 50)*2,
  list_price = c(5, 10),
  discount_price = c(4, 4), # 8.5),
  discount_probability = c(.1, .1)
)

days <- sample(1:200, replace = TRUE) #1:(100*2) #150 #
segment_id <- findInterval(days, segments$start_day)

list_price <- segments$list_price[segment_id]
discount_price <- segments$discount_price[segment_id]
discount_probability <- segments$discount_probability[segment_id]

u <- runif(length(days))
price <- ifelse(u < discount_probability, discount_price, list_price)
df <- data.frame( day = days, price = price )
df %<>% arrange(day) 

df$qty <- 1
df %<>% group_by(day, price) %>% summarize( qty = sum(qty) )
# to-do: sort columns by price point
df %<>% pivot_wider(names_from = price, values_from = qty, values_fill = 0)

quantities <- as.matrix(df[,-1])
prices <- colnames(quantities) %>% as.numeric()
quantities %<>% .[,order(prices)]
prices %<>% sort() 


data_list <- list(
  n_time_points = nrow(quantities),
  n_price_points = ncol(quantities),
  histogram = quantities,
  price_points = prices
)


library(cmdstanr)

# Compile the model
model <- cmdstanr::cmdstan_model("./changepoints_v1.stan") #, force_recompile = TRUE)

# 
# fit <- model$sample(
#   data = data_list,
#   chains = 4,
#   parallel_chains = 4,
#   iter_warmup = 500,
#   iter_sampling = 1000,
#   seed = 123
# )
# fit

opt <- model$optimize(
  data = data_list,
  #chains = 4,
  #parallel_chains = 4,
  #iter_warmup = 500,
  #iter_sampling = 1000,
  seed = 123
)

opt

est <- opt$summary("cp_probs")
plot( df$day, c(NA, est$estimate) )
est %>% arrange(desc(estimate))

mat <- opt$summary("lp_change_prior") #%>% arrange(variable)
matrix(mat$estimate, nrow = 6, ncol = 6)

matx <- opt$summary("change_magnitudex") #%>% arrange(variable)
matx <- array(matx$estimate, dim = c(6, 6, 6))



fit <- model$sample(
  data = data_list,
  chains = 1,
  iter_warmup = 1,
  iter_sampling = 1,
  seed = 123
)

#param_names <- fit$metadata()$model_params %>% setdiff("lp__")
par_constrained <- list(
  cp_probs_raw = rep(0.5, data_list$n_time_points),
  p_change = 0.5
)
par_unconstrained <- fit$unconstrain_variables( par_constrained )
#par_unconstrained

fit$log_prob( par_unconstrained )
# [1] -2.772589

print(logp)


opt <- model$optimize(
  data = data_list,
  #chains = 4,
  #parallel_chains = 4,
  #iter_warmup = 500,
  #iter_sampling = 1000,
  seed = 123
)

opt$summary("prior_p_zero")

est <- opt$summary("cp_probs_raw")
plot( df$day, est$estimate )
est


fit <- model$sample(
  data = data_list,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 500,
  iter_sampling = 1000,
  seed = 123
)
fit$summary("cp_probs_raw[48]")


# ## Parameters
# # 3 hidden states
# K <- 2
# 
# # Initialize means spread across observed price range
# mu <- seq(min(price), max(price), length.out = K)
# sigma <- rep(1.0, K)  # Reasonable spread
# 
# # Uniform transition matrix
# p_state <- matrix(1 / K, K, K)
# p_outlier <- 0.1

prices_qty <- as.matrix(df[,-1])
prices <- colnames(prices_qty) %>% as.numeric()


library(Rcpp)
sourceCpp("changepoint_filter_rcpp_v1.cpp")
#sourceCpp("changepoint_filter_rcpp_similarity_v1.cpp")

#quantities <- prices_qty
(log_lik <- compute_segment_loglik( quantities ))
cp_probs_init <- rep(0.5, nrow(quantities) - 1)
(marginal_ll <- marginal_loglik(cp_probs_init, log_lik))
# [1] -0.4700036

marginal_loglik_wrapper <- function(par, log_lik)
{
  prior_p_zero = plogis(par[1])
  prior_p_one = 1 - prior_p_zero
  cp_probs <- plogis(par[-1]) # 
  
  loglik <- marginal_loglik(cp_probs, log_lik)
  
  log_prior <- log( cp_probs * prior_p_one + (1 - cp_probs) * prior_p_zero )

  loglik + sum(log_prior)
}

cp_probs_init <- rep(0.1, nrow(quantities) - 1)
par_init <- c(qlogis(0.5), qlogis(cp_probs_init)) #  
marginal_loglik_wrapper( par_init, log_lik)

opt <- optim( par_init, marginal_loglik_wrapper, log_lik = log_lik, method = "CG", control = list(maxit = 50000, fnscale = -1) ) # 
opt

plot( plogis(opt$par[-1]) )
plot( plogis(par_init) )



p_change = 0.1

prior_p_zero = 0.9
prior_p_one = 1 - prior_p_zero
p_change * prior_p_one + (1 - p_change) * prior_p_zero


??dbernoulli

# log_lik <- compute_segment_loglik( quantities )

init <- rep(0.0, 149)
marginal_loglik(init, log_lik)

lls <- sapply(seq_along(init), function(i) {
  par <- init
  par[50] = 1; 
  par[99] = 1;
  par[i] <- 1
  marginal_loglik(par, log_lik)
})

plot( lls - min(lls) )
which.max( lls )


##

# Example:
alpha <- rep(1, ncol(quantities))
bf_vec <- log_bf_array_cpp(quantities, alpha)
log_bf_array <- array(bf_vec, dim = c(nrow(quantities), nrow(quantities), nrow(quantities)))

log_bf_array[98:101, 98:101, 98:101]

##


# log_prior <- function(x, spike_loc, spike_scale, slab_loc, slab_scale, pi = 0.05) {
#   spike <- dnorm(x, mean = spike_loc, sd = spike_scale, log = FALSE)
#   slab  <- dnorm(x, mean = slab_loc,  sd = slab_scale,  log = FALSE)
#   log(pi * slab + (1 - pi) * spike)
# }

log_beta_prior <- function(p, pi = 0.05,
                      spike_a = 0.7, spike_b = 8,
                      slab_a = 8,  slab_b = 0.7)
{
    spike <- dbeta(p, spike_a, spike_b, log = FALSE)
    slab  <- dbeta(p, slab_a, slab_b, log = FALSE)
    log(pi * slab + (1 - pi) * spike)
}

marginal_loglik_wrapper <- function(par)
{
  #prior_pi <- plogis(par[1])
  prior_pi <- 0.05 

  cp_probs_logit <- par #[-1]
  cp_probs <- plogis(cp_probs_logit)
  ll <- marginal_loglik(cp_probs, log_lik)
  
  log_prior_terms <- sapply(cp_probs, \(x) log_beta_prior(x, pi = prior_pi ))
  log_prior <- sum(log_prior_terms)  
  
  return(ll+ log_prior ) #
}

par_init <- c( qlogis(runif(99)) ) # qlogis(.1),  
marginal_loglik_wrapper( par_init )
#marginal_loglik_wrapper( c() )

  
opt <- optim( par_init, marginal_loglik_wrapper, control = list(maxit = 10000, fnscale = -1) ) # 
opt

plot( plogis(opt$par) )
plot( plogis(par_init) )


# Helper function to compute log-sum-exp in a numerically stable way
log_sum_exp <- function(log_vals) {
  max_log <- max(log_vals)
  max_log + log(sum(exp(log_vals - max_log)))
}

marginal_loglik <- function(cp_probs, log_lik)
{
  # cp_probs: numeric vector of length n_time_points - 1 (P(changepoint after day t))
  # log_lik: n_time_points x n_time_points matrix, where log_lik[s, t] = log-likelihood of segment from day s to t (1-based)
  n_time_points <- nrow(log_lik)
  stopifnot(length(cp_probs) == n_time_points - 1)
  
  # clip probabilities to be strictly within the (0, 1) interval
  cp_probs <- pmin(pmax(cp_probs, 1e-12), 1 - 1e-12)
  log_cp_probs <- log(cp_probs)
  log_1m_cp_probs <- log1p(-cp_probs)

  path_likelihood <- function(marginal_loglik, s, t)
  {
      cur_lp_cp <- if (s == 1)  { 0 } else { log_cp_probs[s - 1] }
      lp_no_cp_s_to_tm1 <- if (s == t) { 0 } else { sum(log_1m_cp_probs[s:(t - 1)])  }
      ll_segment_s_to_t <- log_lik[s, t]
      
      marginal_loglik[s] +   # total log-likelihood up to day (s - 1)
                 cur_lp_cp + # probability of a changepoint *before* day s
         lp_no_cp_s_to_tm1 + # probability of *no* changepoints between s and t
         ll_segment_s_to_t   # likelihood of the segment [s, t]
  }

  accumulate_path_likelihoods <- function(marginal_loglik, t) {
      path_loglikelihoods <- sapply(1:t, \(s) path_likelihood( marginal_loglik, s, t ) )
      path_loglikelihoods
  }
  
  marginal_loglik <- rep(-Inf, n_time_points + 1)
  marginal_loglik[1] <- 0  # log-prob of empty sequence
  
  for (t in 1:n_time_points) {
      path_loglikelihoods <- accumulate_path_likelihoods( marginal_loglik, t )
      marginal_loglik[t + 1] <- log_sum_exp( path_loglikelihoods )
  }
  
  return(marginal_loglik[n_time_points + 1])
}

# Helper function for numerically stable log-sum-exp
log_sum_exp <- function(log_vals) {
  max_log <- max(log_vals)
  max_log + log(sum(exp(log_vals - max_log)))
}

marginal_logprior <- function(cp_probs, max_segments = NULL) {
  n <- length(cp_probs) + 1  # number of time points
  log_cp <- log(cp_probs)
  log_ncp <- log1p(-cp_probs)
  
  log_prior <- rep(-Inf, n + 1)  # log_prior[t] = log P(y_{1:t})
  log_prior[1] <- 0  # P(empty sequence) = 1
  
  for (t in 1:n) {
    for (s in 1:t) {
      # Segment from s to t (inclusive), last changepoint at s-1
      if (s == 1) {
        log_p_segment <- if (t > 1) sum(log_ncp[s:(t - 1)]) else 0
      } else {
        log_p_segment <- log_cp[s - 1] + if (s < t) sum(log_ncp[s:(t - 1)]) else 0
      }
      
      a <- log_prior[t + 1]
      b <- log_prior[s] + log_p_segment
      
      if (is.na(a) || is.na(b)) {
        stop(sprintf("NA detected at t=%d, s=%d: a=%f, b=%f", t, s, a, b))
      }
      
      log_prior[t + 1] <- logspace_add(a, b)
    }
  }
  
  return(log_prior[n + 1])
}
logspace_add <- function(a, b) {
  if (is.infinite(a) && a == -Inf) return(b)
  if (is.infinite(b) && b == -Inf) return(a)
  m <- max(a, b)
  return(m + log1p(exp(-abs(a - b))))
}

marginal_loglik_wrapper <- function(cp_probs_logit, log_lik) {
  cp_probs <- plogis(cp_probs_logit)
  ll <- marginal_loglik(cp_probs, log_lik)
  log_prior <- dnorm(cp_probs_logit, mean = -5, sd = 5, log = T) %>% sum()
  return(ll+log_prior)
}



#quantity_matrix
quantity <- prices_qty
log_lik <- compute_segment_loglik(prices_qty)

# switch: 70

marginal_loglik( c( rep(0,19) ), log_lik )

marginal_loglik( c( rep(0,4), 1, rep(0,14) ), log_lik )
# -2.502012

marginal_loglik( c( rep(0,40), 1, rep(0,58) ), log_lik )
# -2.870814

start_cp_probs <- qlogis(rep(0.5, nrow(log_lik)-1))
opt <- optim(start_cp_probs, marginal_loglik_wrapper, log_lik = log_lik, control = list(maxit = 20000, fnscale = -1)) #, method = "SANN"
opt


plot(plogis(opt$par))

