
# --------------------


```{r eval=FALSE}
model <- pscl::zeroinfl(sales ~ 1 | 1, data = df, dist = "negbin")
summary(model)
```

```{r eval=FALSE}
mu_hat <- exp(coef(model)["count_(Intercept)"])
theta_hat <- model$theta  # the NB shape parameter

epsilon <- 10e-10
df$loglik_sales_stocked  <- dnbinom(df$sales, size = theta_hat, mu = mu_hat, log = T)
df$loglik_sales_stockout <- ifelse(df$sales == 0, log(1), log(0+epsilon))
df$loglik_ratio_stockout <- df$loglik_sales_stocked - df$loglik_sales_stockout
# df

create_weights <- function(window_len, w_center = 0.5, taper = 0.5) {
  stopifnot(window_len %% 2 == 1)
  K <- (window_len - 1) / 2
  side_mass <- (1 - w_center) / 2
  
  # Geometric taper: w1 * (1 + r + r^2 + ... + r^(K-1)) = side_mass
  r <- taper
  denom <- (1 - r^K) / (1 - r)
  w1 <- side_mass / denom
  side_weights <- w1 * r^(0:(K-1))
  
  c(rev(side_weights), w_center, side_weights)
}

create_weights(7, w_center = 0.5, taper = 0.5)




library(zoo)
#weights <- c(0.1, 0.15, 0.5, 0.15, 0.1)
#weights <- c(0.25, 0.5, 0.25)
weights <- c(1, 2, 6, 2, 1) %>% { . / sum(.) }

x_smooth <- rollapply(
  df$loglik_ratio_stockout,
  width = length(weights),
  align = "center",
  fill = NA,
  FUN = function(y) sum(y * weights)
)


df %>% ggplot(aes(day, x_smooth, color = state)) + geom_point() + theme_bw()

```

```{r eval=FALSE}
#df <- data.frame(day = 1:n_days, sales = sales, loglik_nb = loglik_sales_stocked, loglik_stockout = ifelse(sales == 0, log(1), log(0)))
#df
```

```{r eval=FALSE}
# Initialize
n <- nrow(df)
logalpha <- matrix(NA, nrow = n, ncol = 2)  # [time, state]

# State order: normal = 1, stockout = 2
init_prob <- c(0.95, 0.05)
trans <- matrix(c(0.98, 0.02,
                  0.10, 0.90), nrow = 2, byrow = TRUE)

# Log scale for stability
log_emis <- cbind(df$loglik_nb,
                  df$loglik_stockout)

# Forward pass
logalpha[1, ] <- log(init_prob) + log_emis[1, ]

for (t in 2:n) {
  for (j in 1:2) {
    logalpha[t, j] <- log(sum(exp(logalpha[t-1, ] + log(trans[, j])))) + log_emis[t, j]
  }
}

# Normalize to get filtered P(state | obs)
logscale <- apply(logalpha, 1, function(x) log(sum(exp(x))))
gamma <- exp(logalpha - logscale)

df$P_normal <- gamma[, 1]
df$P_stockout <- gamma[, 2]

df$log_ratio_stockout <- df$P_stockout / df$P_normal
#View(df)

```

```{r eval=FALSE}
df$stockout <- FALSE
df$stockout[unlist(stockouts)] <- TRUE 
df %>% ggplot(aes(day, log_ratio_stockout, color = stockout)) + geom_point() + theme_bw()
```

# --------------------

