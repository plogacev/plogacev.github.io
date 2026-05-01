library(tidyverse)
library(magrittr)

generate_transaction_prices <- function(segments, n_days, lambda_qty, seed = 123)
{
    set.seed(seed)
    days <- 1:n_days
    
    # Map each day to its corresponding pricing regime
    segment_id <- findInterval(days, segments$start_day)
    
    # Generate a synthetic transactional record with total quantity and price assignment
    df <- data.frame(
        day = days, 
        list_price = segments$list_price[segment_id],
        discount_price = segments$discount_price[segment_id],
        discount_percentage = segments$discount_percentage[segment_id],
        qty_total = rpois(n_days, lambda_qty)
    ) %>% 
    mutate(
          qty_discount = rbinom(n_days, qty_total, discount_percentage),
          qty_list_price = qty_total - qty_discount
    ) %>% 
    select(-discount_percentage, -qty_total) %>%
    pivot_longer(
          cols = starts_with("qty_"),
          names_to = "type",
          values_to = "qty"
    ) %>%
    mutate(
        price = if_else(type == "qty_discount", discount_price, list_price)
    ) %>%
    select(day, price, qty)
    
    df
}

compute_price_histogram <- function(df)
{
    # Aggregate total daily quantities and remove days with no transactions.
    df_nonzero <- df %>%
        mutate( total_qty = sum(qty), .by = "day" ) %>%
        filter(total_qty > 0) %>%
        select(-total_qty)
    
    # Reshape the cleaned transactional data into a price-by-day matrix format.
    # Each row corresponds to a day; each column to a discretized price point.
    df_wide <- df_nonzero %>% 
        pivot_wider(names_from = "price", values_from = "qty", values_fill = 0)
    
    # Convert to a numeric matrix and enforce price column ordering.
    histogram_qty <- as.matrix(df_wide[,-1])
    price_points <- as.numeric(colnames(histogram_qty))
    histogram_qty <- histogram_qty[,order(price_points)]
    price_points %<>% sort()
    
    list(df = df, 
         histogram = histogram_qty, 
         price_points = price_points)
}
